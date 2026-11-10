"""Python side of the Tractor training bridge.

Spawns `node training/dist/tractor-env.mjs` (build it with `npm run train:build`)
and drives the vectorized env over stdin/stdout (JSON lines; batch replies add
the feature arrays as raw bytes after the line). The rules, the hidden-
information boundary, the encoder and the candidate lists all live in
TypeScript (src/ai/nn/); this side only ever sees encoded observations,
candidate features and rewards. Every seat is played from this side: the
learner's rows (role 0) and the opponent's (role 1), each a network the caller
runs.

    with TractorEnv() as env:
        batch = env.reset(num_envs=32, lineups=[lineup("team")])
        while training:
            choices = policy(batch)            # one candidate index per env row
            batch = env.step(choices)

A batch is packed. Row i's observation is its global block `glob[i]` plus
`seat_count[i]` seat blocks (its table's seats) at `seats[seat_offsets[i]:]`,
each with its slot in `seat_slot`; its candidates are `count[i]` rows of `cand`
from `offsets[i]`. model.rows_to_tensors pads both into the tensors the
network takes. With `privileged`, `priv_glob` / `priv_seats` (same packing as
the seats) carry what only the critic may see.

For many env processes in parallel, split each step into `send_step` (all
workers) then `recv_batch` (all workers).

Run this file directly for a random-agent smoke test of the bridge:

    python training/tractor/env_client.py --episodes 200 --envs 32
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCRIPT = os.path.normpath(os.path.join(HERE, "..", "dist", "tractor-env.mjs"))

ROLE_LEARNER, ROLE_OPPONENT, ROLE_IDLE = 0, 1, -1
KINDS = ("bid", "bury", "call", "play")


def lineup(learner: str = "all", weight: float = 1.0) -> dict:
    """learner: 'all' | 'one' | 'team' (every other seat); the remaining seats are opponents."""
    return {"weight": weight, "learner": learner}


@dataclass
class Batch:
    seat: np.ndarray  # (n,) int64 -- seat deciding row i
    role: np.ndarray  # (n,) int64 -- 0 learner, 1 opponent, -1 idle (quota reached)
    kind: np.ndarray  # (n,) int64 -- index into KINDS, -1 idle
    players: np.ndarray  # (n,) int64 -- table size of env i
    glob: np.ndarray  # (n, global_dim) float32
    seat_count: np.ndarray  # (n,) int64 -- seat blocks per row (0 idle)
    seat_offsets: np.ndarray  # (n,) int64 -- row i's first seat block in `seats`
    seat_slot: np.ndarray  # (seat_rows,) int64 -- each block's slot, 0..max_seats-1
    seats: np.ndarray  # (seat_rows, seat_dim) float32
    priv_glob: np.ndarray  # (n, priv_global_dim) float32, or (n, 0) unless privileged
    priv_seats: np.ndarray  # (seat_rows, priv_seat_dim) float32, or (seat_rows, 0)
    count: np.ndarray  # (n,) int64 -- candidates per row
    offsets: np.ndarray  # (n,) int64 -- row i's first candidate in `cand`
    cand: np.ndarray  # (sum(count), cand_dim) float32
    reward: np.ndarray  # (n, max_seats) float32 -- per-seat reward since the previous batch
    done: np.ndarray  # (n,) bool -- a round ended; row i is already the next round's first decision
    episodes: list  # summaries of the rounds that ended


class TractorEnv:
    def __init__(self, script: str = DEFAULT_SCRIPT, node: str | None = None):
        if not os.path.exists(script):
            raise FileNotFoundError(f"{script} not found -- run `npm run train:build` first")
        node = node or shutil.which("node")
        if not node:
            raise FileNotFoundError("node executable not found on PATH")
        self._proc = subprocess.Popen([node, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1 << 20)
        spec = self._call({"cmd": "spec"})
        d = spec["dims"]
        self.global_dim: int = d["global"]
        self.seat_dim: int = d["seat"]
        self.priv_global_dim: int = d["privGlobal"]
        self.priv_seat_dim: int = d["privSeat"]
        self.cand_dim: int = d["cand"]
        self.max_candidates: int = d["maxCandidates"]
        self.max_seats: int = d["maxSeats"]
        self.obs_dim: int = self.global_dim + self.max_seats * self.seat_dim
        self.card_slots: int = d["cardSlots"]
        self.layouts: dict = spec["layouts"]
        self.encoder_version: int = spec["encoderVersion"]
        self.action_space_version: int = spec["actionSpaceVersion"]
        self.policy_format: str = spec["policyFormat"]

    @property
    def versions(self) -> dict:
        return {"encoderVersion": self.encoder_version, "actionSpaceVersion": self.action_space_version}

    def _send(self, msg: dict) -> None:
        assert self._proc.stdin
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        assert self._proc.stdout
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"env server exited (code {self._proc.poll()})")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"env server: {reply.get('error')}")
        if "binary" in reply:
            sizes = reply.pop("binary")
            total = sum(size for _, size in sizes)
            data = self._proc.stdout.read(total)
            if data is None or len(data) < total:
                raise RuntimeError(f"env server exited mid-reply (code {self._proc.poll()})")
            at, raw = 0, {}
            for name, size in sizes:
                raw[name] = memoryview(data)[at:at + size]
                at += size
            reply["raw"] = raw
        return reply

    def _call(self, msg: dict) -> dict:
        self._send(msg)
        return self._recv()

    def _batch(self, r: dict) -> Batch:
        n, raw, rows = r["n"], r["raw"], r["seatRows"]

        def starts(c: np.ndarray) -> np.ndarray:
            o = np.zeros(len(c), np.int64)
            if len(c) > 1:
                o[1:] = np.cumsum(c)[:-1]
            return o

        count = np.frombuffer(raw["count"], dtype="<i2").astype(np.int64)
        seat_count = np.frombuffer(raw["seatCount"], dtype="<i2").astype(np.int64)
        pg = np.frombuffer(raw["privGlob"], dtype="<f4")
        ps = np.frombuffer(raw["privSeats"], dtype="<f4")
        return Batch(
            seat=np.asarray(r["seat"], dtype=np.int64),
            role=np.asarray(r["role"], dtype=np.int64),
            kind=np.asarray(r["kind"], dtype=np.int64),
            players=np.asarray(r["players"], dtype=np.int64),
            glob=np.frombuffer(raw["glob"], dtype="<f4").reshape(n, self.global_dim),
            seat_count=seat_count,
            seat_offsets=starts(seat_count),
            seat_slot=np.frombuffer(raw["seatSlot"], dtype=np.int8).astype(np.int64),
            seats=np.frombuffer(raw["seats"], dtype="<f4").reshape(rows, self.seat_dim),
            priv_glob=pg.reshape(n, self.priv_global_dim if len(pg) else 0),
            priv_seats=ps.reshape(rows, self.priv_seat_dim if len(ps) else 0),
            count=count,
            offsets=starts(count),
            cand=np.frombuffer(r["raw"]["cand"], dtype="<f4").reshape(r["candRows"], self.cand_dim),
            reward=np.asarray(r["reward"], dtype=np.float32).reshape(n, self.max_seats),
            done=np.asarray(r["done"], dtype=bool),
            episodes=r["episodes"],
        )

    def reset(
        self,
        num_envs: int = 16,
        seed: int = 1,
        lineups: list[dict] | None = None,
        scenario: dict | None = None,
        reward: dict | None = None,
        quota: int = 0,
        privileged: bool = False,
        points_shaping: float = 0.0,
        duplicate: bool = False,
        max_steps: int = 20_000,
        matches: bool = False,
        max_rounds: int = 200,
    ) -> Batch:
        """lineups: who sits where, sampled per round (see `lineup`). scenario:
        curriculum weights over DEFAULT_SCENARIO (training/tractor/scenario.ts),
        e.g. {"focus": 1}. quota: rounds per env before it idles (0 = forever).
        privileged: rows carry the critic's features. points_shaping: potential-based shaping weight on the points.
        duplicate: rounds in pairs, the same deal with the teams swapped.
        matches: episodes are whole matches from 2 (up to max_rounds rounds), for evals."""
        options: dict = {
            "numEnvs": num_envs,
            "seed": seed,
            "lineups": lineups or [lineup()],
            "scenario": scenario or {},
            "quota": quota,
            "privileged": privileged,
            "pointsShaping": points_shaping,
            "duplicate": duplicate,
            "maxSteps": max_steps,
            "matches": matches,
            "maxRounds": max_rounds,
        }
        if reward is not None:
            options["reward"] = reward
        return self._batch(self._call({"cmd": "reset", "options": options}))

    def step(self, choices) -> Batch:
        self.send_step(choices)
        return self.recv_batch()

    def send_step(self, choices) -> None:
        self._send({"cmd": "step", "actions": [int(a) for a in choices]})

    def recv_batch(self) -> Batch:
        return self._batch(self._recv())

    def configure(self, lineups: list[dict] | None = None, scenario: dict | None = None) -> None:
        """New lineups / curriculum for rounds that start from now on."""
        options: dict = {}
        if lineups is not None:
            options["lineups"] = lineups
        if scenario is not None:
            options["scenario"] = scenario
        self._call({"cmd": "configure", "options": options})

    def policy_logits(self, path: str) -> list[list[float]]:
        """The TypeScript runtime's logits (policy file `path`) for every pending row."""
        return self._call({"cmd": "policyLogits", "path": os.path.abspath(path)})["logits"]

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin
                self._proc.stdin.write(b'{"cmd": "close"}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()

    def __enter__(self) -> "TractorEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def random_choices(batch: Batch, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random candidate per row (0 for idle rows)."""
    return (rng.random(len(batch.count)) * np.maximum(1, batch.count)).astype(np.int64)


def _smoke(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    with TractorEnv() as env:
        print(f"obs {env.global_dim} + {env.max_seats} x {env.seat_dim}, cand {env.cand_dim} (max {env.max_candidates}), "
              f"encoder v{env.encoder_version}, action space v{env.action_space_version}")
        batch = env.reset(num_envs=args.envs, seed=args.seed, lineups=[lineup(args.learner)])
        episodes: list = []
        decisions = 0
        kinds: Counter = Counter()
        cands = []
        t0 = time.perf_counter()
        while len(episodes) < args.episodes:
            assert np.isfinite(batch.glob).all() and np.isfinite(batch.seats).all() and np.isfinite(batch.cand).all()
            assert (batch.count[batch.role >= 0] > 0).all(), "a decision row has no candidates"
            live = batch.role >= 0
            kinds.update(KINDS[k] for k in batch.kind[live])
            cands.extend(batch.count[live].tolist())
            batch = env.step(random_choices(batch, rng))
            decisions += int(live.sum())
            episodes.extend(batch.episodes)
        dt = time.perf_counter() - t0
    print(f"{len(episodes)} rounds, {decisions} network decisions in {dt:.1f}s ({decisions / dt:,.0f} decisions/s, "
          f"{len(episodes) / dt:,.1f} rounds/s)")
    print(f"decisions by kind {dict(kinds)}; candidates per decision mean {np.mean(cands):.1f}, max {max(cands)}")
    print(f"aborted {sum(e['aborted'] for e in episodes)}, truncated {sum(e['truncated'] for e in episodes)}")
    print("tables", dict(sorted(Counter(f"{e['players']}p {e['mode'][:2]}" for e in episodes).items())))
    print("level classes", dict(Counter(e["levelClass"] for e in episodes)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Random-agent smoke test for the Tractor env bridge.")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--learner", default="team", choices=("all", "one", "team"))
    p.add_argument("--seed", type=int, default=1)
    _smoke(p.parse_args())
