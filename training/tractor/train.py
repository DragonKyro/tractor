"""PPO trainer for the Tractor agent: one network for every table, optimized on
4-player Tractor.

    conda activate train
    python training/tractor/train.py --run v5 --init-checkpoint training/runs/tractor/v4/ckpt/latest.pt
    python training/tractor/train.py --run v5 --resume   # Ctrl+C any time: it checkpoints, then exits
    tensorboard --logdir training/runs/tractor

To stop a run you can't Ctrl+C, create training/runs/tractor/<run>/STOP: it
saves and exits after the current update (the file is removed on the next start).

Episodes are single rounds from the curriculum (training/tractor/scenario.ts):
85% 4-player Tractor, 15% a variety mix of every other table (2-12 players,
Finding Friends and Tractor), with the landlord's level sampled so point ranks
(5, 10, K) and A come up often. Reward (reward.ts): levels gained minus the
other side's / 3, +-0.25 for keeping / taking the landlord seat, +-0.25 x how
far the attackers' points were from the threshold, +-1 for winning the match;
plus potential-based shaping on the points as they are captured
(`--points-shaping`, sums to zero over a round).

The anchor (`--anchor`, by default the shipped AI, src/ai/nn/policy.json) is a
fixed opponent and the yardstick: it is copied into the run (anchor.json), so
the run's evals stay comparable after the shipped file changes.

Partners are always the same model: in every lineup the learner is a whole
team (or every seat). Each update:
  1. Collect `--rollout` decisions from every env: `--workers` node processes x
     `--envs-per-worker` rounds. Kinds of worker (`--mix kind=count,...`;
     default 50% self, 33% past, 17% anchor):
       self    every seat the learner (both teams are the current network)
       past    the learner as one team, the other a past snapshot of this run
               (the newest `--pool` policies, or with `--pool-span all` half the
               time any of them; redrawn every `--refresh-every` updates; self
               until a snapshot exists)
       anchor  the learner as one team, the anchor the other
     Every seat is played here: the env only deals, encodes and applies moves.
  2. A network that doesn't start from weights first distills the anchor for
     `--bc-updates`: every learner decision is labelled with the anchor's
     choice, and the anchor's moves are played for the first half; in the last
     `--bc-dagger` the network plays its own (DAgger). After that, PPO on every
     finished transition, per (env, seat) stream, with GAE.
  3. The critic sees every hand (privileged.ts; training only), so its value
     estimates, and the advantages, are far less noisy than the policy's own
     view would allow. Belief heads (training only) learn to predict the hidden
     hands from the actor's features. The actor never sees either; policy files
     hold the actor.
  4. Every `--save-every`: a resumable checkpoint (ckpt/latest.pt) and an
     exported float16 policy (policies/update_N.json, all kept): the EMA weights.
  5. Every `--eval-every` (100): the exported policy vs the anchor on duplicate
     deals of 4-player Tractor (evaluate.py: each deal played twice with the
     cards swapped, so the luck cancels), with skill measures. Every
     `--match-eval-every` (500, and at the end): duplicate full matches and
     single rounds of the other tables vs the anchor. -> evals.jsonl,
     progress.md, tensorboard.

Outputs go to training/runs/tractor/<run>/: config.json, anchor.json, train.log,
metrics.jsonl, evals.jsonl, progress.md, tensorboard events, ckpt/, policies/.
"""

from __future__ import annotations

import argparse
import collections
import copy
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev  # noqa: E402
from env_client import KINDS, ROLE_LEARNER, ROLE_OPPONENT, Batch, TractorEnv, lineup  # noqa: E402
from model import (DEFAULT_BELIEF, DEFAULT_CRITIC, TRAINING_ONLY, TractorNet, export_policy, load_actor_into,  # noqa: E402
                   load_checkpoint, load_policy, net_from_checkpoint, packed_positions, policy_inputs, rows_to_tensors,
                   save_checkpoint)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DEFAULT_RUNS = os.path.join(REPO, "training", "runs", "tractor")
WORKER_KINDS = ("self", "past", "anchor")


def default_workers() -> int:
    return max(1, min(12, (os.cpu_count() or 4) - 4))


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run = p.add_argument_group("run")
    run.add_argument("--run", default=None, help="run name (default: a timestamp)")
    run.add_argument("--runs-dir", default=DEFAULT_RUNS)
    run.add_argument("--resume", action="store_true", help="continue the run from its latest checkpoint")
    run.add_argument("--updates", type=int, default=5000, help="train until this many updates in total")
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--no-build", action="store_true", help="skip rebuilding training/dist")

    envs = p.add_argument_group("envs")
    envs.add_argument("--workers", type=int, default=default_workers(), help="node env processes")
    envs.add_argument("--envs-per-worker", type=int, default=32)
    envs.add_argument("--rollout", type=int, default=32, help="decisions per env per update")
    envs.add_argument("--mix", default="", help="workers by kind, e.g. self=6,past=4,anchor=2 "
                                                "(default: 50%% self, 33%% past, 17%% anchor; 60/40 without one)")
    envs.add_argument("--pool", type=int, default=8, help="past snapshots to draw opponents from (newest)")
    envs.add_argument("--pool-span", choices=("recent", "all"), default="all",
                      help="all: half the draws come from the whole run's snapshots, not just the newest --pool")
    envs.add_argument("--anchor", default=ev.SHIPPED_POLICY, metavar="POLICY",
                      help="the fixed opponent and yardstick: anchor workers play it, a fresh network distills it, "
                           "evals are against it (default: the shipped AI; 'none' for no anchor, no evals)")
    envs.add_argument("--refresh-every", type=int, default=10, help="updates between opponent redraws")
    envs.add_argument("--opponent-play", choices=("greedy", "sample"), default="sample")
    envs.add_argument("--scenario", default="", help="curriculum overrides as JSON (scenario.ts weights)")
    envs.add_argument("--reward", default="", help='reward weights as JSON, e.g. {"levels":1,"seat":0.25,"points":0.25,"win":1}')
    envs.add_argument("--points-shaping", type=float, default=0.5, help="potential-based shaping on captured points (0 = off)")

    net = p.add_argument_group("network")
    net.add_argument("--seat", type=ints, default=[128, 64], help="shared seat encoder widths")
    net.add_argument("--trunk", type=ints, default=[512, 512, 256], help="trunk widths")
    net.add_argument("--cand", type=ints, default=[128, 128], help="candidate tower widths")
    net.add_argument("--joint", type=int, default=256)
    net.add_argument("--critic-hidden", type=int, default=256)
    net.add_argument("--no-privileged", action="store_true", help="critic without the hidden hands")
    net.add_argument("--init-from", default=None, metavar="POLICY", help="start the actor from a policy file")
    net.add_argument("--init-checkpoint", default=None, metavar="PT",
                     help="start from another run's checkpoint: actor, critic, belief heads where present; "
                          "fresh optimizer, updates from 0")
    net.add_argument("--belief-coef", type=float, default=0.3, help="weight of the hidden-hand prediction loss (0 = off)")
    net.add_argument("--critic-warmup", type=int, default=0,
                     help="first updates train only the critic and belief heads, the policy frozen (e.g. after --init-from)")
    net.add_argument("--ema", type=float, default=0.995,
                     help="exported policies and evals use an exponential moving average of the weights (0 = off)")

    bc = p.add_argument_group("imitation")
    bc.add_argument("--bc-updates", type=int, default=None,
                    help="updates of distilling the anchor first (default 50 from scratch, 0 when starting from weights)")
    bc.add_argument("--bc-dagger", type=float, default=0.5, help="share of imitation where the network plays")
    bc.add_argument("--bc-epochs", type=int, default=2)
    bc.add_argument("--bc-lr", type=float, default=1e-3)

    ppo = p.add_argument_group("ppo")
    ppo.add_argument("--lr", type=float, default=3e-4)
    ppo.add_argument("--lr-final", type=float, default=3e-5, help="the learning rate falls linearly to this by --updates")
    ppo.add_argument("--gamma", type=float, default=1.0, help="a round is ~10-50 decisions per seat: no discount")
    ppo.add_argument("--lam", type=float, default=0.95)
    ppo.add_argument("--clip", type=float, default=0.2)
    ppo.add_argument("--epochs", type=int, default=3)
    ppo.add_argument("--minibatch", type=int, default=2048)
    ppo.add_argument("--grad-accum", type=int, default=1,
                     help="minibatches per optimizer step: the step averages minibatch x this many decisions "
                          "(less noisy steps without the GPU memory of a bigger minibatch)")
    ppo.add_argument("--vf-coef", type=float, default=0.5)
    ppo.add_argument("--ent-coef", type=float, default=0.003)
    ppo.add_argument("--max-grad-norm", type=float, default=0.5)
    ppo.add_argument("--lr-warmup", type=int, default=0,
                     help="ramp the learning rate up from 10%% over this many PPO updates (from weights: a fresh "
                          "optimizer's first steps are full-size in every direction)")
    ppo.add_argument("--target-kl", type=float, default=0.0,
                     help="stop an update's remaining epochs once its approximate KL passes 1.5x this (0 = off)")

    out = p.add_argument_group("checkpoints / evals")
    out.add_argument("--save-every", type=int, default=25)
    out.add_argument("--export-dtype", choices=("f16", "f32"), default="f16", help="policy file weights")
    out.add_argument("--eval-every", type=int, default=100, help="duplicate rounds vs the anchor (0 = never)")
    out.add_argument("--eval-deals", type=int, default=1024, help="duplicate deal pairs per eval")
    out.add_argument("--eval-procs", type=int, default=8)
    out.add_argument("--match-eval-every", type=int, default=500,
                     help="full duplicate matches + other tables vs the anchor (0 = never; also at the end)")
    out.add_argument("--match-eval-pairs", type=int, default=24)
    out.add_argument("--variety-eval-rounds", type=int, default=512)
    a = p.parse_args(argv)
    if not a.anchor or a.anchor.lower() == "none":
        a.anchor = None
    if a.bc_updates is None:
        a.bc_updates = 0 if (a.init_checkpoint or a.init_from or not a.anchor) else 50
    return a


def parse_mix(spec: str, workers: int, anchor: bool = False) -> list[str]:
    if not spec:
        n_self = max(1, round(workers * (0.5 if anchor else 0.6)))
        n_anchor = max(1, round(workers * 0.17)) if anchor and workers > 2 else 0
        return ["self"] * n_self + ["past"] * (workers - n_self - n_anchor) + ["anchor"] * n_anchor
    kinds: list[str] = []
    for part in spec.split(","):
        kind, _, count = part.strip().partition("=")
        if kind not in WORKER_KINDS or not count.isdigit():
            raise SystemExit(f"--mix: bad entry {part!r} (kinds: {', '.join(WORKER_KINDS)})")
        kinds += [kind] * int(count)
    if not kinds:
        raise SystemExit("--mix lists no workers")
    return kinds


def lineups_for(kind: str, has_pool: bool) -> list[dict]:
    if kind == "anchor" or (kind == "past" and has_pool):
        return [lineup("team")]
    return [lineup("all")]


def build_typescript() -> None:
    vite = os.path.join(REPO, "node_modules", "vite", "bin", "vite.js")
    node = shutil.which("node")
    if not node or not os.path.isfile(vite):
        raise FileNotFoundError("need node on PATH and `npm install` run in the repo")
    subprocess.run([node, vite, "build", "--config", os.path.join("training", "vite.config.ts")], cwd=REPO, check=True)


def _starts(c: np.ndarray) -> np.ndarray:
    o = np.zeros(len(c), np.int64)
    if len(c) > 1:
        o[1:] = np.cumsum(c)[:-1]
    return o


def merge(batches: list[Batch]) -> SimpleNamespace:
    """Several workers' batches as one (packed arrays concatenated, offsets recomputed)."""
    cat = lambda k: np.concatenate([getattr(b, k) for b in batches])  # noqa: E731
    seat_count, count = cat("seat_count"), cat("count")
    return SimpleNamespace(role=cat("role"), seat=cat("seat"), kind=cat("kind"),
                           glob=cat("glob"), seat_count=seat_count, seat_offsets=_starts(seat_count),
                           seat_slot=cat("seat_slot"), seats=cat("seats"), priv_glob=cat("priv_glob"),
                           priv_seats=cat("priv_seats"), count=count, offsets=_starts(count), cand=cat("cand"))


# ---------------------------------------------------------------- rollout storage

class Packed:
    """A growable (rows, dim) array."""

    def __init__(self, rows: int, dim: int, dtype=np.float32):
        self.data = np.zeros((max(1, rows), dim), dtype)
        self.m = 0

    def add(self, x: np.ndarray) -> int:
        k = len(x)
        if self.m + k > len(self.data):
            grow = np.zeros((max(2 * len(self.data), self.m + k), self.data.shape[1]), self.data.dtype)
            grow[: self.m] = self.data[: self.m]
            self.data = grow
        start = self.m
        self.data[start:start + k] = x
        self.m += k
        return start


class Rollout:
    """Learner transitions for one update, in order. Each (env, seat) is a
    stream: `next[i]` links a transition to that seat's following decision in
    the same round, so rewards and GAE follow the seat that earned them. A
    seat's newest transition is still collecting reward when the rollout ends;
    it is carried into the next update instead of being trained on. Seat
    blocks, their privileged features and candidates are stored packed, with a
    per-row offset and count, in float16 (half the memory; the features are counts
    and fractions, which float16 holds to ~0.1%)."""

    ROW = ("kind", "action", "logp", "value", "teacher", "reward", "done")

    def __init__(self, capacity: int, dims: SimpleNamespace):
        self.cap, self.d = capacity, dims
        self.glob = np.zeros((capacity, dims.glob), np.float16)
        self.priv_glob = np.zeros((capacity, dims.priv_glob), np.float16)
        for k in ("kind", "action", "seat_off", "seat_cnt", "cand_off", "cand_cnt"):
            setattr(self, k, np.zeros(capacity, np.int64))
        self.logp = np.zeros(capacity, np.float32)
        self.value = np.zeros(capacity, np.float32)
        self.teacher = np.full(capacity, -1, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, bool)
        self.next = np.full(capacity, -1, np.int64)
        self.n = 0
        c = capacity
        self.seats = Packed(c * 5, dims.seat, np.float16)
        self.slots = Packed(c * 5, 1, np.int64)
        self.priv_seats = Packed(c * 5 if dims.priv_seat else 1, dims.priv_seat, np.float16)
        self.cand = Packed(c * 9, dims.cand, np.float16)

    def clear(self) -> None:
        self.reward[: self.n] = 0
        self.done[: self.n] = False
        self.next[: self.n] = -1
        self.n = 0
        self.seats.m = self.slots.m = self.priv_seats.m = self.cand.m = 0

    def add(self, src, rows: np.ndarray, **values: np.ndarray) -> np.ndarray:
        """Rows `rows` of a (merged) batch-like `src`, with their per-row values."""
        k = len(rows)
        if self.n + k > self.cap:
            raise RuntimeError("rollout over capacity")
        at = np.arange(self.n, self.n + k)
        self.glob[at] = src.glob[rows]
        if self.d.priv_glob:
            self.priv_glob[at] = src.priv_glob[rows]
        _, spos = packed_positions(src.seat_offsets, src.seat_count, rows)
        s0 = self.seats.add(src.seats[spos])
        self.slots.add(src.seat_slot[spos][:, None])
        if self.d.priv_seat:
            self.priv_seats.add(src.priv_seats[spos])
        self.seat_cnt[at] = src.seat_count[rows]
        self.seat_off[at] = s0 + _starts(src.seat_count[rows])
        _, cpos = packed_positions(src.offsets, src.count, rows)
        c0 = self.cand.add(src.cand[cpos])
        self.cand_cnt[at] = src.count[rows]
        self.cand_off[at] = c0 + _starts(src.count[rows])
        for name, v in values.items():
            getattr(self, name)[at] = v
        self.n += k
        return at

    def take(self, idx: np.ndarray) -> SimpleNamespace:
        """Rows `idx` as a batch-like object (for carrying into the next update)."""
        _, spos = packed_positions(self.seat_off, self.seat_cnt, idx)
        _, cpos = packed_positions(self.cand_off, self.cand_cnt, idx)
        return SimpleNamespace(
            glob=self.glob[idx].copy(), priv_glob=self.priv_glob[idx].copy(),
            seat_count=self.seat_cnt[idx].copy(), seat_offsets=_starts(self.seat_cnt[idx]),
            seat_slot=self.slots.data[spos, 0].copy(), seats=self.seats.data[spos].copy(),
            priv_seats=self.priv_seats.data[spos].copy() if self.d.priv_seat else np.zeros((len(spos), 0), np.float16),
            count=self.cand_cnt[idx].copy(), offsets=_starts(self.cand_cnt[idx]), cand=self.cand.data[cpos].copy(),
            values={name: getattr(self, name)[idx].copy() for name in self.ROW})


@dataclass
class Worker:
    kind: str
    env: TractorEnv
    batch: Batch | None = None
    opponent: TractorNet | None = None
    opponent_name: str = ""
    has_pool: bool = False


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.update = 0
        self.workers: list[Worker] = []
        self.carry: SimpleNamespace | None = None
        self.carry_streams = np.zeros((0, 2), np.int64)
        self.window: dict[str, collections.deque] = {k: collections.deque(maxlen=4000) for k in WORKER_KINDS}
        self.landlord_wins: collections.deque = collections.deque(maxlen=4000)
        self.episodes = self.aborted = 0
        self.decisions = self.learner_decisions = 0
        self.evals: list[dict] = []

    # ------------------------------------------------------------ setup

    def setup(self) -> None:
        a = self.args
        a.run = a.run or time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(a.runs_dir, a.run)
        self.ckpt_path = os.path.join(self.dir, "ckpt", "latest.pt")
        self.policy_dir = os.path.join(self.dir, "policies")
        os.makedirs(self.policy_dir, exist_ok=True)
        stop = os.path.join(self.dir, "STOP")
        if os.path.exists(stop):
            os.remove(stop)
        if not a.no_build:
            build_typescript()
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)

        self.anchor_path = self.pin_anchor()
        kinds = parse_mix(a.mix, a.workers, anchor=bool(self.anchor_path))
        if "anchor" in kinds and not self.anchor_path:
            raise SystemExit("--mix has anchor workers but no --anchor policy")
        if not self.anchor_path:
            self.log("no anchor: no distillation, no evals")
            a.bc_updates, a.eval_every, a.match_eval_every = 0, 0, 0
        self.privileged = not a.no_privileged
        self.scenario = json.loads(a.scenario) if a.scenario else None
        reward = json.loads(a.reward) if a.reward else None
        for k, kind in enumerate(kinds):
            env = TractorEnv()
            w = Worker(kind, env)
            w.batch = env.reset(num_envs=a.envs_per_worker, seed=a.seed * 1000 + k, lineups=lineups_for(kind, kind == "anchor"),
                                scenario=self.scenario, reward=reward, privileged=self.privileged,
                                points_shaping=a.points_shaping)
            self.workers.append(w)
        e0 = self.workers[0].env
        self.versions = e0.versions
        self.max_seats = e0.max_seats
        self.dims = SimpleNamespace(glob=e0.global_dim, seat=e0.seat_dim, cand=e0.cand_dim,
                                    priv_glob=e0.priv_global_dim if self.privileged else 0,
                                    priv_seat=e0.priv_seat_dim if self.privileged else 0)
        self.critic_spec = {"privGlobal": e0.priv_global_dim, "privSeat": e0.priv_seat_dim,
                            "seatWidth": DEFAULT_CRITIC["seatWidth"], "hidden": a.critic_hidden}
        self.belief_spec = {"slots": e0.priv_seat_dim, "hidden": DEFAULT_BELIEF["hidden"]}
        self.anchor_net = load_policy(self.anchor_path, self.versions, self.device) if self.anchor_path else None
        for w in self.workers:
            if w.kind == "anchor":
                w.opponent, w.opponent_name, w.has_pool = self.anchor_net, self.anchor_path, True
        n_envs = a.envs_per_worker * len(self.workers)
        self.slices = [slice(k * a.envs_per_worker, (k + 1) * a.envs_per_worker) for k in range(len(self.workers))]
        self.last = np.full((n_envs, self.max_seats), -1, np.int64)
        # One learner row per env per step, plus up to one carried row per seat.
        self.buf = Rollout(n_envs * (a.rollout + self.max_seats), self.dims)
        half = max(1, len(self.workers) // 2)
        idx = list(range(len(self.workers)))
        self.groups = [g for g in (idx[:half], idx[half:]) if g]
        self.in_flight = [False] * len(self.workers)

        ema_state = None
        if a.resume and os.path.exists(self.ckpt_path):
            ckpt = load_checkpoint(self.ckpt_path, self.device)
            self.net = net_from_checkpoint(ckpt, self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=a.lr)
            self.opt.load_state_dict(ckpt["optimizer"])
            self.update = int(ckpt["update"])
            ema_state = ckpt.get("ema")
            self.evals = self.read_jsonl("evals.jsonl")
            self.log(f"resumed {self.ckpt_path} at update {self.update}")
        else:
            if a.resume:
                self.log(f"--resume: no checkpoint at {self.ckpt_path}, starting fresh")
            if a.init_checkpoint:
                init = load_checkpoint(a.init_checkpoint, self.device)
                arch = init["arch"]
                self.net = TractorNet(e0.global_dim, e0.seat_dim, e0.max_seats, e0.cand_dim, arch["seat"], arch["trunk"],
                                      arch["cand"], arch["joint"], critic=init.get("critic") or self.critic_spec,
                                      belief=init.get("belief") or self.belief_spec).to(self.device)
                missing, _ = self.net.load_state_dict(init["model"], strict=False)
                if any(not k.startswith(TRAINING_ONLY) for k in missing):
                    raise SystemExit(f"{a.init_checkpoint} lacks actor weights {missing[:3]}")
                self.log(f"started from {a.init_checkpoint} ({init.get('meta', {})}); fresh: {sorted(set(k.split('.')[0] for k in missing)) or 'nothing'}")
            else:
                self.net = TractorNet(e0.global_dim, e0.seat_dim, e0.max_seats, e0.cand_dim, a.seat, a.trunk, a.cand, a.joint,
                                      critic=self.critic_spec, belief=self.belief_spec).to(self.device)
                if a.init_from:
                    load_actor_into(self.net, a.init_from)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=a.lr)
        if self.net.belief_arch is None and a.belief_coef > 0:
            raise SystemExit("this checkpoint's network has no belief heads; use --belief-coef 0")
        self.net.eval()
        # The averaged weights that are exported and evaluated (the trained weights keep moving).
        self.ema_net = copy.deepcopy(self.net).eval()
        if ema_state:
            self.ema_net.load_state_dict(ema_state)
        for q in self.ema_net.parameters():
            q.requires_grad_(False)
        self.ema_pairs = [(q, p) for (k, q), (_, p) in zip(self.ema_net.named_parameters(), self.net.named_parameters())
                          if not k.startswith(TRAINING_ONLY)]
        with open(os.path.join(self.dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({**vars(a), "versions": self.versions, "arch": self.net.arch, "critic": self.net.critic_arch,
                       "worker_kinds": kinds, "parameters": self.net.num_parameters(),
                       "actor_parameters": self.net.num_parameters(actor_only=True)}, f, indent=2)
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(self.dir)
        except Exception as e:  # tensorboard is optional
            self.log(f"(no tensorboard: {e})")
        self.log(f"{len(self.workers)} workers {dict(collections.Counter(kinds))} x {a.envs_per_worker} envs; "
                 f"actor {self.net.num_parameters(actor_only=True):,} parameters {self.net.arch}, critic "
                 f"{self.net.num_parameters() - self.net.num_parameters(actor_only=True):,} "
                 f"({'privileged' if self.privileged else 'plain'}); anchor {self.anchor_name()}; device {self.device}")
        self.refresh_opponents()

    def pin_anchor(self) -> str | None:
        """The run's own copy of the anchor (made on the first start; a resume keeps using it)."""
        pinned = os.path.join(self.dir, "anchor.json")
        if os.path.exists(pinned) and self.args.resume:
            return pinned
        if not self.args.anchor:
            return None
        if os.path.abspath(self.args.anchor) != os.path.abspath(pinned):
            shutil.copyfile(self.args.anchor, pinned)
        return pinned

    def anchor_name(self) -> str:
        """'<run> update <N>' from the anchor's file, else its file name."""
        if not self.anchor_path:
            return "none"
        if not hasattr(self, "_anchor_name"):
            with open(self.anchor_path, encoding="utf-8") as f:
                meta = json.load(f).get("meta") or {}
            self._anchor_name = (f"{meta['run']} update {meta['update']}" if "run" in meta and "update" in meta
                                 else os.path.basename(self.args.anchor or self.anchor_path))
        return self._anchor_name

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "train.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_jsonl(self, name: str) -> list[dict]:
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    def append_jsonl(self, name: str, row: dict) -> None:
        with open(os.path.join(self.dir, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------ schedule

    @property
    def imitating(self) -> bool:
        return self.update < self.args.bc_updates

    @property
    def teacher_forcing(self) -> bool:
        a = self.args
        return self.update < a.bc_updates * (1 - a.bc_dagger)

    def lr(self) -> float:
        a = self.args
        if self.imitating:
            return a.bc_lr
        span = max(1, a.updates - a.bc_updates)
        t = min(1.0, max(0.0, (self.update - a.bc_updates) / span))
        lr = a.lr + (a.lr_final - a.lr) * t
        ramp_from = max(a.bc_updates, a.critic_warmup)
        if a.lr_warmup > 0 and self.update < ramp_from + a.lr_warmup:
            lr *= 0.1 + 0.9 * max(0, self.update - ramp_from) / a.lr_warmup
        return lr

    def phase(self) -> str:
        if self.update < self.args.critic_warmup:
            return "critic warm-up"
        return "imitate(teacher)" if self.teacher_forcing else "imitate(dagger)" if self.imitating else "ppo"

    # ------------------------------------------------------------ opponents

    def pool(self, everything: bool = False) -> list[str]:
        files = glob.glob(os.path.join(self.policy_dir, "update_*.json"))
        files.sort(key=lambda p: int(os.path.basename(p)[7:-5]))
        return files if everything else files[-self.args.pool:]

    def refresh_opponents(self) -> None:
        pool = self.pool()
        for w in self.workers:
            if w.kind != "past":
                continue
            if pool:
                every = self.pool(everything=True)
                path = random.choice(every if self.args.pool_span == "all" and random.random() < 0.5 else pool)
                if path != w.opponent_name:
                    w.opponent = load_policy(path, self.versions, self.device)
                    w.opponent_name = path
            if bool(pool) != w.has_pool:
                w.has_pool = bool(pool)
                w.env.configure(lineups=lineups_for("past", w.has_pool))

    # ------------------------------------------------------------ collecting

    def inputs(self, src, rows: np.ndarray) -> dict:
        x = rows_to_tensors(src, rows, self.device, privileged=self.privileged, max_seats=self.max_seats)
        if not self.privileged:
            x["priv_glob"] = torch.zeros(len(rows), self.critic_spec["privGlobal"], device=self.device)
            x["priv_seats"] = torch.zeros(len(rows), self.max_seats, self.critic_spec["privSeat"], device=self.device)
        return x

    def teach(self, x: dict) -> np.ndarray:
        """The anchor's choice per row (the distillation label)."""
        with torch.no_grad():
            logits, _ = self.anchor_net(*policy_inputs(x))
        return logits.argmax(-1).cpu().numpy()

    def sample(self, x: dict, forced: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A sampled candidate per row (or `forced` where >= 0), its log-probability, and the critic's value."""
        with torch.no_grad():
            logits, h = self.net(*policy_inputs(x))
            value = self.net.value(h, x["priv_glob"], x["priv_seats"], x["seat_mask"])
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            if forced is not None:
                f = torch.from_numpy(forced).to(action.device)
                action = torch.where(f >= 0, f, action)
            return action.cpu().numpy(), dist.log_prob(action).cpu().numpy(), value.cpu().numpy()

    def decide(self, group: list[int]) -> None:
        buf, last = self.buf, self.last
        ws = [self.workers[k] for k in group]
        b = merge([w.batch for w in ws])
        envs = np.concatenate([np.arange(self.slices[k].start, self.slices[k].stop) for k in group])
        actions = np.zeros(len(b.role), np.int64)

        learner = np.flatnonzero(b.role == ROLE_LEARNER)
        if len(learner):
            x = self.inputs(b, learner)
            teacher = self.teach(x) if self.imitating else np.full(len(learner), -1, np.int64)
            act, logp, value = self.sample(x, teacher if self.teacher_forcing else None)
            actions[learner] = act
            rows = buf.add(b, learner, kind=b.kind[learner], action=act, logp=logp, value=value, teacher=teacher)
            env_ids, seats = envs[learner], b.seat[learner]
            prev = last[env_ids, seats]
            linked = prev >= 0
            buf.next[prev[linked]] = rows[linked]
            last[env_ids, seats] = rows

        at = 0
        for w in ws:
            n = len(w.batch.role)
            if w.opponent is not None:
                rows_w = np.flatnonzero(w.batch.role == ROLE_OPPONENT) + at
                if len(rows_w):
                    x = rows_to_tensors(b, rows_w, self.device, max_seats=self.max_seats)
                    with torch.no_grad():
                        logits, _ = w.opponent(*policy_inputs(x))
                        if self.args.opponent_play == "greedy":
                            actions[rows_w] = logits.argmax(-1).cpu().numpy()
                        else:
                            actions[rows_w] = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            at += n

        at = 0
        for k, w in zip(group, ws):
            n = len(w.batch.role)
            w.env.send_step(actions[at:at + n])
            self.in_flight[k] = True
            at += n
        self.decisions += int((b.role >= 0).sum())
        self.learner_decisions += len(learner)

    def absorb(self, group: list[int]) -> None:
        buf, last = self.buf, self.last
        for k in group:
            w = self.workers[k]
            w.batch = w.env.recv_batch()
            self.in_flight[k] = False
            lw = last[self.slices[k]]
            for s in range(self.max_seats):
                ok = lw[:, s] >= 0
                buf.reward[lw[ok, s]] += w.batch.reward[ok, s]
                buf.done[lw[ok & w.batch.done, s]] = True
            lw[w.batch.done] = -1
            for ep in w.batch.episodes:
                self.record_episode(w.kind, ep)

    def record_episode(self, kind: str, ep: dict) -> None:
        self.episodes += 1
        if ep.get("aborted"):
            self.aborted += 1
            return
        if self.teacher_forcing or ep["truncated"]:
            return
        mine = [s for s, r in enumerate(ep["roles"]) if r == "learner"]
        if len(mine) == ep["players"]:
            if ep["focus"]:
                self.landlord_wins.append(float(ep["landlordWon"]))
            return
        if mine and ep["focus"]:
            key = "anchor" if kind == "anchor" else "past"
            self.window[key].append(float(np.mean([ep["gain"][s] for s in mine])))

    def collect(self) -> np.ndarray:
        """Fill the rollout; returns a mask of the transitions still pending (carried, not trained on now)."""
        a, buf, last = self.args, self.buf, self.last
        buf.clear()
        last[:] = -1
        if self.carry is not None and len(self.carry_streams):
            c = self.carry
            rows = buf.add(c, np.arange(len(c.glob)), **c.values)
            last[self.carry_streams[:, 0], self.carry_streams[:, 1]] = rows
        for _ in range(a.rollout):
            for g in self.groups:
                if self.in_flight[g[0]]:
                    self.absorb(g)
                self.decide(g)
        for g in self.groups:
            if self.in_flight[g[0]]:
                self.absorb(g)
        pending = np.zeros(buf.n, bool)
        streams = np.argwhere(last >= 0)
        pending[last[streams[:, 0], streams[:, 1]]] = True
        self.carry = buf.take(last[streams[:, 0], streams[:, 1]])
        self.carry_streams = streams
        return pending

    def advantages(self, pending: np.ndarray) -> np.ndarray:
        a, buf = self.args, self.buf
        adv = np.zeros(buf.n, np.float32)
        for i in range(buf.n - 1, -1, -1):
            if pending[i]:
                continue
            if buf.done[i]:
                next_value = next_adv = 0.0
            else:
                j = buf.next[i]
                next_value = buf.value[j]
                next_adv = 0.0 if pending[j] else adv[j]
            delta = buf.reward[i] + a.gamma * next_value - buf.value[i]
            adv[i] = delta + a.gamma * a.lam * next_adv
        return adv

    # ------------------------------------------------------------ learning

    def learn(self, pending: np.ndarray, adv: np.ndarray) -> dict:
        a, buf, dev, S = self.args, self.buf, self.device, self.max_seats
        idx = np.flatnonzero(~pending)
        n = len(idx)
        if n == 0:
            return {"transitions": 0}
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
        glob_t = t(buf.glob[idx]).float()
        priv_glob = t(buf.priv_glob[idx]).float() if self.privileged else torch.zeros(n, self.critic_spec["privGlobal"], device=dev)
        seats_d = t(buf.seats.data[: buf.seats.m]).float()
        slots_d = t(buf.slots.data[: buf.slots.m, 0])
        priv_d = t(buf.priv_seats.data[: buf.priv_seats.m]).float() if self.privileged else None
        cand_d = t(buf.cand.data[: buf.cand.m]).float()
        seat_off, seat_cnt = t(buf.seat_off[idx]), t(buf.seat_cnt[idx])
        cand_off, cand_cnt = t(buf.cand_off[idx]), t(buf.cand_cnt[idx])
        kind, action, old_logp = t(buf.kind[idx]), t(buf.action[idx]), t(buf.logp[idx])
        advantage = t(adv[idx])
        returns = advantage + t(buf.value[idx])
        teacher = t(buf.teacher[idx])
        norm_adv = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        def gather(mb: torch.Tensor) -> tuple:
            cnt, off = seat_cnt[mb], seat_off[mb]
            which = torch.repeat_interleave(torch.arange(len(mb), device=dev), cnt)
            starts = torch.cumsum(cnt, 0) - cnt
            pos = torch.repeat_interleave(off, cnt) + torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(starts, cnt)
            sl = slots_d[pos]
            seats = torch.zeros(len(mb), S, seats_d.shape[1], device=dev)
            seats[which, sl] = seats_d[pos]
            mask = torch.zeros(len(mb), S, dtype=torch.bool, device=dev)
            mask[which, sl] = True
            ps = torch.zeros(len(mb), S, self.critic_spec["privSeat"], device=dev)
            if priv_d is not None:
                ps[which, sl] = priv_d[pos]
            c = cand_cnt[mb]
            k = int(c.max())
            ar = torch.arange(k, device=dev)
            cmask = ar[None, :] < c[:, None]
            cpos = torch.where(cmask, cand_off[mb][:, None] + ar[None, :], torch.zeros_like(cand_off[mb][:, None]))
            cand = cand_d[cpos] * cmask[..., None]
            return glob_t[mb], seats, mask, cand, cmask, priv_glob[mb], ps

        stats: dict[str, list[float]] = collections.defaultdict(list)
        warm = self.update < a.critic_warmup
        lr = self.lr()
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.net.train()
        accum = max(1, a.grad_accum)
        starts = list(range(0, n, a.minibatch))
        for epoch in range(a.bc_epochs if self.imitating else a.epochs):
            if (a.target_kl > 0 and not self.imitating and not warm and epoch > 0 and stats["approx_kl"]
                    and np.mean(stats["approx_kl"][-len(starts):]) > 1.5 * a.target_kl):
                stats["kl_stops"].append(1.0)
                break
            perm = torch.randperm(n, device=dev)
            self.opt.zero_grad()
            for i, start in enumerate(starts):
                mb = perm[start:start + a.minibatch]
                g_, seats, mask, cand, cmask, pg, ps = gather(mb)
                logits, h = self.net(g_, seats, mask, cand, cmask)
                hv = h.detach() if warm else h  # warm-up: the heads learn on frozen features
                value = self.net.value(hv, pg, ps, mask)
                belief_loss = logits.new_zeros(())
                if self.net.belief_arch is not None and a.belief_coef > 0 and self.privileged:
                    bl, kl = self.net.beliefs(h, seats, mask, detach=warm)
                    others = mask.clone()
                    others[:, 0] = False  # slot 0 is me: my own hand isn't hidden
                    target = ps.clamp(0, 1)
                    seat_bce = F.binary_cross_entropy_with_logits(bl, target, reduction="none").mean(-1)
                    belief_loss = (seat_bce * others).sum() / others.sum().clamp_min(1)
                    kitty_t = pg[:, :self.belief_spec["slots"]].clamp(0, 1)
                    belief_loss = belief_loss + 0.5 * F.binary_cross_entropy_with_logits(kl, kitty_t)
                    with torch.no_grad():
                        # Holder accuracy: for each card someone else holds, is the most likely seat a true holder?
                        prob = torch.sigmoid(bl) * others.unsqueeze(-1)
                        held = (target * others.unsqueeze(-1)).sum(1) > 0
                        guess = prob.argmax(1, keepdim=True)
                        right = target.gather(1, guess).squeeze(1) > 0
                        if held.any():
                            stats["belief_holder_acc"].append((right & held).float().sum().item() / held.float().sum().item())
                    stats["belief_loss"].append(belief_loss.item())
                logp_all = F.log_softmax(logits, dim=-1)
                entropy = -(logp_all.exp() * logp_all.clamp_min(-1e4)).sum(-1).mean()
                tb = teacher[mb]
                labelled = tb >= 0
                imitation = F.nll_loss(logp_all[labelled], tb[labelled]) if labelled.any() else logp_all.new_zeros(())
                if labelled.any():
                    agree = logp_all[labelled].argmax(-1) == tb[labelled]
                    stats["teacher_agreement"].append(agree.float().mean().item())
                    kb = kind[mb][labelled]
                    for ki, name in enumerate(KINDS):
                        sel = kb == ki
                        if sel.any():
                            stats[f"agree_{name}"].append(agree[sel].float().mean().item())
                if warm:
                    pi_loss = logits.new_zeros(())
                elif self.imitating:
                    pi_loss = imitation
                else:
                    new_logp = logp_all.gather(1, action[mb, None]).squeeze(1)
                    log_ratio = new_logp - old_logp[mb]
                    ratio = log_ratio.exp()
                    adv_b = norm_adv[mb]
                    pi_loss = -torch.min(ratio * adv_b, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_b).mean()
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
                v_loss = 0.5 * (value - returns[mb]).pow(2).mean()
                loss = (pi_loss + a.vf_coef * v_loss + a.belief_coef * belief_loss
                        - (0.0 if self.imitating or warm else a.ent_coef) * entropy)
                if not torch.isfinite(loss):
                    stats["skipped_steps"].append(1.0)
                else:
                    # Each minibatch's share of the step (the last group of an epoch may be smaller).
                    group = min(accum, len(starts) - (i // accum) * accum)
                    (loss / group).backward()
                if (i + 1) % accum == 0 or i == len(starts) - 1:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), a.max_grad_norm)
                    self.opt.step()
                    self.opt.zero_grad()
                    if a.ema > 0:
                        with torch.no_grad():
                            for q, p_ in self.ema_pairs:
                                q.mul_(a.ema).add_(p_.detach(), alpha=1 - a.ema)
                if not torch.isfinite(loss):
                    continue
                stats["policy_loss"].append(pi_loss.item())
                stats["value_loss"].append(v_loss.item())
                stats["entropy"].append(entropy.item())
        self.net.eval()
        out = {k: float(np.sum(v) if k in ("skipped_steps", "kl_stops") else np.mean(v)) for k, v in stats.items()}
        ret_np, val_np = returns.cpu().numpy(), buf.value[idx]
        out["explained_variance"] = float(1 - np.var(ret_np - val_np) / (np.var(ret_np) + 1e-8))
        out["transitions"] = n
        out["imitation"] = 1.0 if self.imitating else 0.0
        out["teacher_forcing"] = 1.0 if self.teacher_forcing else 0.0
        out["critic_warmup"] = 1.0 if warm else 0.0
        out["lr"] = lr
        return out

    # ------------------------------------------------------------ checkpoints / evals

    def save(self) -> str:
        """A resumable checkpoint, and the policy file: the averaged weights (EMA), unless --ema 0."""
        save_checkpoint(self.ckpt_path, self.net, self.opt, {"update": self.update, "args": vars(self.args),
                                                             "ema": self.ema_net.state_dict()})
        path = os.path.join(self.policy_dir, f"update_{self.update}.json")
        export_policy(self.ema_net if self.args.ema > 0 else self.net, path, self.versions,
                      meta={"update": self.update, "run": self.args.run, "ema": self.args.ema}, dtype=self.args.export_dtype)
        return path

    def evaluate(self, full: bool) -> None:
        """The exported policy file (exactly what the game would run) on duplicate deals of 4-player
        Tractor vs the anchor. With `full`, also duplicate matches and other tables vs the anchor."""
        a = self.args
        t0 = time.perf_counter()
        net = load_policy(self.save(), self.versions, self.device)
        vs, label = self.anchor_net, "the anchor"
        row: dict = {"update": self.update, "vs": self.anchor_name()}
        d = ev.summarize_duplicate(ev.play_duplicate(net, a.eval_deals, vs, a.eval_procs, device=self.device))
        row["duplicate"] = d
        lines = [ev.format_duplicate(d, label)]
        if full:
            m = ev.summarize_duplicate_matches(ev.play_duplicate(net, a.match_eval_pairs, vs, a.eval_procs,
                                                                 seed=ev.EVAL_SEED + 1, device=self.device, matches=True))
            row["matches"] = m
            lines.append(ev.format_duplicate_matches(m, label))
            r = ev.summarize_rounds(ev.play_variety(net, a.variety_eval_rounds, vs, device=self.device))
            row["variety"] = r
            lines.append(ev.format_rounds(r, label))
        row["seconds"] = round(time.perf_counter() - t0, 1)
        self.log(f"eval @ {self.update} ({row['seconds']}s)\n" + "\n".join(lines))
        self.evals.append(row)
        self.append_jsonl("evals.jsonl", row)
        if self.tb:
            for k in ("gain", "ahead", "keep_landlord", "take_landlord"):
                if d[k]["mean"] is not None:
                    self.tb.add_scalar(f"eval/{k}", d[k]["mean"], self.update)
            for side in ("net", "opp"):
                for k, v in d["skills"][side].items():
                    if v is not None:
                        self.tb.add_scalar(f"skills_{side}/{k}", v, self.update)
            if full:
                for k in ("wins", "margin", "rounds"):
                    if row["matches"][k]["mean"] is not None:
                        self.tb.add_scalar(f"match/{k}", row["matches"][k]["mean"], self.update)
                if row["variety"].get("all", {}).get("mean") is not None:
                    self.tb.add_scalar("variety/gain", row["variety"]["all"]["mean"], self.update)
        self.write_progress()

    def write_progress(self) -> None:
        vs = f"the anchor ({self.anchor_name()})"

        def f(v: dict | None, pct: bool = False, ci: bool = False) -> str:
            if not v or v.get("mean") is None:
                return "-"
            if pct:
                return f"{v['mean']:.0%}"
            c = f" ±{v['ci']:.2f}" if ci and v.get("ci") is not None else ""
            return f"{v['mean']:+.3f}{c}"

        lines = [f"# Progress vs {vs}", "",
                 "## 4-player Tractor, duplicate deals", "",
                 "Every deal is played twice with the cards swapped between the teams, so the luck of the deal cancels. "
                 "*gain* = levels the network's team gained minus the other team's, per round (0 = even), with its 95% "
                 "interval over deal pairs; *ahead / behind* = share of deal pairs it came out ahead on / behind on.", "",
                 "| update | gain | ahead | behind | keeps landlord | takes landlord |", "|---|---|---|---|---|---|"]
        for r in self.evals:
            d = r.get("duplicate")
            if d:
                lines.append(f"| {r['update']} | {f(d['gain'], ci=True)} | {f(d['ahead'], True)} | {f(d['behind'], True)} | "
                             f"{f(d['keep_landlord'], True)} | {f(d['take_landlord'], True)} |")

        def pair(sk: dict, k: str, fmt: str) -> str:
            a_, b_ = sk["net"].get(k), sk["opp"].get(k)
            g = lambda v: "-" if v is None else (f"{v:.0%}" if fmt == "%" else fmt.format(v))  # noqa: E731
            return f"{g(a_)} / {g(b_)}"
        lines += ["", "## Skills on those deals (network / opponent, per player per round)", "",
                  "The same cards for both sides, so differences are play. *throws*: multi-unit leads tried, *failed*: "
                  "the share a hidden hand beat (card counting); *trump leads*: drawing trumps; *ruff wins*: side-suit "
                  "tricks won by trumping in (voids, often from the kitty); *fed / given*: points played into tricks "
                  "partners / opponents won; *last trick*: won by the player's side; *kitty lost*: kitty points × "
                  "multiplier conceded as defenders.", "",
                  "| update | throws | failed | trump leads | ruff wins | fed | given | last trick | kitty lost |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in self.evals:
            sk = (r.get("duplicate") or {}).get("skills")
            if sk:
                lines.append(f"| {r['update']} | {pair(sk, 'throws', '{:.2f}')} | {pair(sk, 'throwFailRate', '%')} | "
                             f"{pair(sk, 'trumpLeads', '{:.2f}')} | {pair(sk, 'ruffWins', '{:.2f}')} | "
                             f"{pair(sk, 'pointsFed', '{:.1f}')} | {pair(sk, 'pointsGiven', '{:.1f}')} | "
                             f"{pair(sk, 'lastTrick', '%')} | {pair(sk, 'kittyConceded', '{:.1f}')} |")
        lines += ["", "## 4-player Tractor, full matches (duplicate pairs)", "",
                  "Whole matches from everyone at 2 until a team passes A. *wins* = how often the network's team wins "
                  "(even = 50%); *ahead* = how many levels ahead it finishes (a winner counts as one past A).", "",
                  "| update | wins | ahead | rounds per match |", "|---|---|---|---|"]
        for r in self.evals:
            m = r.get("matches")
            if m:
                lines.append(f"| {r['update']} | {f(m['wins'], True)} | {f(m['margin'], ci=True)} | "
                             f"{(m['rounds']['mean'] or 0):.0f} |")
        lines += ["", f"## Other tables vs {vs}", "",
                  "Single rounds from the variety mix (15% of training): one network seat, or every other seat. "
                  "Gain per round (0 = even).", "",
                  "| update | all | Tractor | Finding Friends | 2-3p | 4p | 5p | 6p | 7-8p | 9-12p |", "|---|---|---|---|---|---|---|---|---|---|"]
        buckets = {"2-3p": (2, 3), "4p": (4,), "5p": (5,), "6p": (6,), "7-8p": (7, 8), "9-12p": (9, 10, 11, 12)}
        for r in self.evals:
            v = r.get("variety")
            if not v:
                continue
            cells = []
            for sizes in buckets.values():
                xs = [(v[f"players:{p}"]["mean"], v[f"players:{p}"]["n"]) for p in sizes
                      if v.get(f"players:{p}", {}).get("mean") is not None]
                n = sum(k for _, k in xs)
                cells.append(f"{sum(m * k for m, k in xs) / n:+.3f}" if n else "-")
            lines.append(f"| {r['update']} | {f(v.get('all'))} | {f(v.get('mode:tractor'))} | "
                         f"{f(v.get('mode:findingFriends'))} | " + " | ".join(cells) + " |")
        with open(os.path.join(self.dir, "progress.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def log_update(self, stats: dict, seconds: float, started: float, first_update: int, phase: str) -> None:
        a = self.args
        rate = stats.get("transitions", 0) / max(1e-6, seconds)
        gains = {k: (float(np.mean(v)) if v else None) for k, v in self.window.items()}
        done = self.update - first_update
        eta = (time.perf_counter() - started) / max(1, done) * (a.updates - self.update)
        g = "  ".join(f"vs {k} {v:+.3f}" for k, v in gains.items() if v is not None) or "-"
        if self.landlord_wins:
            g += f" | self-play landlord wins {np.mean(self.landlord_wins):.1%}"
        keys = ("policy_loss", "value_loss", "belief_loss", "belief_holder_acc", "entropy", "teacher_agreement", "approx_kl",
                "clipfrac", "explained_variance")
        s = " ".join(f"{k}={stats[k]:.3f}" for k in keys if k in stats)
        self.log(f"update {self.update}/{a.updates} [{phase}] {seconds:.1f}s {rate:,.0f} trans/s | rounds {self.episodes} "
                 f"aborted {self.aborted} | gain {g} | {s} | lr {stats.get('lr', 0):.1e} | eta {fmt_duration(eta)}")
        row = {"update": self.update, "seconds": seconds, "episodes": self.episodes, "aborted": self.aborted,
               "decisions": self.decisions, "gain": gains,
               "selfplay_landlord_wins": float(np.mean(self.landlord_wins)) if self.landlord_wins else None, **stats}
        self.append_jsonl("metrics.jsonl", row)
        if self.tb:
            for k, v in stats.items():
                self.tb.add_scalar(f"train/{k}", v, self.update)
            for k, v in gains.items():
                if v is not None:
                    self.tb.add_scalar(f"gain/{k}", v, self.update)

    # ------------------------------------------------------------ loop

    def train(self) -> None:
        a = self.args
        started, first = time.perf_counter(), self.update
        try:
            while self.update < a.updates:
                t0 = time.perf_counter()
                phase = self.phase()
                pending = self.collect()
                adv = self.advantages(pending)
                stats = self.learn(pending, adv)
                self.update += 1
                self.log_update(stats, time.perf_counter() - t0, started, first, phase)
                final = self.update == a.updates
                full = a.match_eval_every > 0 and (self.update % a.match_eval_every == 0 or final)
                if (a.eval_every > 0 and self.update % a.eval_every == 0) or full or (final and a.eval_every > 0):
                    self.evaluate(full)  # saves first: it evaluates the exported file
                elif self.update % a.save_every == 0 or final:
                    self.save()
                if self.update % a.refresh_every == 0:
                    self.refresh_opponents()
                if os.path.exists(os.path.join(self.dir, "STOP")):
                    self.log("STOP file found")
                    break
        except KeyboardInterrupt:
            self.log("interrupted")
        finally:
            path = self.save()
            self.log(f"saved {path}")

    def close(self) -> None:
        for w in self.workers:
            w.env.close()
        if self.tb:
            self.tb.close()


def main(argv: list[str] | None = None) -> str:
    t = Trainer(parse_args(argv))
    try:
        t.setup()
        t.train()
    finally:
        t.close()
    return t.dir


if __name__ == "__main__":
    main()
