"""Python-side checks: the v3 network's shape, policy export round trip,
TypeScript / PyTorch logit parity, the bridge under random play (privileged
features, benchmark seats, duplicate pairs), and a tiny train -> checkpoint ->
resume -> evaluate run. Needs `npm run train:build`.

    python -m pytest training/tractor -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev  # noqa: E402
import train  # noqa: E402
from env_client import ROLE_LEARNER, TractorEnv, lineup, random_choices  # noqa: E402
from model import TractorNet, export_policy, load_policy, policy_inputs, rows_to_tensors  # noqa: E402

BENCH = ev.DEFAULT_BENCHMARK
HAVE_BENCH = os.path.exists(BENCH)
needs_bench = pytest.mark.skipif(not HAVE_BENCH, reason="no benchmark policy file (training/benchmarks/)")


def _rows(batch):
    return np.flatnonzero(batch.role >= 0)


def _critic(env) -> dict:
    return {"privGlobal": env.priv_global_dim, "privSeat": env.priv_seat_dim, "seatWidth": 64, "hidden": 256}


def test_default_architecture_matches_the_report():
    with TractorEnv() as env:
        assert (env.global_dim, env.seat_dim, env.max_seats, env.cand_dim) == (679, 637, 12, 115)
        assert (env.encoder_version, env.action_space_version, env.policy_format) == (3, 2, "tractor/seat-mlp@3")
        net = TractorNet(env.global_dim, env.seat_dim, env.max_seats, env.cand_dim, critic=_critic(env))
    assert net.num_parameters(actor_only=True) == 1_355_457
    actor = {k for k in net.state_dict() if not k.startswith("critic")}
    assert actor == {
        "seat.0.weight", "seat.0.bias", "seat.2.weight", "seat.2.bias",
        "trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias", "trunk.4.weight", "trunk.4.bias",
        "cand.0.weight", "cand.0.bias", "cand.2.weight", "cand.2.bias", "joint_state.weight", "joint_state.bias",
        "joint_cand.weight", "score.weight", "score.bias"}


def test_policy_export_round_trip_and_typescript_parity(tmp_path):
    """f32 files reload exactly; f16 files (the default) are about half the size and
    play almost like f32; the TypeScript runtime computes what Python computes from
    the same file; the critic is never exported."""
    torch.manual_seed(0)
    with TractorEnv() as env:
        batch = env.reset(num_envs=8, seed=3, lineups=[lineup("all")], scenario={"focus": 0.5})
        net = TractorNet(env.global_dim, env.seat_dim, env.max_seats, env.cand_dim, seat=[32, 16], trunk=[64, 32],
                         cand=[32], joint=48, critic=_critic(env))
        p32, p16 = os.path.join(tmp_path, "p32.json"), os.path.join(tmp_path, "p16.json")
        export_policy(net, p32, env.versions, meta={"note": "test"}, dtype="f32")
        export_policy(net, p16, env.versions, meta={"note": "test"})
        assert os.path.getsize(p16) < 0.55 * os.path.getsize(p32)
        back32, back16 = load_policy(p32, env.versions), load_policy(p16, env.versions)
        for _ in range(40):  # walk into every kind of decision, several table sizes
            rows = _rows(batch)
            x = rows_to_tensors(batch, rows, "cpu")
            with torch.no_grad():
                a, _ = net(*policy_inputs(x))
                b, _ = back32(*policy_inputs(x))
                h, _ = back16(*policy_inputs(x))
            assert torch.equal(a, b)
            for path, ref in ((p32, a), (p16, h)):
                ts = env.policy_logits(path)
                for r, logits in zip(rows, ref):
                    k = int(batch.count[r])
                    assert len(ts[r]) == k
                    assert np.allclose(np.asarray(ts[r]), logits[:k].numpy(), atol=1e-4), (path, ts[r], logits[:k])
                    assert (logits[k:] < -1e8).all()
            live = a > -1e8
            assert torch.allclose(h[live], a[live], atol=2e-2, rtol=2e-2)
            batch = env.step(random_choices(batch, np.random.default_rng(1)))
    with open(p16, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["meta"] == {"note": "test"}
    assert {t["dtype"] for t in doc["tensors"].values()} == {"f16"}
    assert not any(k.startswith("critic") for k in doc["tensors"])


@needs_bench
def test_random_bridge_keeps_features_sane():
    rng = np.random.default_rng(0)
    kinds = set()
    with TractorEnv() as env:
        batch = env.reset(num_envs=8, lineups=[lineup("team", "benchmark"), lineup("one", "opponent")], teacher=True,
                          privileged=True, points_shaping=0.5, benchmark_policy=BENCH, scenario={"focus": 0.7})
        rounds = 0
        for _ in range(300):
            live = batch.role >= 0
            assert (batch.count[live] > 0).all() and (batch.count[live] <= env.max_candidates).all()
            assert (batch.seat_count[live] == batch.players[live]).all()
            for arr in (batch.glob, batch.seats, batch.cand, batch.priv_glob, batch.priv_seats):
                assert np.isfinite(arr).all() and np.abs(arr).max(initial=0) <= 4
            assert batch.priv_seats.shape == (len(batch.seats), env.priv_seat_dim)
            assert (batch.teacher[batch.role != ROLE_LEARNER] == -1).all()
            assert (batch.teacher < np.maximum(1, batch.count)).all()
            kinds.update(batch.kind[live].tolist())
            batch = env.step(random_choices(batch, rng))
            rounds += len(batch.episodes)
            for e in batch.episodes:
                assert not e["aborted"]
        assert rounds > 5 and {0, 1, 3} <= kinds


@needs_bench
def test_duplicate_eval_against_the_benchmark_pairs_every_deal():
    with TractorEnv() as env:
        net = TractorNet(env.global_dim, env.seat_dim, env.max_seats, env.cand_dim, seat=[16], trunk=[32], cand=[16], joint=16)
    eps = ev.play_duplicate(net, deals=4, procs=2, envs=2)
    s = ev.summarize_duplicate(eps)
    assert s["pairs"] == 4 and s["gain"]["mean"] is not None
    assert all("benchmark" in e["roles"] for e in eps)


@needs_bench
def test_tiny_training_run_checkpoints_resumes_and_evaluates(tmp_path):
    args = ["--runs-dir", str(tmp_path), "--run", "t", "--no-build", "--device", "cpu", "--workers", "3",
            "--mix", "self=1,past=1,bench=1", "--envs-per-worker", "4", "--rollout", "12",
            "--seat", "16", "--trunk", "32", "--cand", "16", "--joint", "16", "--critic-hidden", "16",
            "--bc-updates", "2", "--updates", "3", "--save-every", "1", "--refresh-every", "1",
            "--eval-every", "3", "--eval-deals", "2", "--eval-procs", "1", "--match-eval-every", "3",
            "--match-eval-pairs", "1", "--variety-eval-rounds", "8", "--minibatch", "32", "--grad-accum", "2"]
    run_dir = train.main(args)
    assert os.path.exists(os.path.join(run_dir, "ckpt", "latest.pt"))
    for u in (1, 2, 3):
        assert os.path.exists(os.path.join(run_dir, "policies", f"update_{u}.json"))
    with open(os.path.join(run_dir, "policies", "update_3.json"), encoding="utf-8") as f:
        assert {t["dtype"] for t in json.load(f)["tensors"].values()} == {"f16"}
    metrics = [json.loads(x) for x in open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8")]
    assert [m["update"] for m in metrics] == [1, 2, 3]
    assert metrics[0]["imitation"] == 1.0 and metrics[2]["imitation"] == 0.0
    assert "teacher_agreement" in metrics[0] and "approx_kl" in metrics[2] and np.isfinite(metrics[2]["policy_loss"])
    evals = [json.loads(x) for x in open(os.path.join(run_dir, "evals.jsonl"), encoding="utf-8")]
    assert evals and evals[0]["duplicate"]["pairs"] >= 2
    assert evals[0]["matches"]["pairs"] >= 1 and evals[0]["variety"]["rounds"] >= 8
    progress = open(os.path.join(run_dir, "progress.md"), encoding="utf-8").read()
    assert "## 4-player Tractor, duplicate deals" in progress and "## Other tables vs the v2 benchmark" in progress
    # Resume: picks up at update 3 and trains one more.
    train.main(args + ["--resume", "--updates", "4", "--eval-every", "0", "--match-eval-every", "0"])  # later flags win
    metrics = [json.loads(x) for x in open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8")]
    assert [m["update"] for m in metrics] == [1, 2, 3, 4]
    assert "resumed" in open(os.path.join(run_dir, "train.log"), encoding="utf-8").read()
    # The exported policy evaluates from the command line.
    s = ev.main([os.path.join(run_dir, "policies", "update_4.json"), "--deals", "2", "--procs", "1", "--device", "cpu", "--json"])
    assert s["duplicate"]["pairs"] >= 2
