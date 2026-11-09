"""Evaluate a Tractor policy (greedy, as the shipped AI plays).

    python training/tractor/evaluate.py <policy.json>                       # vs the shipped AI, 4p Tractor
    python training/tractor/evaluate.py <policy.json> --matches 24          # + full duplicate matches
    python training/tractor/evaluate.py <policy.json> --variety 512         # + other tables
    python training/tractor/evaluate.py <policy.json> --vs <other policy.json>

Duplicate deals (the main score): every deal is played twice with the same
cards, the network's team holding one side's cards and then the other's, the
opponent (the shipped AI, src/ai/nn/policy.json, or `--vs`) in the other
seats. A pair's score is the mean of
the two halves, so the luck of the deal cancels and far fewer deals give the
same precision. On the focus table, 4-player Tractor, the network is one
fixed-partner team.

    gain     levels the network's side gained minus the other side's, per round
             (0 = as good as the opponent), with its 95% interval over deal pairs
    pairs    share of deal pairs the network came out ahead on / behind on
    landlord how often its side keeps the landlord seat / takes it
    skills   per player per round, the network's seats next to the opponent's
             on the same cards (skills.ts): throws tried and the share that
             failed, trump leads, side-suit tricks won by trumping in, points
             fed to partners vs given to opponents, the last trick won, and
             kitty points conceded as defenders

Full matches (`--matches N`): N duplicate pairs of whole matches from 2 to
past A: how often the network's team wins (the pair average) and how many
levels ahead it finishes (a winner counts as one past A).

Other tables (`--variety N`): N single rounds from the curriculum's variety
mix (2-12 players, Finding Friends and Tractor) vs the same opponent, one network
seat or every other seat, by mode and table size: a check that the network
still plays every form of the game.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_client import ROLE_LEARNER, ROLE_OPPONENT, TractorEnv, lineup  # noqa: E402
from model import TractorNet, load_policy, policy_inputs, rows_to_tensors  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SHIPPED_POLICY = os.path.join(REPO, "src", "ai", "nn", "policy.json")
EVAL_SEED = 424_242
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "NT"]


def greedy(net: TractorNet, batch, rows: np.ndarray, device) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros(0, np.int64)
    with torch.no_grad():
        logits, _ = net(*policy_inputs(rows_to_tensors(batch, rows, device, max_seats=net.max_seats)))
    return logits.argmax(-1).cpu().numpy()


def run_jobs(net: TractorNet, jobs: list[dict], device="cpu", opponent: TractorNet | None = None) -> list[dict]:
    """Each job ({"name", **TractorEnv.reset kwargs}) in its own env process, all
    stepped together; returns every finished episode tagged with its job's name."""
    out: list[dict] = []
    net.eval()
    envs = [TractorEnv() for _ in jobs]
    try:
        for n in [net] + ([opponent] if opponent is not None else []):
            a, e = n.arch, envs[0]
            if (a["globalDim"], a["seatDim"], a["maxSeats"], a["candDim"]) != (e.global_dim, e.seat_dim, e.max_seats, e.cand_dim):
                raise ValueError(f"policy takes {a['globalDim']} + {a['maxSeats']}x{a['seatDim']} / {a['candDim']} inputs; this "
                                 f"build's encoder makes {e.global_dim} + {e.max_seats}x{e.seat_dim} / {e.cand_dim} "
                                 f"(encoder v{e.encoder_version})")
        batches = [env.reset(**{k: v for k, v in job.items() if k != "name"}) for env, job in zip(envs, jobs)]
        while any((b.role >= 0).any() for b in batches):
            live = [k for k, b in enumerate(batches) if (b.role >= 0).any()]
            for k in live:
                b = batches[k]
                choices = np.zeros(len(b.role), np.int64)
                mine = np.flatnonzero(b.role == ROLE_LEARNER)
                choices[mine] = greedy(net, b, mine, device)
                theirs = np.flatnonzero(b.role == ROLE_OPPONENT)
                if len(theirs):
                    choices[theirs] = greedy(opponent or net, b, theirs, device)
                envs[k].send_step(choices)
            for k in live:
                batches[k] = envs[k].recv_batch()
                for e in batches[k].episodes:
                    out.append({**e, "job": jobs[k]["name"]})
    finally:
        for env in envs:
            env.close()
    return out


def _opponent(vs: str | TractorNet, device) -> TractorNet:
    """A policy file, or a network already loaded."""
    return (load_policy(vs) if isinstance(vs, str) else vs).to(device)


def _split(total: int, procs: int, envs: int) -> tuple[int, int, int]:
    """(processes, envs per process, items per env) covering at least `total` items."""
    procs = max(1, min(procs, total))
    envs = max(1, min(envs, math.ceil(total / procs)))
    return procs, envs, max(1, math.ceil(total / (procs * envs)))


def play_duplicate(net: TractorNet, deals: int = 256, vs: str | TractorNet = SHIPPED_POLICY, procs: int = 8,
                   envs: int = 16, seed: int = EVAL_SEED, device="cpu", matches: bool = False,
                   scenario: dict | None = None) -> list[dict]:
    """`deals` duplicate pairs (rounds, or whole matches) on the focus table, network team vs `vs`."""
    procs, envs, per_env = _split(deals, procs, envs)
    jobs = [{"name": f"dup{k}", "num_envs": envs, "seed": seed + 7919 * k, "lineups": [lineup("team")],
             "scenario": scenario or {"focus": 1, "turnBidding": 0}, "quota": 2 * per_env, "duplicate": True,
             "matches": matches} for k in range(procs)]
    return run_jobs(net, jobs, device, _opponent(vs, device))


def play_variety(net: TractorNet, rounds: int = 512, vs: str | TractorNet = SHIPPED_POLICY, procs: int = 4,
                 envs: int = 32, seed: int = EVAL_SEED + 3, device="cpu") -> list[dict]:
    """Single rounds from the variety mix (every other table) vs `vs`."""
    procs, envs, per_env = _split(rounds, procs, envs)
    jobs = [{"name": f"var{k}", "num_envs": envs, "seed": seed + 7919 * k,
             "lineups": [lineup("one"), lineup("team")], "scenario": {"focus": 0},
             "quota": per_env} for k in range(procs)]
    return run_jobs(net, jobs, device, _opponent(vs, device))


# ---------------------------------------------------------------- summaries

def _mean_ci(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "ci": None, "n": 0}
    a = np.asarray(xs, np.float64)
    ci = 1.96 * a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else None
    return {"mean": float(a.mean()), "ci": None if ci is None else float(ci), "n": len(a)}


def _learners(e: dict) -> list[int]:
    return [s for s, r in enumerate(e["roles"]) if r == "learner"]


def _rank(level: str, won: bool) -> int:
    """0 for 2 ... 12 for A; a match winner counts as one past the final rank."""
    if won:
        return RANKS.index("NT") + 1 if level == "NT" else RANKS.index("A") + 1
    return RANKS.index(level)


def _pairs(episodes: list[dict]) -> list[tuple[dict, dict]]:
    by: dict[tuple, dict] = {}
    for e in episodes:
        if e["aborted"] or e["truncated"]:
            continue
        by.setdefault((e["job"], e["env"], e["pair"]), {})[e["flip"]] = e
    return [(p[0], p[1]) for p in by.values() if 0 in p and 1 in p]


SKILLS = ("throws", "throwFails", "trumpLeads", "ruffWins", "pointsFed", "pointsGiven", "lastTrick", "kittyBuried",
          "kittyConceded")


def summarize_skills(episodes: list[dict]) -> dict:
    """Per player per round, the network's seats and the others' (the same deals, so the same cards)."""
    tot = {side: collections.defaultdict(float) for side in ("net", "opp")}
    n = {"net": 0, "opp": 0}
    for e in episodes:
        sk = e.get("skills")
        if not sk or e["aborted"] or e["truncated"]:
            continue
        mine = set(_learners(e))
        for seat in range(e["players"]):
            side = "net" if seat in mine else "opp"
            n[side] += 1
            for k in SKILLS:
                tot[side][k] += sk[k][seat]
    out: dict = {}
    for side in ("net", "opp"):
        m = max(1, n[side])
        out[side] = {k: tot[side][k] / m for k in SKILLS}
        out[side]["throwFailRate"] = tot[side]["throwFails"] / tot[side]["throws"] if tot[side]["throws"] else None
    return out


def summarize_duplicate(episodes: list[dict]) -> dict:
    """Duplicate rounds: per-pair gain, pairs won / lost, landlord keep / take, skills."""
    gains, keep, take = [], [], []
    for a, b in _pairs(episodes):
        g = [float(np.mean([e["gain"][s] for s in _learners(e)])) for e in (a, b)]
        gains.append(float(np.mean(g)))
        for e in (a, b):
            mine = _learners(e)
            if e["landlord"] in mine:
                keep.append(float(e["landlordWon"]))
            elif not any(s in e["landlordTeam"] for s in mine):
                take.append(float(not e["landlordWon"]))
    paired = [e for pair in _pairs(episodes) for e in pair]
    return {"gain": _mean_ci(gains), "ahead": _mean_ci([float(x > 0) for x in gains]),
            "behind": _mean_ci([float(x < 0) for x in gains]), "keep_landlord": _mean_ci(keep),
            "take_landlord": _mean_ci(take), "pairs": len(gains), "skills": summarize_skills(paired)}


def summarize_duplicate_matches(episodes: list[dict]) -> dict:
    """Duplicate matches: per pair, match wins (0, 1/2, 1) and the finishing margin in levels."""
    wins, margin, rounds = [], [], []
    for pair in _pairs(episodes):
        w, m = [], []
        for e in pair:
            mine, winners = _learners(e), set(e["winners"])
            others = [s for s in range(e["players"]) if s not in mine]
            rank = [_rank(lv, s in winners) for s, lv in enumerate(e["finalLevels"])]
            w.append(float(any(s in winners for s in mine)))
            m.append(float(np.mean([rank[s] for s in mine]) - np.mean([rank[s] for s in others])))
            rounds.append(float(e["rounds"]))
        wins.append(float(np.mean(w)))
        margin.append(float(np.mean(m)))
    return {"wins": _mean_ci(wins), "margin": _mean_ci(margin), "rounds": _mean_ci(rounds), "pairs": len(wins)}


def summarize_rounds(episodes: list[dict]) -> dict:
    """Single rounds: gain by lineup, mode, table size, level class (+ landlord keep / take)."""
    groups: dict[str, list[float]] = collections.defaultdict(list)
    keep, take = [], []
    for e in episodes:
        if e["aborted"] or e["truncated"]:
            continue
        mine = _learners(e)
        if not mine:
            continue
        g = float(np.mean([e["gain"][s] for s in mine]))
        for key in ("all", f"lineup:{e['lineup']}", f"mode:{e['mode']}", f"players:{e['players']}", f"level:{e['levelClass']}"):
            groups[key].append(g)
        if e["landlord"] in mine:
            keep.append(float(e["landlordWon"]))
        elif not any(s in e["landlordTeam"] for s in mine):
            take.append(float(not e["landlordWon"]))
    out = {k: _mean_ci(v) for k, v in sorted(groups.items())}
    out["keep_landlord"] = _mean_ci(keep)
    out["take_landlord"] = _mean_ci(take)
    out["rounds"] = sum(1 for e in episodes if not (e["aborted"] or e["truncated"]))
    return out


def _f(v: dict | None, pct: bool = False) -> str:
    if not v or v.get("mean") is None:
        return "-"
    if pct:
        return f"{v['mean']:.0%}"
    ci = f" +/- {v['ci']:.2f}" if v.get("ci") is not None else ""
    return f"{v['mean']:+.3f}{ci}"


def format_skills(sk: dict) -> str:
    def row(k: str, label: str, fmt: str = "{:.2f}") -> str:
        a, b = sk["net"].get(k), sk["opp"].get(k)
        f = lambda v: "-" if v is None else (f"{v:.0%}" if fmt == "%" else fmt.format(v))  # noqa: E731
        return f"{label} {f(a)} / {f(b)}"
    return "  skills, network / opponent, per player per round: " + "; ".join([
        row("throws", "throws"), row("throwFailRate", "failed", "%"), row("trumpLeads", "trump leads"),
        row("ruffWins", "ruff wins"), row("pointsFed", "points fed", "{:.1f}"), row("pointsGiven", "points given", "{:.1f}"),
        row("lastTrick", "last trick", "%"), row("kittyConceded", "kitty conceded", "{:.1f}")])


def format_duplicate(s: dict, vs: str) -> str:
    line = (f"vs {vs}, duplicate rounds ({s['pairs']} deal pairs): gain {_f(s['gain'])} per round; "
            f"ahead on {_f(s['ahead'], True)} of pairs, behind on {_f(s['behind'], True)}; "
            f"keeps the landlord seat {_f(s['keep_landlord'], True)}, takes it {_f(s['take_landlord'], True)}")
    return line + ("\n" + format_skills(s["skills"]) if s.get("skills") else "")


def format_duplicate_matches(s: dict, vs: str) -> str:
    return (f"vs {vs}, duplicate matches ({s['pairs']} pairs): wins {_f(s['wins'], True)} (even 50%), finishes "
            f"{_f(s['margin'])} levels ahead, {s['rounds']['mean'] or 0:.0f} rounds per match")


def format_rounds(s: dict, vs: str) -> str:
    def row(prefix: str) -> str:
        return "  " + "  ".join(f"{k.split(':', 1)[1]} {_f(s[k])} (n={s[k]['n']})" for k in s if k.startswith(prefix))
    return "\n".join([f"vs {vs}, other tables ({s['rounds']} rounds): gain {_f(s.get('all'))} per round",
                      row("mode:"), row("players:"),
                      f"  keeps the landlord seat {_f(s['keep_landlord'], True)}, takes it {_f(s['take_landlord'], True)}"])


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy")
    p.add_argument("--vs", default=SHIPPED_POLICY, help="the opponent's policy file (default: the shipped AI)")
    p.add_argument("--deals", type=int, default=256, help="duplicate deal pairs (rounds) on 4-player Tractor")
    p.add_argument("--matches", type=int, default=0, help="also this many duplicate pairs of full matches")
    p.add_argument("--variety", type=int, default=0, help="also this many single rounds on other tables")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    net = load_policy(a.policy, device=a.device)
    s: dict = {}
    lines = []
    vs = os.path.relpath(a.vs, REPO)
    if a.deals > 0:
        s["duplicate"] = summarize_duplicate(play_duplicate(net, a.deals, a.vs, a.procs, seed=a.seed, device=a.device))
        lines.append(format_duplicate(s["duplicate"], vs))
    if a.matches > 0:
        s["matches"] = summarize_duplicate_matches(play_duplicate(net, a.matches, a.vs, a.procs,
                                                                  seed=a.seed + 1, device=a.device, matches=True))
        lines.append(format_duplicate_matches(s["matches"], vs))
    if a.variety > 0:
        s["variety"] = summarize_rounds(play_variety(net, a.variety, a.vs, device=a.device))
        lines.append(format_rounds(s["variety"], vs))
    print(json.dumps(s) if a.json else "\n".join(lines))
    return s


if __name__ == "__main__":
    main()
