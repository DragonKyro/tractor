"""Policy network for Tractor (and its training-only critic), checkpoints, and
the policy export the TypeScript side runs (src/ai/nn/runtime.ts, the same
layers in the same order).

    seat encoder  each present seat block (637) -> Linear+ReLU per width in arch["seat"] -> e_s,
                  shared by all 12 seat slots; absent seats give 0
    trunk         [global block (679), e_0 ... e_11] -> Linear+ReLU per width in arch["trunk"] -> h
    cand tower    each candidate (115) -> Linear+ReLU per width in arch["cand"] -> c_k
    scorer        logit_k = score(ReLU(joint_state(h) + joint_cand(c_k)))      one per candidate
    critic        V = critic(ReLU([h, privileged global, critic_seat(each seat's hand)]))  -- training only

The critic sees every hand (privileged.ts): a perfect-information critic for
an imperfect-information policy, so advantages are far less noisy. It is never
exported; policy files hold the actor alone ("tractor/seat-mlp@3"), every
tensor base64 little-endian float16 (default) or float32 with its `dtype`.
"""

from __future__ import annotations

import base64
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

POLICY_FORMAT = "tractor/seat-mlp@3"
GAME = "tractor"
ILLEGAL_LOGIT = -1e9
DEFAULT_ARCH = {"seat": [128, 64], "trunk": [512, 512, 256], "cand": [128, 128], "joint": 256}
DEFAULT_CRITIC = {"seatWidth": 64, "hidden": 256}


def _mlp(n_in: int, widths: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for w in widths:
        layers += [nn.Linear(n_in, w), nn.ReLU()]
        n_in = w
    return nn.Sequential(*layers)


class TractorNet(nn.Module):
    def __init__(self, global_dim: int, seat_dim: int, max_seats: int, cand_dim: int, seat: list[int] | None = None,
                 trunk: list[int] | None = None, cand: list[int] | None = None, joint: int | None = None,
                 critic: dict | None = None):
        """critic: {"privGlobal", "privSeat", "seatWidth", "hidden"} for a privileged critic, or None (actor only)."""
        super().__init__()
        seat = list(seat or DEFAULT_ARCH["seat"])
        trunk = list(trunk or DEFAULT_ARCH["trunk"])
        cand = list(cand or DEFAULT_ARCH["cand"])
        joint = int(joint or DEFAULT_ARCH["joint"])
        self.arch = {"globalDim": global_dim, "seatDim": seat_dim, "maxSeats": max_seats, "candDim": cand_dim,
                     "seat": seat, "trunk": trunk, "cand": cand, "joint": joint}
        self.max_seats = max_seats
        self.seat = _mlp(seat_dim, seat)
        self.trunk = _mlp(global_dim + max_seats * seat[-1], trunk)
        self.cand = _mlp(cand_dim, cand)
        self.joint_state = nn.Linear(trunk[-1], joint)
        self.joint_cand = nn.Linear(cand[-1], joint, bias=False)
        self.score = nn.Linear(joint, 1)
        self.critic_arch = dict(critic) if critic else None
        if critic:
            cw = int(critic.get("seatWidth", DEFAULT_CRITIC["seatWidth"]))
            hidden = int(critic.get("hidden", DEFAULT_CRITIC["hidden"]))
            self.critic_seat = _mlp(int(critic["privSeat"]), [cw])
            self.critic = nn.Sequential(nn.Linear(trunk[-1] + int(critic["privGlobal"]) + max_seats * cw, hidden),
                                        nn.ReLU(), nn.Linear(hidden, 1))

    def embed(self, glob: torch.Tensor, seats: torch.Tensor, seat_mask: torch.Tensor) -> torch.Tensor:
        """glob (B, G), seats (B, S, seat_dim), seat_mask (B, S) bool -> h (B, trunk[-1])."""
        e = self.seat(seats) * seat_mask.unsqueeze(-1)
        return self.trunk(torch.cat([glob, e.flatten(1)], 1))

    def forward(self, glob, seats, seat_mask, cand, cand_mask) -> tuple[torch.Tensor, torch.Tensor]:
        """-> masked logits (B, K), h (B, trunk[-1]) (for the critic)."""
        h = self.embed(glob, seats, seat_mask)
        c = self.cand(cand)
        j = F.relu(self.joint_state(h).unsqueeze(1) + self.joint_cand(c))
        return masked_logits(self.score(j).squeeze(-1), cand_mask), h

    def value(self, h, priv_glob, priv_seats, seat_mask) -> torch.Tensor:
        """The critic: (B,) expected round reward of the deciding seat."""
        e = self.critic_seat(priv_seats) * seat_mask.unsqueeze(-1)
        return self.critic(torch.cat([h, priv_glob, e.flatten(1)], 1)).squeeze(-1)

    def num_parameters(self, actor_only: bool = False) -> int:
        return sum(p.numel() for k, p in self.named_parameters() if not (actor_only and k.startswith("critic")))


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, ILLEGAL_LOGIT)


def net_from_arch(arch: dict, critic: dict | None = None) -> TractorNet:
    return TractorNet(arch["globalDim"], arch["seatDim"], arch["maxSeats"], arch["candDim"], arch["seat"],
                      arch["trunk"], arch["cand"], arch["joint"], critic)


# ---------------------------------------------------------------- batches

def _starts(c: np.ndarray) -> np.ndarray:
    o = np.zeros(len(c), np.int64)
    if len(c) > 1:
        o[1:] = np.cumsum(c)[:-1]
    return o


def packed_positions(offsets: np.ndarray, count: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For the given rows: (index of the row within `rows`, position in the packed array) of each packed item."""
    c = count[rows]
    total = int(c.sum())
    which = np.repeat(np.arange(len(rows)), c)
    pos = np.repeat(offsets[rows], c) + (np.arange(total) - np.repeat(_starts(c), c))
    return which, pos


def rows_to_tensors(batch, rows: np.ndarray, device, privileged: bool = False, max_seats: int = 12) -> dict:
    """The network's inputs for some rows of an env Batch: seat blocks scattered to their slots
    (B, 12, seat_dim) with a mask, candidates padded to the longest list (B, K, cand_dim) with a mask."""
    n, S = len(rows), max_seats
    which, pos = packed_positions(batch.seat_offsets, batch.seat_count, rows)
    slots = batch.seat_slot[pos]
    seats = np.zeros((n, S, batch.seats.shape[1]), np.float32)
    seats[which, slots] = batch.seats[pos]
    seat_mask = np.zeros((n, S), bool)
    seat_mask[which, slots] = True
    c = batch.count[rows]
    k = max(1, int(c.max(initial=1)))
    ar = np.arange(k)
    cand_mask = ar[None, :] < c[:, None]
    cpos = np.where(cand_mask, batch.offsets[rows][:, None] + ar[None, :], 0)
    cand = batch.cand[cpos] if len(batch.cand) else np.zeros((n, k, batch.cand.shape[1]), np.float32)
    cand[~cand_mask] = 0
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)  # noqa: E731
    out = {"glob": t(batch.glob[rows]), "seats": t(seats), "seat_mask": t(seat_mask), "cand": t(cand), "cand_mask": t(cand_mask)}
    if privileged:
        ps = np.zeros((n, S, batch.priv_seats.shape[1]), np.float32)
        ps[which, slots] = batch.priv_seats[pos]
        out["priv_glob"] = t(batch.priv_glob[rows])
        out["priv_seats"] = t(ps)
    return out


def policy_inputs(x: dict) -> tuple:
    return x["glob"], x["seats"], x["seat_mask"], x["cand"], x["cand_mask"]


# ---------------------------------------------------------------- policy files

DTYPES = {"f16": "<f2", "f32": "<f4"}
F16_MAX = 65504.0


def _tensor(t: torch.Tensor, dtype: str) -> dict:
    a = t.detach().float().cpu().numpy()
    if dtype == "f16" and np.abs(a).max(initial=0) > F16_MAX:
        raise ValueError(f"a weight of magnitude {np.abs(a).max():.3g} does not fit float16")
    data = base64.b64encode(np.ascontiguousarray(a, dtype=DTYPES[dtype]).tobytes()).decode()
    return {"shape": list(t.shape), "dtype": dtype, "data": data}


def export_policy(net: TractorNet, path: str, versions: dict, meta: dict | None = None, dtype: str = "f16") -> None:
    """The actor, in the format src/ai/nn/runtime.ts reads. float16 rounds each
    weight to about 3 significant digits; reload the file (load_policy) to get
    exactly the network the game will run."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}")
    doc = {
        "format": POLICY_FORMAT,
        "game": GAME,
        **versions,
        "arch": net.arch,
        "tensors": {k: _tensor(v, dtype) for k, v in net.state_dict().items() if not k.startswith("critic")},
        "meta": meta or {},
    }
    _atomic_write(path, json.dumps(doc))


def load_policy(path: str, versions: dict | None = None, device: str | torch.device = "cpu") -> TractorNet:
    """An actor-only network from a policy file (opponents, evals)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != POLICY_FORMAT or doc.get("game") != GAME:
        raise ValueError(f"{path} is not a {POLICY_FORMAT} {GAME} policy ({doc.get('format')}, {doc.get('game')})")
    if versions:
        mine = {k: doc.get(k) for k in versions}
        if mine != versions:
            raise ValueError(f"{path} was trained on {mine}, this build is {versions}")
    net = net_from_arch(doc["arch"])
    state = {k: torch.from_numpy(np.frombuffer(base64.b64decode(t["data"]), dtype=DTYPES[t.get("dtype", "f32")])
                                 .astype(np.float32).reshape(t["shape"]))
             for k, t in doc["tensors"].items()}
    net.load_state_dict(state)
    return net.to(device).eval()


def load_actor_into(net: TractorNet, path: str) -> None:
    """Copy a policy file's actor weights into `net` (its critic keeps its own)."""
    src = load_policy(path)
    net.load_state_dict(src.state_dict(), strict=False)


# ---------------------------------------------------------------- checkpoints

def save_checkpoint(path: str, net: TractorNet, optimizer: torch.optim.Optimizer, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), "arch": net.arch,
                "critic": net.critic_arch, **state}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def net_from_checkpoint(ckpt: dict, device: str | torch.device = "cpu") -> TractorNet:
    net = net_from_arch(ckpt["arch"], ckpt.get("critic")).to(device)
    net.load_state_dict(ckpt["model"])
    return net


def _atomic_write(path: str, text: str) -> None:
    """Readers (opponent pools, evals) never see a half-written file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
