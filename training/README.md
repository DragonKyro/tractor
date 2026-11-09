# training/

Offline tooling for the AI players. Nothing here ships to GitHub Pages.

- `simulate.ts`: headless matches between the shipped AI levels through the real reducer (`npm run sim -- 4 6 hard,easy`).
- `tractor/`: the learned AI's training pipeline. A TypeScript env server runs the game; a PyTorch trainer learns from it.
- `benchmarks/v2-update1500.json`: the v2 network from its 1,500th update, which beat the old hand-written heuristic AI (since removed) consistently. It is the benchmark the current network is measured against, and the teacher for distillation. Git ignores it (4.3 MB); without it, `train.py` skips distillation and evals.

## The network (shipped code: `src/ai/nn/`, encoder v3, action space v2)

One network plays every table the game allows: 2–12 players, Tractor and Finding Friends. Training is optimized for 4-player Tractor.

### Inputs (`encode.ts`)

Everything comes from the deciding seat's `Observation`, so nothing hidden leaks in. Cards are counted in **72 trump-relative slots** (`slots.ts`), each count divided by the number of decks:

| Slots | What they hold |
|---|---|
| 0–51 | four side-suit blocks × 13 ranks. The blocks follow the trump suit in a fixed rotation, so every trump suit looks alike. |
| 52–64 | trump-suit cards |
| 65–68 | the trump number in each side suit |
| 69 | the trump number in the trump suit |
| 70–71 | the jokers |

The vector is a **global block** (679 numbers) followed by **12 seat blocks** (637 each), 8,323 in all.

**Seat slots.** Seats are placed relative to the decider, from both ends:

| Slot | Seat |
|---|---|
| 0 | me |
| 1 | the player after me |
| 11 | the player before me |
| others | counting up from 1 and down from 11 |

So the next and previous players keep their slots at any table size. In 4-player Tractor: me 0, left opponent 1, partner 10, right opponent 11. Absent seats are zeros.

| Block | Contents |
|---|---|
| Global: 8 card planes | <ul><li>my hand</li><li>unseen cards</li><li>the trick's winning play</li><li>the kitty as dealt and as buried (landlord only)</li><li>cards turned over from the kitty</li><li>unrevealed friend calls, and the copies still to come before a call's friend copy</li></ul> |
| Global: 103 features | <ul><li>table size, decks, mode, friends hidden, stage</li><li>trump: kind, number, whether it's a point rank, final rank, who is about to pass it</li><li>points against the threshold, unseen points, kitty</li><li>the trick: the led format, its points, whether the winner is me, a partner or an opponent, partners and opponents still to play</li><li>the last trick's points and winner</li><li>bidding</li><li>my role</li><li>rule variants</li></ul> |
| Each seat: 8 card planes | how that seat played each card:<ul><li>**led**</li><li>**followed** (in the led suit)</li><li>**ruffed** (trump on a side-suit lead)</li><li>**discarded** (neither)</li><li>its **first discard** (the classic signal)</li><li>its play **this trick**, and in the **last completed trick**</li><li>cards it has **shown** and still holds: bids, and failed throws taken back</li></ul> |
| Each seat: 61 features | <ul><li>present / me</li><li>slot</li><li>level (and against mine and the final rank)</li><li>side: landlord, known defender or attacker, unknown, my partner</li><li>points captured, tricks won, cards left</li><li>known voids</li><li>this trick: played, winning, leader, next</li><li>last trick: led, won</li><li>penalties</li><li>bids</li><li>dealer</li></ul> |

### Outputs (`decision.ts`): one score per candidate move

Plays are combinatorial, so the network scores a list of up to **64 legal candidates**. Each has **115 features**:
- its cards in the 72 slots
- shape, points, all trump
- for a play: is it a throw, does it win now, is it unbeatable, is it a ruff, what share is discarded
- what it does to my hand: the suits it leaves me void in, pairs broken, its highest and lowest rank, that suit's length
- where the points go if the trick ended now: to my side or the other
- for a lead: whether my partner or an opponent is known to be void in that suit
- for a bid: count, joker bid, how much of my hand would be trump
- for a friend call: which copy, and how many I hold

The decisions:
- **bid**: in a live deal, a seat is asked whenever its legal bids change. "No bid" means "not now".
- **bury**: the kitty, one card at a time.
- **call**: friend calls, one at a time.
- **play**: every lead (singles, tuples, tractors, unbeatable throws), and every follow the rules allow:
  - the legal matchings of the led shape
  - **every distinct card when one card is free to choose**: any discard when void, any card of the suit when following a single
  - a cheapest fill taken from each suit first
  - a fill that keeps pairs

Single-candidate steps play themselves. In the app, a network seat that declined to bid isn't asked again until its options change (`BidMemo` in `agent.ts`), as in training.

### Architecture (`runtime.ts`, format `tractor/seat-mlp@3`)

```
seat encoder   each present seat block 637 → 128 → 64 (ReLU), shared by all 12 slots; absent seats 0
trunk          [global 679, 12 × 64] = 1447 → 512 → 512 → 256 (ReLU)              = h
cand tower     each candidate 115 → 128 → 128 (ReLU)                              = c_k
scorer         logit_k = w · ReLU(J_s·h + J_c·c_k + b) + b'    (joint width 256)
```

That is **1,355,457 parameters**. A float16 policy file is 3.6 MB.

The shared seat encoder reads every seat with the same weights, so seat history costs about 90k weights instead of about 4M. It is also what makes the network usable at any table size. The value head (the critic, below) is training-only and is not in policy files.

The runtime refuses a file whose encoder version, action-space version or input sizes differ from the current build. TypeScript and PyTorch agree on the same file to about 1e-8.

## Training (`training/tractor/`)

| File | Role |
|---|---|
| `scenario.ts` | the curriculum |
| `reward.ts` | round rewards |
| `privileged.ts` | the critic's extra inputs |
| `vec-env.ts` | many rounds at once: benchmark seats played in the env, teacher labels, shaping, duplicate pairs, match mode, fault tolerance |
| `env-server.ts` | stdio JSON lines plus raw bytes, built to `training/dist/tractor-env.mjs` |
| `legacy/v2/` | a frozen copy of the v2 encoder, candidates, runtime and agent, so the benchmark plays exactly as it was trained |
| `env_client.py` | the Python side of the bridge |
| `model.py` | network, critic, export and load |
| `train.py` | the trainer |
| `evaluate.py` | evaluations |
| `test_training.py` | pytest |

### What it plays
Episodes are single rounds, started mid-match from sampled levels.

**Table:**
- **85% 4-player Tractor**: 2 decks, final rank A.
- **15% everything else**:
  - 2–12 players, weighted toward 5, 6 and 8
  - Finding Friends on every odd table and on half the even ones
  - other deck counts 15% of the time, NT as the final rank 10%

**The landlord's level:**
- 35% point ranks (5, 10, K)
- 15% A
- 50% the rest
- In Tractor the attackers' level is sampled separately; in Finding Friends each player's level is within ±3 of the landlord's.

**Also:** 15% first rounds; bidding 90% live and 10% in turns; standard rules only.

### Reward
Paid at the round's end, zero-sum between the sides:

    (levels my side gained − the other side's) / 3
    ± 0.25   for keeping / taking the landlord seat
    ± 0.25 × clip((attackers' points − threshold) / threshold, ±1)
    ± 1      for winning the match

There is also **point shaping** (`--points-shaping 0.5`), paid as tricks are won. Each seat's potential is ±0.5 × attackers' points / threshold: plus for the attackers, minus for the landlord's team. The potential is taken back at the round's end, so the shaping sums to zero over a round. It doesn't change what's best; it only moves credit earlier, to the tricks that earned it.

### Opponents: partners are always the same model
In every lineup the learner is a whole team, so it learns partnership play with itself.

| Workers | Opponents |
|---|---|
| 60% | **self-play**: both teams are the current network |
| 40% | **past snapshots**: the newest 8 policies, redrawn every 10 updates, sampling their moves |

`--mix self=6,past=4,bench=2` adds workers that play against the v2 benchmark, inside the env by its own frozen code. They aren't in the default mix: the benchmark runs in plain TypeScript (about 3 ms a move), and one such worker slows every update from about 1.5 s to about 8 s. The benchmark still teaches the distillation phase and is the eval opponent.

### Schedule (`train.py`)
1. **Distillation** (50 updates). Every learner decision is labelled with the **v2 benchmark's** choice, so v3 starts from v2's strength. The benchmark plays for the first 25 updates. For the last 25 (DAgger) the network plays its own moves and keeps learning the benchmark's answer in the positions it gets itself into.
2. **PPO**:
   - per-(env, seat) streams with GAE (γ = 1, λ = 0.95)
   - entropy 0.003
   - learning rate 3e-4 falling linearly to 3e-5 by update 5,000
   - no pull toward the teacher after distillation
3. **A perfect-information critic.** The value head sees every hand, the kitty and the undealt deck (`privileged.ts`; training only), so its estimates, and the advantages, are far less noisy than the seat's own view would allow. The policy never sees them; the shipped AI is unchanged.
4. **Evals** use the exported float16 file, exactly what the game runs:
   - **Every 100 updates**: 1,024 **duplicate deal pairs** of 4-player Tractor against the benchmark. Each deal is played twice with the cards swapped between the teams, so the luck of the deal cancels and a thousand deals resolve differences of a few hundredths of a level (±0.03). It reports gain per round (levels the network's team gained minus the benchmark's), the share of deal pairs it came out ahead or behind on, and landlord keep and take rates.
   - **Every 500 updates and at the end**: 24 duplicate pairs of **full matches** against the benchmark (wins, levels ahead at the end), and 512 rounds of the **other tables** against the benchmark (by mode and table size).
   - Results go to `evals.jsonl`, `progress.md` and tensorboard.

### Settings that worked better when continuing the run
The v3 run plateaued at about +0.10 a round against the benchmark by update ~500 (the score was flat from update 500 to 2000). Continuing from update 2007 with `--rollout 128 --grad-accum 4 --lr 6e-4 --lr-final 1e-4` (four times the decisions behind each gradient step, and a learning rate raised to keep the policy moving as fast as before) turned it into a slow climb of about +0.03 a round over the next 750 updates, to +0.18. Those are not the defaults, because they haven't been tried from a fresh start. Memory: a rollout of 128 holds about 5 GB; the buffer is float16.

### Difficulty levels
The game's Easy / Normal / Hard are the v3 run's updates 100, 500 and 2000, shipped as `src/ai/nn/policies/{easy,normal,hard}.json` and registered in `NN_LEVELS` (`src/ai/nn/levels.ts`); `prepareAi` fetches a level's file before a match with it. Head to head on 4-player Tractor duplicate deals (576 pairs): Normal beats Easy by +0.10 ± 0.04 levels a round, Hard beats Normal by +0.05 ± 0.04, Hard beats Easy by +0.14 ± 0.04. To ship a new set, copy three policy files over those names and update the update numbers expected in `src/ai/nn/nn.test.ts`.

There is no hand-written AI: every AI seat, hint and suggestion is a network.

## Commands

```bash
npm run train:build                                   # bundle the env server (train.py does this itself)
python training/tractor/env_client.py --episodes 200  # random-agent bridge smoke test
python training/tractor/train.py --run v3             # Ctrl+C or a STOP file saves and exits
python training/tractor/train.py --run v3 --resume
python training/tractor/evaluate.py <policy.json> --matches 24 --variety 512   # vs the benchmark (+ matches, other tables)
python training/tractor/evaluate.py <policy.json> --vs <other policy.json>   # two checkpoints head to head
npm run sim -- 4 6 hard,easy                          # the shipped levels, whole matches
python -m pytest training/tractor -q                  # shapes, export, TS/PyTorch parity, bridge, tiny train+resume
npx vitest run training/tractor                       # env tests
tensorboard --logdir training/runs/tractor
```

Use the `train` conda env (`C:/ProgramData/anaconda3/envs/train/python.exe`: PyTorch with CUDA). Runs go to `training/runs/tractor/<run>/`, which git ignores.

When `encode.ts` or `decision.ts` change what they produce, bump `ENCODER_VERSION` / `ACTION_SPACE_VERSION`; old policy files are then refused instead of misread. To keep benchmarking against an older network, freeze a copy of its code the way `legacy/v2/` does.
