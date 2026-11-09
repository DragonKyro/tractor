# training/

Offline tooling for the AI players. Nothing here ships to GitHub Pages.

- `simulate.ts`: headless matches of the shipped AI against itself through the real driver and reducer (`npm run sim -- 4 6`).
- `tractor/`: the AI's training pipeline. A TypeScript env server runs the game; a PyTorch trainer learns from it.

The game has one AI: **update 2000 of the v4 run**, `src/ai/nn/policy.json`, loaded by `src/ai/nn/network.ts` (`prepareAi` fetches it before a match with AI seats; hints load it on first use). It is also the default opponent and yardstick for training (`--anchor`). There is no hand-written AI and no second network format: the app and the trainer read only the current encoder's policy files.

## The network (shipped code: `src/ai/nn/`, encoder v4, action space v3)

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

The vector is a **global block** (725 numbers) followed by **12 seat blocks** (788 each), 10,181 in all. v4 is v3 with fields appended to each block (the "v4" rows below); the v4 run started from v3's network with the new inputs at zero weight.

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
| v4, each seat: 2 card planes + 7 features | <ul><li>**when** it led and when it discarded each card (the order of its signals)</li><li>trumps played; tricks it trumped in on and discarded in</li><li>points it fed to its own side vs gave to the other</li><li>whether it may still hold trump, and at most how many</li></ul> |
| v4, global: 46 features | <ul><li>**per suit group**: my length, unseen cards and points, whether my top single and my top pair are the boss, how many unseen cards beat my top card, how many opponents and whether a partner are known void</li><li>**trump control**: my share of the trumps still out, opponents who may still hold trump, a partner void in trump</li><li>**endgame**: tricks left, the final trick, the largest kitty multiplier my hand can still make</li></ul> |

### Outputs (`decision.ts`): one score per candidate move

Plays are combinatorial, so the network scores a list of up to **64 legal candidates**. Each has **123 features**:
- its cards in the 72 slots
- shape, points, all trump
- for a play: is it a throw, does it win now, is it unbeatable, is it a ruff, what share is discarded
- what it does to my hand: the suits it leaves me void in, pairs broken, its highest and lowest rank, that suit's length
- where the points go if the trick ended now: to my side or the other
- for a lead: whether my partner or an opponent is known to be void in that suit
- for a bid: count, joker bid, how much of my hand would be trump
- for a friend call: which copy, and how many I hold
- v4: **kill risk** (unseen cards that could beat the play, for a throw: its weakest unit), **ruff risk** (opponents still to play known void in the suit but not in trump), trumps and boss cards I keep after it, whether it **draws trumps** from opponents who may hold them, whether it is the **final trick** and the kitty multiplier it would carry, and how many boss units of the suit it holds back

The decisions:
- **bid**: in a live deal, a seat is asked whenever its legal bids change. "No bid" means "not now".
- **bury**: the kitty, one card at a time.
- **call**: friend calls, one at a time.
- **play**: every lead (singles, tuples, tractors, unbeatable throws, and v4's throw variants: the throw without its smallest boss unit, with only its two largest, or with the suit's highest other card added when at most two unseen cards beat it), and every follow the rules allow:
  - the legal matchings of the led shape
  - **every distinct card when one card is free to choose**: any discard when void, any card of the suit when following a single
  - a cheapest fill taken from each suit first
  - a fill that keeps pairs

Single-candidate steps play themselves. In the app, a network seat that declined to bid isn't asked again until its options change (`BidMemo` in `agent.ts`), as in training.

### Architecture (`runtime.ts`, format `tractor/seat-mlp@3`)

```
seat encoder   each present seat block 788 → 128 → 64 (ReLU), shared by all 12 slots; absent seats 0
trunk          [global 725, 12 × 64] = 1493 → 512 → 512 → 256 (ReLU)              = h
cand tower     each candidate 123 → 128 → 128 (ReLU)                              = c_k
scorer         logit_k = w · ReLU(J_s·h + J_c·c_k + b) + b'    (joint width 256)
training only:
critic         V = 256 → [h, hidden hands, kitty, deck] → 256 → 1
belief         for each other seat: [h, e_s] → 128 → 72 (its hand); h → 128 → 72 (the kitty)
```

The actor is **1,399,361 parameters**; a float16 policy file is 3.7 MB.

**The belief heads** predict every hidden hand, and the kitty, from what the seat has seen, trained against the true hands (`--belief-coef`, 0.3). To do that, the shared trunk has to count cards, read voids and read a partner's discards: the skills that separate strong players. The policy reads the same trunk. `belief_holder_acc` in the log is how often the seat it thinks most likely holds a hidden card does hold it (about 1 in 3 is chance with three other seats).

The shared seat encoder reads every seat with the same weights, so seat history costs about 100k weights instead of about 5M. It is also what makes the network usable at any table size. The critic and the belief heads are training-only and are not in policy files.

The runtime refuses a file whose encoder version, action-space version or input sizes differ from the current build. TypeScript and PyTorch agree on the same file to about 1e-8.

## Training (`training/tractor/`)

| File | Role |
|---|---|
| `scenario.ts` | the curriculum |
| `reward.ts` | round rewards |
| `privileged.ts` | the critic's extra inputs |
| `skills.ts` | per-seat skill measures of a finished round (evals) |
| `vec-env.ts` | many rounds at once: every decision, learner's and opponent's, goes to the trainer; shaping, duplicate pairs, match mode, fault tolerance |
| `env-server.ts` | stdio JSON lines plus raw bytes, built to `training/dist/tractor-env.mjs` |
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
| 60% (50% with an anchor) | **self-play**: both teams are the current network |
| 40% (33%) | **past snapshots**: redrawn every 10 updates; with `--pool-span all` (the default) half the draws come from the whole run's history, half from the newest 8, so it doesn't only meet near-copies of itself |
| (17%) | **the anchor** (`--anchor`, by default the shipped AI): a fixed network, so it keeps beating a strong fixed opponent rather than drifting with itself. It is copied into the run (`anchor.json`) when the run starts, so the run's evals stay comparable after the shipped file changes |

Every seat is played by the trainer, batched on the GPU; the env only deals, encodes and applies moves. `--anchor none` trains without an anchor (no distillation, no evals).

### Schedule (`train.py`)
1. **Distillation** (50 updates, only for a network that doesn't start from weights, e.g. a new width). Every learner decision is labelled with the **anchor's** choice (the trainer runs the anchor on the same inputs), so a fresh network starts from its strength. The anchor's moves are played for the first 25 updates. For the last 25 (DAgger) the network plays its own moves and keeps learning the anchor's answer in the positions it gets itself into.
2. **PPO**:
   - per-(env, seat) streams with GAE (γ = 1, λ = 0.95)
   - entropy 0.003
   - learning rate 3e-4 falling linearly to 3e-5 by update 5,000
   - no pull toward the anchor after distillation
3. **A perfect-information critic.** The value head sees every hand, the kitty and the undealt deck (`privileged.ts`; training only), so its estimates, and the advantages, are far less noisy than the seat's own view would allow. The policy never sees them; the shipped AI is unchanged.
4. **Evals** use the exported float16 file, exactly what the game runs:
   - **Every 100 updates**: 1,024 **duplicate deal pairs** of 4-player Tractor against the anchor. Each deal is played twice with the cards swapped between the teams, so the luck of the deal cancels and a thousand deals resolve differences of a few hundredths of a level (±0.03). It reports gain per round (levels the network's team gained minus the anchor's), the share of deal pairs it came out ahead or behind on, landlord keep and take rates, and the skill measures below.
   - **Every 500 updates and at the end**: 24 duplicate pairs of **full matches** against the anchor (wins, levels ahead at the end), and 512 rounds of the **other tables** against the anchor (by mode and table size).
   - Results go to `evals.jsonl`, `progress.md` and tensorboard.

### Starting from weights
`--init-checkpoint <run>/ckpt/latest.pt` starts from another run's actor, critic and belief heads (fresh optimizer, updates from 0); `--init-from <policy.json>` from a policy file's actor, with a fresh critic. Either skips distillation. `--critic-warmup N` first trains only the critic and belief heads for N updates with the policy frozen, so PPO's first advantages come from a critic that fits. With a fresh optimizer, Adam's first steps are full-size in every direction (the first PPO update of a test run moved the policy by KL 0.06, about ten times the usual), so `--lr-warmup 50` ramps the learning rate up from 10% and `--target-kl 0.02` ends an update's epochs early if it still moves too far. `--ema 0.995` exports (and evaluates) a moving average of the weights, which smooths out update-to-update noise.

**Changing the inputs or candidates.** Bump `ENCODER_VERSION` / `ACTION_SPACE_VERSION` (old files are then refused instead of misread). To keep a trained network rather than start over, only append fields (new `Layout` segments at the end of a block) and carry the network over: copy each old weight column of the three input layers (seat encoder, the trunk's global part, candidate tower) to the same-named new input and give the new inputs zero weight. The carried-over network then plays exactly as before (check it on a fixture of real decisions), and training learns how much to use the new inputs. That is how v4 started from v3.

### Measuring progress
- **The main score**: duplicate deals of 4-player Tractor against the anchor (`--anchor`, by default the shipped AI). A run that starts from the shipped network's checkpoint starts at exactly 0, so every gain is improvement over it. 1,024 deal pairs resolve about ±0.03 levels a round; +0.10 a round is about 72% of full matches won.
- **Skills** on the same deals, the network's seats next to the anchor's (`skills.ts`, `progress.md`): throws tried and the share that failed (card counting before a throw), trump leads (tempo), side-suit tricks won by trumping in (voids, often made by the kitty), points fed to partners vs given to opponents (partner play), the last trick won, and kitty points conceded as defenders. These show which skills the gain comes from.
- **Belief accuracy** (training log, tensorboard): how well the trunk tracks the hidden cards.
- **Whole matches and other tables**: every 500 updates, full duplicate matches and the other tables against the anchor.

### Settings that worked better when continuing a run
The v3 run plateaued at about +0.10 a round against the v2 network by update ~500. Continuing from update 2007 with `--rollout 128 --grad-accum 4 --lr 6e-4 --lr-final 1e-4` (four times the decisions behind each gradient step, and a learning rate raised to keep the policy moving as fast as before) turned it into a slow climb of about +0.03 a round over the next 750 updates. The v4 run used the same settings from v3's update 2000 and reached +0.10 a round over it by update 1000, then +0.13 at update 2000 as the learning rate fell. Those are not the defaults, because they haven't been tried from a fresh start. Memory: a rollout of 128 holds about 5 GB; the buffer is float16.

### The v4 run (the shipped AI)
2,000 updates from v3's update 2000 (`--critic-warmup 20 --lr-warmup 50 --target-kl 0.02`, the settings above). At update 2000, against v3's update 2000 on the same cards: +0.133 ± 0.03 levels a round, ahead on 38% of deal pairs and behind on 17%, 77% of full matches won (3.8 levels ahead at the end), +0.06 a round on the other tables (−0.10 at 4-player Finding Friends). Its throws fail 3% of the time (v3: 13%) and it gives opponents fewer points (15.7 a round vs 18.4); it wins the last trick less often (47% vs 53%) and concedes a little more of the kitty as a defender. `training/runs/tractor/v4/` keeps its checkpoint (`ckpt/latest.pt`, the starting point for the next run), its final policy, logs and `progress.md`.

To ship a new network, copy its policy file over `src/ai/nn/policy.json` and update the run and update expected in `src/ai/nn/nn.test.ts`.

## Commands

```bash
npm run train:build                                   # bundle the env server (train.py does this itself)
python training/tractor/env_client.py --episodes 200  # random-agent bridge smoke test
python training/tractor/train.py --run v5 --init-checkpoint training/runs/tractor/v4/ckpt/latest.pt \
    --critic-warmup 20 --lr-warmup 50 --target-kl 0.02 \
    --rollout 128 --grad-accum 4 --lr 6e-4 --lr-final 1e-4 --updates 2000      # Ctrl+C or a STOP file saves and exits
python training/tractor/train.py --run v5 --resume ...   # the same flags
python training/tractor/evaluate.py <policy.json> --matches 24 --variety 512   # vs the shipped AI (+ matches, other tables)
python training/tractor/evaluate.py <policy.json> --vs <other policy.json>   # two checkpoints head to head
npm run sim -- 4 6                                    # the shipped AI, whole matches through the app's driver
python -m pytest training/tractor -q                  # shapes, export, TS/PyTorch parity, bridge, tiny train+resume
npx vitest run training/tractor                       # env tests
tensorboard --logdir training/runs/tractor
```

Use the `train` conda env (`C:/ProgramData/anaconda3/envs/train/python.exe`: PyTorch with CUDA). Runs go to `training/runs/tractor/<run>/`, which git ignores.
