# Tractor · 升级

Tractor (升级 / shengji, "Finding Friends", "80 points") in the browser: hot-seat on one screen, against AI players, or online with friends through Firebase. Any number of players (2–12), any number of decks, classic fixed-partner Tractor or Finding Friends.

The rules engine is a TypeScript port of the open-source [shengji](https://github.com/rbtying/shengji) Rust implementation (trick formats, tractors, throws, protections, bidding, scoring), with its house-rule options. The app structure (pure game core → room transport → stores → UI) follows the `catan` and `card-collecting` projects.

## Play

```sh
npm install
npm run dev          # http://localhost:5173/tractor/
```

- **Hot-seat**: set two or more seats to human in the lobby. The screen hides hands between turns ("pass the device"), and trump is declared in turns after the deal.
- **Vs AI**: one human seat, the rest AI (Easy, Normal or Hard). Trump is declared live while the cards are dealt.
- **Online**: in the lobby, "Host online" opens a room; friends join with the 4-letter code or the invite link (`?room=CODE`). The host's device deals and runs the AI seats; if it drops, another player's device takes over.

## Commands

| | |
|---|---|
| `npm run dev` | dev server |
| `npm run build` | typecheck + production build to `dist/` |
| `npm run test:run` | all tests (rules ports from shengji, engine, full AI matches, online over an in-memory server) |
| `FIREBASE_LIVE=1 npx vitest run src/store/networkStore.live.test.ts` | online test against the real database (prints the room code to clean up) |
| `npm run sim -- 4 6 hard,easy` | headless matches between the shipped AI levels, with stats (players, matches, seat levels) |
| `npm run train:build`, `python training/tractor/train.py --run main` | build the env server and train the learned AI (see [training/README.md](training/README.md)) |
| `npm run deploy:rules` | deploy the database rules (merged into the shared project's live rules, see below) |

## Online setup

Firebase project `github-games-e7c7d` (web app "Tractor"), **shared with other apps** (catan, card-collecting, …). Everything this app stores lives under `/tractor` in the Realtime Database, and `database.rules.json` defines only the `tractor` key. Because a database has a single rules set, `firebase deploy --only database` runs `scripts/merge-db-rules.mjs` first: it reads the live rules, swaps in the `tractor` key and keeps every other app's key untouched. Rooms are cleaned up about a week after they're made.

## Rule decisions

Everything follows shengji, with one deliberate exception. Sometimes a play answering a throw can be grouped into the lead's shape in more than one way. For example, trumping "pair + two singles" with 7♠7♠9♠9♠ reads as "pair of 9s + 7, 7" or "pair of 7s + 9, 9". A later player beating one reading but not the other would win or lose depending on which grouping shengji's search happens to find first. Here, ambiguity goes to the earlier player, the same way exact ties do: to take the trick, a play must beat every way the current winner's cards can be read.

## Credits

- Rules reference and test scenarios: [shengji](https://github.com/rbtying/shengji) by Robert Ying (MIT).
- Card faces: [Adrian Kennard's SVG playing cards](https://www.me.uk/cards/) (CC0), as packaged by shengji (`public/cards`, four-color set in `public/cards-4c`).

## Checklist

### Done

- [x] Card order under any trump (suit / number / no-trump), tractor successors through the trump number and jokers
- [x] Lead decomposition (singles, tuples, tractors, throws) and ambiguous-lead readings
- [x] Follow legality with shengji's decomposition order and all five draw policies (incl. "longer tuples protected")
- [x] Throws: checked against every other hand; a failed throw plays only the beatable unit (optional 10-point penalty)
- [x] Trick winners with all three throw-evaluation policies
- [x] Ported shengji's tests: 43 trick tests, the bidding and scoring tables, and its game-state scenarios (bid sequence, legal friend calls, must-defend level sequences, a 17-trick six-player Finding Friends game, the short-handed bonus)
- [x] Scenario tests for the fiddly rules (`src/game/rules.test.ts`): overcalling to become landlord, point cards, the kitty multiplier, thresholds per deck count, "the 2nd A♠" friend calls, the landlord playing a called card, friends' earlier points, failed throws and the penalty in both directions, take-overs, must-defend and winning the game
- [x] Bidding: live while dealing or in turns after the deal; bid / joker / reinforcement policies; first-landlord choice; redeal; kitty reveal when nobody bids
- [x] Kitty exchange; Finding Friends calls ("the 2nd A♠"), reveal on play, double-join policy, friend-call restrictions
- [x] Scoring with shengji's step table, kitty multiplier (×2·unit or 2^unit), short-handed bonus level, must-defend rules, next landlord, game end past the final rank
- [x] 2–12 players, 1–12 decks, auto or explicit kitty size (removes low cards to deal evenly)
- [x] Three AI levels, **Easy / Normal / Hard**: checkpoints at updates 100, 500 and 2000 of one training run (v3), deciding from a seat's own view only. Each is a 3.6 MB float16 file fetched when a match with that level starts. Head to head on 4-player Tractor duplicate deals (576 pairs each): Normal beats Easy by +0.10 ± 0.04 levels a round, Hard beats Normal by +0.05 ± 0.04, Hard beats Easy by +0.14 ± 0.04. The hand-written heuristic AI that came first has been removed.
- [x] Options to hide the defenders' captured points and the record of played cards (the table keeps count; everything shows at the round's end)
- [x] "End round early": offered only when public information proves nothing left can change the result — unseen point cards, the largest possible kitty multiplier and future throw penalties all included (`canEndRoundEarly`); checked against playing the round out in AI games (`earlyEnd.test.ts`)
- [x] Live deal one card at a time, each player drawing in turn: auto-draw or click to draw (D / Space); deal speed and the post-deal "time to overcall" (3 / 6 / 12 s, with a countdown) are set right in the deal panel; "No more bids" lets an AI landlord take the kitty at once
- [x] Animations (cards fly in from their seat, tricks sweep to the winner, drawn cards pop in, the active seat glows; reduced-motion respected) and synthesized sounds (toggle in Settings)
- [x] Drag across cards to select several; keyboard: Enter plays / confirms, Esc clears, H hints
- [x] Team view: a banner says who you're with (and, in Finding Friends, which friends are still hidden and whether a card you hold could make you one); seat badges ringed by side
- [x] Watch mode for all-AI tables with every hand face up
- [x] Match review: stats (levels by round, attackers' points against the threshold, round table — each chart with a table view) and a move-by-move replay of finished rounds
- [x] Phones: landscape layout, side panel as a drawer, wider card spacing and taller buttons on touch screens
- [x] Online: a disconnected player's turn goes to the AI after 30 s (their draws at once); a connected player who doesn't draw for 20 s gets the card drawn for them
- [x] Hot-seat with pass-the-device screens
- [x] Online rooms: lobby, seats (reorder for partners), options, chat, presence, spectators, rejoin by link, driver hand-off, rollback on races
- [x] Firebase: `/tractor` namespace, rules deployed through the merge script, live round-trip test
- [x] UI: home, lobbies, options (with house rules), felt table, hand with selection, bid / exchange / friend / play panels, hints, log, round summary, rules page, four-color cards, AI speed

- [x] Learned AI, v3 (`src/ai/nn`, `training/tractor`, see [training/README.md](training/README.md)):
  - an encoder of a seat's view with a block per seat for every table up to 12 players: how each seat played each card (led, followed, ruffed, discarded, first discard, this and the last trick, cards shown), levels, sides, voids, bids
  - a wider action space: every discard and every follow the rules leave open, with features for what each does to the hand and the side
  - a shared per-seat encoder: 1.36M parameters, a 3.6 MB float16 policy file
  - plain-TypeScript runtime, TS/PyTorch parity tested
  - the curriculum: 85% 4-player Tractor, 15% every other table
  - partners are always the same model; opponents are self-play, past snapshots and the v2 benchmark
  - distillation from the v2 network, then PPO with a perfect-information critic (training only) and point shaping
  - duplicate-deal evals against the v2 benchmark (`training/benchmarks/v2-update1500.json`, run by a frozen copy of the v2 code)
  - network AI levels play every table in the game
- [x] Learned AI v2 (superseded): one run of 2,486 updates on 4/5/6-player tables. Its update 1500 beat the old hand-written heuristic in about 100% of 4- and 6-player Tractor matches, about 8 levels ahead, and is kept as the benchmark.
- [x] v3 run: stopped at update 2760 (+0.18 ± 0.03 a round against the benchmark, 75% of full matches won). The policies after update 2000 were deleted: head to head, update 2760 vs 2000 is +0.01 ± 0.04, indistinguishable. The weights at update 2760 remain in the run's `ckpt/latest.pt`.
- [x] `applyAction` copies the state with a hand-written JSON deep copy instead of `structuredClone`. The states are identical (checked on seeded full matches) and headless play is 3.5× faster.

### To do

AI
- [ ] Make the levels further apart (Hard is only +0.05 a round over Normal): train further, using the ideas below
- [ ] Wider opponent pool: draw half the past-snapshot opponents from the whole run history, not just the newest 8 (self-play plateaued around update 500-1000)
- [ ] Signaling: check whether v3 partners learn conventions. Measure a network pair against a network paired with the benchmark, and how readable a seat's hidden cards are from its plays. Shuffle the side-suit order during training if arbitrary suit codes appear
- [ ] Belief head: an auxiliary loss predicting each opponent's hand from the public record, to sharpen what the network infers about hidden cards
- [ ] Search at play time: sample hidden hands and simulate ahead for the key decisions (strongest, but costly in the browser)
- [ ] Faster benchmark seats: the v2 network runs in plain TypeScript inside the env (about 3 ms a move). Batch it on the GPU if training against it more than for distillation and evals
- [ ] Train on more tables: 4-player Tractor is 85% of the curriculum; the other tables get 15% between them

UI / UX
- [ ] Try the offline hand-off and the online draw flow with real people on several devices (covered by unit / fake-server tests and typechecks, not yet by hand)
- [ ] Deal animation from the deck to each seat (cards currently pop into your hand)
