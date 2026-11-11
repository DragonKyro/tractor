# Tractor · 升级

**🚜 Play it:** https://dragonkyro.github.io/tractor/

Tractor (升级 / shengji, "Finding Friends", "80 points") in the browser: hot-seat on one screen, against AI players, or online with friends through Firebase. Any number of players (2–12), any number of decks, classic fixed-partner Tractor or Finding Friends.

The rules engine is a TypeScript port of the open-source [shengji](https://github.com/rbtying/shengji) Rust implementation (trick formats, tractors, throws, protections, bidding, scoring), with its house-rule options. The app structure (pure game core → room transport → stores → UI) follows the `catan` and `card-collecting` projects.

## Play

```sh
npm install
npm run dev          # http://localhost:5173/tractor/
```

- **Hot-seat**: set two or more seats to human in the lobby. The screen hides hands between turns ("pass the device"), and trump is declared in turns after the deal.
- **Vs AI**: one human seat, the rest AI. Trump is declared live while the cards are dealt.
- **Online**: in the lobby, "Host online" opens a room; friends join with the 4-letter code or the invite link (`?room=CODE`). The host's device deals and runs the AI seats; if it drops, another player's device takes over.

## Commands

| | |
|---|---|
| `npm run dev` | dev server |
| `npm run build` | typecheck + production build to `dist/` |
| `npm run test:run` | all tests (rules ports from shengji, engine, full AI matches, online over an in-memory server) |
| `FIREBASE_LIVE=1 npx vitest run src/store/networkStore.live.test.ts` | online test against the real database (prints the room code to clean up) |
| `npm run sim -- 4 6` | headless matches of the shipped AI against itself, with stats (players, matches) |
| `npm run train:build`, `python training/tractor/train.py --run <name>` | build the env server and train the AI (see [training/README.md](training/README.md)) |
| `npm run deploy:rules` | deploy the database rules (merged into the shared project's live rules, see below) |

## Online setup

Firebase project `github-games-e7c7d` (web app "Tractor"), **shared with other apps** (catan, card-collecting, …). Everything this app stores lives under `/tractor` in the Realtime Database, and `database.rules.json` defines only the `tractor` key. Because a database has a single rules set, `firebase deploy --only database` runs `scripts/merge-db-rules.mjs` first: it reads the live rules, swaps in the `tractor` key and keeps every other app's key untouched. Rooms are cleaned up about a week after they're made.

## Rule decisions

Everything follows shengji, with one deliberate exception. Sometimes a play answering a throw can be grouped into the lead's shape in more than one way. For example, trumping "pair + two singles" with 7♠7♠9♠9♠ reads as "pair of 9s + 7, 7" or "pair of 7s + 9, 9". A later player beating one reading but not the other would win or lose depending on which grouping shengji's search happens to find first. Here, ambiguity goes to the earlier player, the same way exact ties do: to take the trick, a play must beat every way the current winner's cards can be read.

## Credits

- Rules reference and test scenarios: [shengji](https://github.com/rbtying/shengji) by Robert Ying (MIT).
- Card art: drawn for this project in code (`src/ui/cardArt.ts`): four-colour suits, cartoon court cards, jester jokers and a tractor back.

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
- [x] One AI: update 2000 of the v4 training run (`src/ai/nn/policy.json`), deciding from a seat's own view only, a 3.7 MB float16 file fetched when a match with AI seats starts (or a hint is asked for). Against the earlier best network (v3 update 2000) on 4-player Tractor duplicate deals: +0.13 ± 0.03 levels a round, 77% of full matches won, +0.06 a round on the other tables. Difficulty levels and the hand-written heuristic have been removed.
- [x] Options to hide the defenders' captured points and the record of played cards (the table keeps count; everything shows at the round's end)
- [x] "End round early": offered only when public information proves nothing left can change the result — unseen point cards, the largest possible kitty multiplier and future throw penalties all included (`canEndRoundEarly`); checked against playing the round out in AI games (`earlyEnd.test.ts`)
- [x] Live deal one card at a time, each player drawing in turn: auto-draw or click to draw (D / Space); deal speed and the post-deal "time to overcall" (3 / 6 / 12 s, with a countdown) are set right in the deal panel; "No more bids" lets an AI landlord take the kitty at once
- [x] Animations: cards are dealt from the deck to each seat; plays spin in from their seat and flip face up, the viewer's own fly up from the hand; a won trick flashes, sweeps to the winner and its points pop up there; the hand fans, lifts what's selected and slides cards into place when it re-sorts; bids pop with a ring, the active seat has a spinning ring, friends found and failed throws get a banner; the round summary counts the points up, with confetti when your side wins. All of it respects reduced motion. Synthesized sounds (toggle in Settings)
- [x] Hand order: while bidding, every card stays in its printed suit (so suit lengths show for overcalling; the leading bid's suit last, then the jokers) with the trump numbers and jokers marked; once the kitty is taken, the trump suit, the trump numbers and the jokers sort together as one suit
- [x] Trump is one suit when following: any trump-number card or joker follows a trump-suit lead and vice versa, a trump number never counts as its printed suit; within trump, trump suit < off-suit numbers (tied: the first played wins) < the trump-suit number < small joker < big joker (`rules.test.ts`, "trump is one suit")
- [x] Last trick: the status bar's "Last trick" button (tap to toggle, hold to peek) or L shows the previous trick on the felt, in place of the live one, which carries on underneath; playing a card closes it. Shown even with "hide played cards", as shengji does (that option only hides the log)
- [x] Drag across cards to select several; keyboard: Enter plays / confirms, Esc clears, H hints
- [x] Team view: a banner says who you're with (and, in Finding Friends, which friends are still hidden and whether a card you hold could make you one); seat badges ringed by side
- [x] Watch mode for all-AI tables with every hand face up
- [x] Match review: stats (levels by round, attackers' points against the threshold, round table — each chart with a table view) and a move-by-move replay of finished rounds
- [x] Phones: landscape layout, side panel as a drawer, wider card spacing and taller buttons on touch screens
- [x] Online: a disconnected player's turn goes to the AI after 30 s (their draws at once); a connected player who doesn't draw for 20 s gets the card drawn for them
- [x] Hot-seat with pass-the-device screens
- [x] Online rooms: lobby, seats (reorder for partners), options, chat, presence, spectators, rejoin by link, driver hand-off, rollback on races
- [x] Firebase: `/tractor` namespace, rules deployed through the merge script, live round-trip test
- [x] UI: home, lobbies, options (with house rules), felt table, hand with selection, bid / exchange / friend / play panels, hints, log, round summary, rules page, AI speed
- [x] Our own deck drawn in code: four-colour suits by default (♠ ink, ♥ red, ♦ blue, ♣ green; classic red / black in Settings), big stroked indices, a compact face with a giant index for the small sizes, cartoon kings / queens / jacks, big (colour) and small (grey) jester jokers, a crimson tractor back; trump cards in the hand get a gold ring and top edge, and suit symbols in the log, status bar and banners use the same colours

- [x] Learned AI (`src/ai/nn`, `training/tractor`, see [training/README.md](training/README.md)):
  - an encoder of a seat's view with a block per seat for every table up to 12 players: how each seat played each card (led, followed, ruffed, discarded, first discard, when it led and discarded, this and the last trick, cards shown), levels, sides, voids, bids, points fed to each side; per suit: unseen cards and points, boss status, cards above my top; trump control, the endgame and the kitty multiplier
  - a wide action space: every discard and every follow the rules leave open, boss throws and their variants, with features for what each does to the hand and the side (kill and ruff risk, control kept, drawing trumps, the final trick)
  - a shared per-seat encoder: 1.40M parameters, a 3.7 MB float16 policy file; plain-TypeScript runtime, TS/PyTorch parity tested
  - the curriculum: 85% 4-player Tractor, 15% every other table
  - partners are always the same model; opponents are self-play, the run's past snapshots and the anchor (by default the shipped AI)
  - PPO with a perfect-information critic and belief heads that predict the hidden hands (both training only), point shaping, averaged weights; a fresh network first distills the anchor
  - duplicate-deal evals against the anchor, with per-skill measures (throws, trump leads, ruffs, points fed and given, the last trick, the kitty)
  - one format: the app and the trainer read only the current encoder's policy files. The v2 benchmark and its frozen code, and the v3 → v4 upgrade tool, have been removed.
- [x] Training history: v2 (2,486 updates, 4/5/6-player tables) beat the old heuristic in about 100% of matches; v3 (distilled from v2, stopped at update 2760) was +0.18 a round over v2; v4 (v3's update 2000 with the v4 inputs appended at zero weight, 2,000 more updates) is +0.28 a round over v2 and +0.13 over v3. Only v4's checkpoint and final policy are kept (`training/runs/tractor/v4`).
- [x] `applyAction` copies the state with a hand-written JSON deep copy instead of `structuredClone`. The states are identical (checked on seeded full matches) and headless play is 3.5× faster.

### To do

AI
- [ ] The v4 network's weak spots against v3 (same cards): it wins the last trick 47% of the time (v3 53%) and concedes more kitty points as a defender (2.0 a round vs 1.7); 4-player Finding Friends is −0.10 a round
- [ ] Search at play time: sample hidden hands (the belief heads can propose them) and simulate ahead for the key decisions — the endgame and the kitty — (strongest, but costly in the browser)
- [ ] Signaling: check whether partners learn conventions. Measure a network pair against a network paired with an older one, and how readable a seat's hidden cards are from its plays. Shuffle the side-suit order during training if arbitrary suit codes appear
- [ ] Train on more tables: 4-player Tractor is 85% of the curriculum; the other tables get 15% between them

UI / UX
- [ ] Try the offline hand-off and the online draw flow with real people on several devices (covered by unit / fake-server tests and typechecks, not yet by hand)
