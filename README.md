# Tractor · 升级

Tractor (升级 / shengji, "Finding Friends", "80 points") in the browser: hot-seat on one screen, against AI players, or online with friends through Firebase. Any number of players (2–12), any number of decks, classic fixed-partner Tractor or Finding Friends.

The rules engine is a TypeScript port of the open-source [shengji](https://github.com/rbtying/shengji) Rust implementation (trick formats, tractors, throws, protections, bidding, scoring), with its house-rule options. The app structure (pure game core → room transport → stores → UI) follows the `catan` and `card-collecting` projects.

## Play

```sh
npm install
npm run dev          # http://localhost:5173/tractor/
```

- **Hot-seat**: set two or more seats to human in the lobby. The screen hides hands between turns ("pass the device"), and trump is declared in turns after the deal.
- **Vs AI**: one human seat, the rest AI (Easy or Normal). Trump is declared live while the cards are dealt.
- **Online**: in the lobby, "Host online" opens a room; friends join with the 4-letter code or the invite link (`?room=CODE`). The host's device deals and runs the AI seats; if it drops, another player's device takes over.

## Commands

| | |
|---|---|
| `npm run dev` | dev server |
| `npm run build` | typecheck + production build to `dist/` |
| `npm run test:run` | all tests (rules ports from shengji, engine, full AI matches, online over an in-memory server) |
| `FIREBASE_LIVE=1 npx vitest run src/store/networkStore.live.test.ts` | online test against the real database (prints the room code to clean up) |
| `npm run sim -- 4 20 normal,easy` | headless AI-vs-AI matches with stats (players, matches, seat levels) |
| `npm run deploy:rules` | deploy the database rules (merged into the shared project's live rules, see below) |

## Online setup

Firebase project `github-games-e7c7d` (web app "Tractor"), **shared with other apps** (catan, card-collecting, …). Everything this app stores lives under `/tractor` in the Realtime Database, and `database.rules.json` defines only the `tractor` key. Because a database has a single rules set, `firebase deploy --only database` runs `scripts/merge-db-rules.mjs` first: it reads the live rules, swaps in the `tractor` key and keeps every other app's key untouched. Rooms are cleaned up about a week after they're made.

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
- [x] Ported 43 shengji trick tests (all pass)
- [x] Bidding: live while dealing or in turns after the deal; bid / joker / reinforcement policies; first-landlord choice; redeal; kitty reveal when nobody bids
- [x] Kitty exchange; Finding Friends calls ("the 2nd A♠"), reveal on play, double-join policy, friend-call restrictions
- [x] Scoring with shengji's step table, kitty multiplier (×2·unit or 2^unit), short-handed bonus level, must-defend rules, next landlord, game end past the final rank
- [x] 2–12 players, 1–12 decks, auto or explicit kitty size (removes low cards to deal evenly)
- [x] Heuristic AI (Easy / Normal) deciding from a seat's own view only; Normal beats Easy 30/30 in 4-player matches
- [x] Hot-seat with pass-the-device screens
- [x] Online rooms: lobby, seats (reorder for partners), options, chat, presence, spectators, rejoin by link, driver hand-off, rollback on races
- [x] Firebase: `/tractor` namespace, rules deployed through the merge script, live round-trip test
- [x] UI: home, lobbies, options (with house rules), felt table, hand with selection, bid / exchange / friend / play panels, hints, log, round summary, rules page, four-color cards, AI speed

### To do

Rules
- [ ] Kitty theft (over-bidding during the exchange and re-exchanging; `Bid.epoch` is already carried)
- [ ] Take-backs (bids and plays) for casual tables
- [ ] Bombs and rainbows (shengji's compound formats)
- [ ] Jack variation ("back to 2" when the last trick is won with a jack)
- [ ] Hide landlord points / hide played cards options
- [ ] Short decks (excluding low ranks or jokers per deck)
- [ ] End a round early once the outcome can't change
- [ ] Custom scoring tables (per-deck step adjustments, dead zone, number of steps)
- [ ] Port shengji's bidding.rs and scoring.rs tests too (only trick.rs is ported so far)
- [ ] Known edge differences from shengji: when a follow can be read several ways, the first winning reading is kept (shengji keeps the first in its search order); leads tie-break the same way but can differ on exotic multi-deck shapes

AI
- [ ] Learned AI at several difficulties (see `training/README.md`): observation encoder, env server, PPO trainer, exported policies loaded by `prepareAi`
- [ ] Better heuristics: void / trump tracking per player, smarter friend calls and partner inference in Finding Friends, deliberate throws, end-game kitty play

UI / UX
- [ ] Animations (deal, card play, trick sweep) and sounds
- [ ] Drag-to-select and keyboard shortcuts in the hand
- [ ] Phone landscape layout, larger tap targets for big hands
- [ ] Post-game match stats and replay (as in catan / card-collecting)
- [ ] "Show all hands" for all-AI tables (watch mode)
- [ ] Clear "who is on my team" view in Finding Friends as friends reveal
- [ ] Hand off an offline human's seat to the AI after a timeout (online)

Project
- [ ] Turn on GitHub Pages for the repo (`.github/workflows/deploy.yml` builds and deploys on push to main)
