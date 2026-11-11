# CLAUDE.md

Context for Claude working in this repo.

## What this is

Tractor (升级 / shengji / Finding Friends) as a static web app for GitHub Pages: hot-seat, vs AI, or online rooms in a Firebase Realtime Database. Any player count (2–12), any deck count (1–12), classic fixed-partner Tractor or Finding Friends. No server code of our own.

The rules are a port of the shengji Rust project (`../shengji`, `mechanics/src/*.rs`, `core/src/game_state/*.rs`, `core/src/settings.rs`). When a rule question comes up, read the Rust first and keep behavior the same; deliberate differences are listed in the README checklist. The architecture follows `../catan` and `../card-collecting` (same room transport, ActionSync and store design).

## Tech stack (locked)

TypeScript, Vite, React 19, Zustand, Firebase Realtime Database + anonymous Auth (lazy-loaded), Vitest. Card art is our own, drawn in code (`src/ui/cardArt.ts` → cached SVG data URLs); no image assets.

## Commands

- `npm run dev`, `npm run build` (tsc + vite), `npm run test:run`, `npx tsc --noEmit -p .`
- `npm run sim -- <players> <matches>` — headless matches of the shipped AI against itself (vite-node `training/simulate.ts`)
- Training the AI: `npm run train:build`, then `C:/ProgramData/anaconda3/envs/train/python.exe training/tractor/train.py --run <name> --init-checkpoint training/runs/tractor/v4/ckpt/latest.pt` (`--resume`); `python -m pytest training/tractor -q`; `npx vitest run training/tractor`. Details in `training/README.md`.
- `FIREBASE_LIVE=1 npx vitest run src/store/networkStore.live.test.ts` — real-database test; then `MSYS_NO_PATHCONV=1 npx firebase-tools database:remove /tractor/rooms/<CODE> --force --project github-games-e7c7d`
- `npm run deploy:rules` — deploy database rules (merge step below). Confirm with the user before deploying: the database is shared.

## Architecture

Layers, by directory; arrows are the only allowed import directions (`ui → store → net/game/ai → core`):

- **`src/core/`** — game-agnostic: `Seat`/`PlayerId`, seeded RNG (mulberry32), seat colors.
- **`src/game/`** — pure rules. No React, DOM, network, `Date.now()` or `Math.random()`.
  - `cards.ts` — card strings: rank+suit (`'2S'`, `'TH'` = ten of hearts, `'AD'`), `'LJ'` small joker, `'BJ'` big joker; levels (`Rank | 'NT'`); points.
  - `trump.ts` — `Trump` (`standard` suit+number / `noTrump` number|null), effective suit (`'T'` = trump), `compareEffective` (off-suit number cards tie) vs `compareCards` (total order, shengji's tie-break), `successor` (tractor ladder).
  - `format.ts` — trick units/shapes, `findPlays` (lead readings), `formatFromCards`, shengji's decomposition ordering (`fullDecompositionOrdering`, `subsequentDecompositionOrdering`), `findFormatMatches` (backtracking matcher with protection filter), `isLegalFollow`, `defeats`, `findThrowBreaker`, descriptions.
  - `trick.ts` — a trick: `playCards` (throw check, forced unit), `computeWinner`, `completeTrick`.
  - `bidding.ts`, `scoring.ts` (step table, `advancePlayer`), `config.ts` (`TractorConfig`, defaults, `validateConfig`, `resolveRules` → decks / kitty / friends / bidding style).
  - `types.ts` — `TractorState`, `RoundState`, `Action`. `engine.ts` — `createInitialState`, `applyAction` (clones, then mutates the copy; throws `IllegalAction`), selectors (`activeSeat`, `validBidsFor`, `canPickUp`, `playProblemFor`, `sortedHand`, …).
- **`src/ai/`** — `observation.ts` (`observe(state, seat)`: only what that seat may see — every AI decision goes through it), `plays.ts` (legal lead / follow candidates from the seat's own hand), `index.ts` (`chooseAIAction` (null until the network is loaded), `adviseAction` (loads it first: hints, suggestions, taking over for a dropped player), `prepareAi` loads it before a match with AI seats), `nn/` (the AI: one trained network, no difficulty levels and no hand-written fallback; `slots.ts` 72 trump-relative card slots, `encode.ts` the v4 observation: a global block plus 12 per-seat blocks, seats placed from both ends (`seatSlot`), `decision.ts` candidates per decision — bid, bury one card at a time, friend call, play — `runtime.ts` the plain-TS network for `tractor/seat-mlp@3` files (shared seat encoder), `agent.ts` `actWithPolicy` + `legalFriendCalls`, `network.ts` loads `policy.json` — update 2000 of the v4 run, float16, fetched on first use; it plays every table), `driver.ts` (`nextAutomatedStep`: AI seats' draws and bids in the live deal, the post-deal bidding window, AI moves, all-AI round advance — used by the UI driver and the simulator; returns a `Pace` the UI maps to delays via `prefsStore.paceMs`), `simulate.ts`.
- **`src/net/`** — room transport copied from card-collecting with catan's shared-database changes: `transport.ts` (interface, log-slot encoding), `firebaseRoom.ts` (real, lazy-loaded, everything under `DB_ROOT = 'tractor'`, day-bucketed stale-room index), `fakeTransport.ts` (tests), `firebaseConfig.ts`, `identity.ts`, `roomLink.ts`, `types.ts`.
- **`src/store/`** — `gameStore` (state + `dispatch` returning an error string or null, `applyLocal`, `restoreState`), `networkStore` (rooms, lobby transactions, presence, chat, which device drives), `actionSync.ts` (write-once log slots, rollback + replay on races; copied unchanged), `prefsStore` (per-device: AI / deal speed, post-deal bidding window, auto-draw, sounds, classic red / black suits (`twoColor`; four-colour is the default), show-all-hands; localStorage). `gameStore` also records the match (`initial` + `actions`, trimmed on rollback via a state→step WeakMap) for the review screen's replay.
- **`src/ui/`** — `App` (home / lobby / game, or the room's screens), `LocalLobby`, `OnlineLobby`, `TractorOptions` (shared options form), `GameHost` (top bar, the driver effect, this device's own auto-draw, online hand-off of a disconnected player's turn to the AI after 30 s / an idle draw after 20 s, sounds, hot-seat pass screen, online banners), `table/` (`Table` with the status bar (trump chip, points meter), team banner and watch-mode `AllHands`, `Felt` (seat plates, plays spinning in and flipping, sweeps to the winner with a points pop, deal flights, event banners; `peek` shows the previous trick over the live one, which stays mounted but hidden; `live` off in the replay), `flight.ts` (the viewer's played cards fly from the hand: measured just before dispatch, picked up when the play mounts), `Hand` (fan, drag-select, slides on re-sort; `Table` passes its grouping and trump marks — printed suits while bidding, trump as one group once the kitty is taken, matching `engine.sortedHand`), `ActionPanel` (deal controls, countdowns, keyboard shortcuts), `SidePanel`, `RoundSummary` (count-up, confetti), `labels`), `review/` (`MatchReview`: stats charts built to the dataviz skill + `Replay`), `sounds.ts` (Web Audio), `cardArt.ts` (the deck: `full` faces for lg/md, `compact` for sm/xs; four-colour or two-colour), `CardView`, `SuitText` (suit symbols in their colours), `Rules`, `lobbyParts`, `ChatPanel`, `OnlineStatus`. Styles: `theme.css` (shell) + `table.css`; every animation has a `prefers-reduced-motion` fallback.

## Invariants

- `applyAction` and `createInitialState` are pure and deterministic; shuffles use `state.rngState`. Online, every device rebuilds the match from the start record (config, seed, seats) and replays the action log, so anything that changes the reducer's output for the same inputs breaks running rooms — bump `PROTOCOL_VERSION` (and the `v` check in the rules) when the room layout changes.
- State is plain JSON (no Maps, classes or undefined-sensitive fields). Payloads to Firebase are JSON strings.
- Actions name their seat by index (`seat`); seats are in play order (clockwise; Tractor partners are same-parity seats).
- Hidden information is a UI concern (every device holds the full state; friends-only honor). AI code must read only an `Observation`.
- `activePlayerId` is null during the live deal (anyone may bid) and between rounds; `drawingSeat(state)` says whose card is next. Each `draw` action deals one card to `round.dealTo` and names that seat: AI seats are drawn by the driving device, a human's by their own device (auto-draw at the deal pace, or the Draw button / D), a disconnected human's by the driver.
- After a live deal an AI landlord-to-be waits `bidWindow` before taking the kitty unless every human has sent `doneBidding`; any new bid clears `round.bidDone`.
- One device drives a match (deals, AI seats, next round): locally this one; online the host while connected, else the first connected seated human (`networkStore.aiDriver`). Driver moves are dispatched with `{ ai: true }` so a lost race is dropped quietly.
- Hot-seat (2+ human seats on one device) always bids in turns (`validateConfig` rejects live bidding there; `biddingStyle: 'auto'` resolves it).
- The AI's inputs and candidates (`src/ai/nn/encode.ts`, `decision.ts`, and the candidate generators in `src/ai/plays.ts`) are a contract with the trained policy file: changing what they produce means bumping `ENCODER_VERSION` / `ACTION_SPACE_VERSION` (old files are then refused) and retraining or carrying the network over. The Python trainer (`training/tractor/`) never re-implements game logic; it only sees what the TS env server (`vec-env.ts`) encodes, and it plays every seat (the env plays none). One format only: the app and the trainer read just the current encoder's files — no frozen copies of old encoders, no second architecture. To change the encoder or candidates without retraining from scratch: only append fields (new `Layout` segments at the end of a block), bump the versions, and carry the network over by copying the input layers' weight columns by segment name (zero for new ones); check with a fixture of real decisions that the old outputs are reproduced, then retire the old file.

## Firebase (shared project)

Project `github-games-e7c7d`, web app "Tractor" (config in `src/net/firebaseConfig.ts`). Other apps (catan, card-collecting, …) use the same database under their own top-level keys. Rules:

- `database.rules.json` defines only `tractor` (root `.read`/`.write` stay false).
- `firebase.json`'s predeploy runs `scripts/merge-db-rules.mjs` (copied from catan): reads the live rules, replaces the `tractor` key, keeps the rest, writes `.firebase/database.rules.deploy.json`, which is what gets deployed. Never deploy a rules file that would drop other apps' keys.
- Layout under `/tractor`: `rooms/{CODE}/{meta, lobby, matches/m{N}/{start, actions/aNNNNNN}, chat, presence}`, plus `roomDays/{day}` and `roomIndex/{day}/{CODE}` for the week-later sweep.

## Conventions

- Keep the README checklist current: tick what's done, add what's discovered.
- Tests never fetch the real policy file: `src/test/setup.ts` (vitest `setupFiles`) installs a tiny random network (seed 4: its matches finish in normal lengths), so whole matches run headless. A test that makes a fresh module graph (`vi.resetModules()`) imports it again. The shipped file itself is checked from disk in `nn.test.ts`.
- Shipping a new network: copy its policy file over `src/ai/nn/policy.json` and update the run and update expected in `nn.test.ts`. Training's default anchor is that file, so the next run measures itself against what ships.
- Rule changes: add or port a test (shengji's Rust tests are the reference — `trick.test.ts` ports trick.rs; `rules.test.ts` ports the bidding / scoring / game-state tests and holds the scenario tests, built with `src/game/testing.ts`'s `scenario()` which sets hands, kitty, levels and landlord directly), and make sure `engine.test.ts`'s full AI matches still finish.
- The one deliberate difference from shengji: when a follow can be read several ways, a later play must beat *every* winning reading of the current winner (`computeWinner` in `trick.ts`); see the README's "Rule decisions".
- Windows + Git Bash: heredocs with non-ASCII text can break; write scripts to files. Use `MSYS_NO_PATHCONV=1` for firebase-tools database paths.
- `package-lock.json` resolves to public npm (`https://registry.npmjs.org/`), like catan's, and CI runs `npm ci`. On this machine the `NPM_CONFIG_REGISTRY` environment variable (DRW Artifactory) overrides the project `.npmrc`: installs fetch through Artifactory and keep the lockfile's npmjs URLs (npm's `replace-registry-host`), but a newly added or upgraded package is written with an Artifactory URL. After changing dependencies, check `grep -c drwholdings package-lock.json` is 0; if not, rewrite `https://artifacts.drwholdings.com/artifactory/api/npm/ext-npm/` to `https://registry.npmjs.org/` (the integrity hashes are the same).
