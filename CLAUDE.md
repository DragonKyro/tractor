# CLAUDE.md

Context for Claude working in this repo.

## What this is

Tractor (升级 / shengji / Finding Friends) as a static web app for GitHub Pages: hot-seat, vs AI, or online rooms in a Firebase Realtime Database. Any player count (2–12), any deck count (1–12), classic fixed-partner Tractor or Finding Friends. No server code of our own.

The rules are a port of the shengji Rust project (`../shengji`, `mechanics/src/*.rs`, `core/src/game_state/*.rs`, `core/src/settings.rs`). When a rule question comes up, read the Rust first and keep behavior the same; deliberate differences are listed in the README checklist. The architecture follows `../catan` and `../card-collecting` (same room transport, ActionSync and store design).

## Tech stack (locked)

TypeScript, Vite, React 19, Zustand, Firebase Realtime Database + anonymous Auth (lazy-loaded), Vitest. Card art is static SVG in `public/cards` and `public/cards-4c`.

## Commands

- `npm run dev`, `npm run build` (tsc + vite), `npm run test:run`, `npx tsc --noEmit -p .`
- `npm run sim -- <players> <matches> <levels>` — headless AI matches (vite-node `training/simulate.ts`)
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
- **`src/ai/`** — `observation.ts` (`observe(state, seat)`: only what that seat may see — every AI decision goes through it), `plays.ts` (legal lead / follow candidates from the seat's own hand), `heuristic.ts` (Easy / Normal), `index.ts` (`AI_LEVELS`, `chooseAIAction`, `prepareAi` hook for loading trained policies), `driver.ts` (`nextAutomatedStep`: live deal, AI moves, all-AI round advance — used by the UI driver and the simulator), `simulate.ts`.
- **`src/net/`** — room transport copied from card-collecting with catan's shared-database changes: `transport.ts` (interface, log-slot encoding), `firebaseRoom.ts` (real, lazy-loaded, everything under `DB_ROOT = 'tractor'`, day-bucketed stale-room index), `fakeTransport.ts` (tests), `firebaseConfig.ts`, `identity.ts`, `roomLink.ts`, `types.ts`.
- **`src/store/`** — `gameStore` (state + `dispatch` returning an error string or null, `applyLocal`, `restoreState`), `networkStore` (rooms, lobby transactions, presence, chat, which device drives), `actionSync.ts` (write-once log slots, rollback + replay on races; copied unchanged), `prefsStore` (four-color cards, AI speed; localStorage).
- **`src/ui/`** — `App` (home / lobby / game, or the room's screens), `LocalLobby`, `OnlineLobby`, `TractorOptions` (shared options form), `GameHost` (top bar, the driver effect, hot-seat pass screen, online banners), `table/` (`Table`, `Felt`, `Hand`, `ActionPanel`, `SidePanel`, `RoundSummary`, `labels`), `CardView`, `Rules`, `lobbyParts`, `ChatPanel`, `OnlineStatus`. Styles: `theme.css` (shell, from card-collecting) + `table.css`.

## Invariants

- `applyAction` and `createInitialState` are pure and deterministic; shuffles use `state.rngState`. Online, every device rebuilds the match from the start record (config, seed, seats) and replays the action log, so anything that changes the reducer's output for the same inputs breaks running rooms — bump `PROTOCOL_VERSION` (and the `v` check in the rules) when the room layout changes.
- State is plain JSON (no Maps, classes or undefined-sensitive fields). Payloads to Firebase are JSON strings.
- Actions name their seat by index (`seat`); seats are in play order (clockwise; Tractor partners are same-parity seats).
- Hidden information is a UI concern (every device holds the full state; friends-only honor). AI code must read only an `Observation`.
- `activePlayerId` is null during the live deal (anyone may bid) and between rounds.
- One device drives a match (deals, AI seats, next round): locally this one; online the host while connected, else the first connected seated human (`networkStore.aiDriver`). Driver moves are dispatched with `{ ai: true }` so a lost race is dropped quietly.
- Hot-seat (2+ human seats on one device) always bids in turns (`validateConfig` rejects live bidding there; `biddingStyle: 'auto'` resolves it).

## Firebase (shared project)

Project `github-games-e7c7d`, web app "Tractor" (config in `src/net/firebaseConfig.ts`). Other apps (catan, card-collecting, …) use the same database under their own top-level keys. Rules:

- `database.rules.json` defines only `tractor` (root `.read`/`.write` stay false).
- `firebase.json`'s predeploy runs `scripts/merge-db-rules.mjs` (copied from catan): reads the live rules, replaces the `tractor` key, keeps the rest, writes `.firebase/database.rules.deploy.json`, which is what gets deployed. Never deploy a rules file that would drop other apps' keys.
- Layout under `/tractor`: `rooms/{CODE}/{meta, lobby, matches/m{N}/{start, actions/aNNNNNN}, chat, presence}`, plus `roomDays/{day}` and `roomIndex/{day}/{CODE}` for the week-later sweep.

## Conventions

- Keep the README checklist current: tick what's done, add what's discovered.
- Rule changes: add or port a test (shengji's Rust tests are the reference — `trick.test.ts` ports trick.rs; `rules.test.ts` ports the bidding / scoring / game-state tests and holds the scenario tests, built with `src/game/testing.ts`'s `scenario()` which sets hands, kitty, levels and landlord directly), and make sure `engine.test.ts`'s full AI matches still finish.
- The one deliberate difference from shengji: when a follow can be read several ways, a later play must beat *every* winning reading of the current winner (`computeWinner` in `trick.ts`); see the README's "Rule decisions".
- Windows + Git Bash: heredocs with non-ASCII text can break; write scripts to files. Use `MSYS_NO_PATHCONV=1` for firebase-tools database paths.
- The local npm registry is an Artifactory proxy; CI uses `npm install` against npmjs.org via the project `.npmrc` (the lockfile carries local URLs).
