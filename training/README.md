# training/

Offline tooling for AI players. Nothing here ships to GitHub Pages.

Today: `simulate.ts`, headless AI-vs-AI matches through the real reducer (`npm run sim -- 4 20 normal,easy`), for checking the engine and comparing AI levels.

## Plan for the learned AI

The goal is trained networks at several difficulty levels, alongside the heuristic Easy / Normal. The pieces are laid out so that work slots in without touching the rules or the UI:

1. **Observation** — `src/ai/observation.ts` already gives each seat only what it may see (own hand, public tricks, bids, levels, friend calls, the kitty if it buried it). The encoder turns an `Observation` into a fixed-size feature vector: own hand as per-card counts in the trump-relative order (suits by `suitOrdinal`, cards by `strength`), cards still unseen, known voids per seat (`knownVoids`), the current trick (format shapes, each play, current winner, whether partner), points so far, levels, role, stage.
2. **Action space** — tractor plays are combinatorial, so the network scores *candidates* rather than picking from fixed slots: `src/ai/plays.ts` (`leadCandidates`, `followCandidates`) lists legal plays; each candidate gets features (shape, cards, points, whether it would win now, cost), and a policy head scores candidate ⊕ observation. Bids, kitty burial and friend calls get small dedicated heads (bury = pick cards one at a time with a mask).
3. **Env server** — like catan's `training/catan/env-server.ts`: a vite-built Node process running many matches (`src/ai/simulate.ts` / `nextAutomatedStep` for the non-learner seats), talking to Python over stdio in batches.
4. **Trainer** — PyTorch: imitation of the Normal heuristic first (DAgger), then PPO with self-play + heuristic + past-checkpoint opponents, as in catan's trainer. Reward: levels gained/lost per round (shaped by points captured).
5. **Shipping** — export a small MLP (card-collecting's `mlp@1` format: base64 float32 weights) per difficulty checkpoint; add the levels to `AI_LEVELS` in `src/ai/index.ts` and load the files in `prepareAi` (the lobby and the online host already await it before a match).
