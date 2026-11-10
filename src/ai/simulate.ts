// Headless matches: AI vs AI through the real reducer, no React or timers.
// Used by the tests (every configuration must play to the end) and by the
// training tools (training/simulate.ts), and the seed of the future env
// server for a learned agent.

import type { Seat } from '@/core/types';
import { SEAT_COLORS } from '@/core/seatColors';
import { defaultConfig, type TractorConfig } from '@/game/config';
import { applyAction, createInitialState } from '@/game/engine';
import type { TractorState } from '@/game/types';
import { nextAutomatedStep } from './driver';

export function aiSeats(n: number): Seat[] {
  return Array.from({ length: n }, (_, i) => ({
    id: `ai-${i + 1}`,
    name: `AI ${i + 1}`,
    color: SEAT_COLORS[i % SEAT_COLORS.length]!,
    isAI: true,
    isLocal: true,
  }));
}

export interface SimResult {
  state: TractorState;
  steps: number;
  rounds: number;
  finished: boolean;
}

/** Play until the match ends, `maxRounds` rounds finish, or `maxSteps` actions (a stuck engine). */
export function simulateMatch(opts: {
  seats: Seat[];
  config?: Partial<TractorConfig>;
  seed?: number;
  maxRounds?: number;
  maxSteps?: number;
}): SimResult {
  const config = { ...defaultConfig(opts.seats.length), ...(opts.config ?? {}) };
  let state = createInitialState(config, opts.seed ?? 1, opts.seats);
  const maxSteps = opts.maxSteps ?? 200_000;
  const maxRounds = opts.maxRounds ?? Infinity;
  let steps = 0;
  const isAI = (seat: number) => state.seats[seat]!.isAI;
  while (state.status === 'playing' && steps < maxSteps && state.history.length < maxRounds) {
    const step = nextAutomatedStep(state, isAI);
    if (!step) throw new Error(`Simulation stuck at stage ${state.round.stage} (round ${state.round.number}).`);
    state = applyAction(state, step.action);
    steps++;
  }
  return { state, steps, rounds: state.history.length, finished: state.status === 'gameOver' };
}
