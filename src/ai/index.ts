// The AI players. Every decision goes through an Observation (what the seat
// may see), then a trained network (src/ai/nn): Easy, Normal and Hard are
// three checkpoints of one training run (nn/levels.ts).

import type { Seat } from '@/core/types';
import type { Action, TractorState } from '@/game/types';
import { actWithPolicy, netPolicy, type BidMemo } from './nn/agent';
import { NN_LEVELS, loadNnLevel, loadedNnLevel } from './nn/levels';

export type AiLevel = string;

/** Live-deal declines of the AI seats on this device (see BidMemo). */
const bidMemo: BidMemo = new Map();

/** Every level an AI seat can play at, weakest first. */
export const AI_LEVELS: ReadonlyArray<{ id: AiLevel; label: string }> = NN_LEVELS.map((l) => ({ id: l.id, label: l.label }));
export const DEFAULT_AI_LEVEL: AiLevel = 'normal';
/** The level that gives hints and suggestions to human players. */
export const ADVICE_LEVEL: AiLevel = 'hard';

export function seatAiLevel(seat: Pick<Seat, 'aiLevel'> | undefined): AiLevel {
  const id = seat?.aiLevel;
  return AI_LEVELS.some((l) => l.id === id) ? id! : DEFAULT_AI_LEVEL;
}

/**
 * The AI's next action for `seat`, or null when it has nothing to do (or
 * chooses to wait). The level's network must be loaded (prepareAi, or
 * adviseAction); until then the seat just waits.
 */
export function chooseAIAction(state: TractorState, seat: number, level: AiLevel = seatAiLevel(state.seats[seat])): Action | null {
  const net = loadedNnLevel(level);
  return net ? actWithPolicy(state, seat, netPolicy(net), bidMemo) : null;
}

/** `seat`'s best action by `level`'s network, loading it first if need be (hints, taking over for a player who dropped). */
export async function adviseAction(state: TractorState, seat: number, level: AiLevel = ADVICE_LEVEL): Promise<Action | null> {
  await loadNnLevel(level);
  return chooseAIAction(state, seat, level);
}

/**
 * Load whatever the AI seats need before a match: the policy file of every
 * level in play. The online host and the lobby await this.
 */
export async function prepareAi(seats: readonly Seat[]): Promise<void> {
  const levels = new Set(seats.filter((s) => s.isAI).map((s) => seatAiLevel(s)));
  await Promise.all([...levels].map(loadNnLevel));
}

export { observe } from './observation';
export type { Observation } from './observation';
