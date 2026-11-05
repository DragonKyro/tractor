// The AI players. Every decision goes through an Observation (what the seat
// may see), then a policy: the heuristic levels today; trained networks at
// more difficulty levels later (training/README.md).

import type { Seat } from '@/core/types';
import type { Action, TractorState } from '@/game/types';
import { chooseHeuristicAction, type HeuristicLevel } from './heuristic';
import { observe } from './observation';

export type AiLevel = HeuristicLevel;

export const AI_LEVELS: ReadonlyArray<{ id: AiLevel; label: string }> = [
  { id: 'easy', label: 'Easy' },
  { id: 'normal', label: 'Normal' },
];
export const DEFAULT_AI_LEVEL: AiLevel = 'normal';

export function seatAiLevel(seat: Pick<Seat, 'aiLevel'> | undefined): AiLevel {
  const id = seat?.aiLevel;
  return AI_LEVELS.some((l) => l.id === id) ? (id as AiLevel) : DEFAULT_AI_LEVEL;
}

/** The AI's next action for `seat`, or null when it has nothing to do (or chooses to wait). */
export function chooseAIAction(state: TractorState, seat: number, level: AiLevel = seatAiLevel(state.seats[seat])): Action | null {
  return chooseHeuristicAction(observe(state, seat), level);
}

export { observe } from './observation';
export type { Observation } from './observation';

/**
 * Load whatever the AI seats need before a match (trained policy files, once
 * they exist). The heuristic levels need nothing, so this resolves at once;
 * the online host and the lobby already await it, so a learned level only
 * has to fill it in.
 */
export async function prepareAi(_seats: readonly Seat[]): Promise<void> {}
