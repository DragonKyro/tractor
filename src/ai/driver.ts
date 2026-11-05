// What happens next without a human click: the live deal, AI decisions, and
// (all-AI tables) moving on to the next round. Pure — the UI's AI driver and
// the headless simulator both call it and dispatch the step it returns.

import { activeSeat } from '@/game/engine';
import type { Action, TractorState } from '@/game/types';
import { chooseAIAction } from './index';

/**
 * How long a step should wait when a human is watching:
 *   deal   — the next round of cards in a live deal
 *   ai     — an AI's move
 *   linger — a step that closes a window humans might still act in
 *            (taking the kitty while others could still bid, moving on from
 *            a finished round)
 */
export type Pace = 'deal' | 'ai' | 'linger';

export interface DriverStep {
  action: Action;
  pace: Pace;
}

export function nextAutomatedStep(state: TractorState, isAI: (seat: number) => boolean): DriverStep | null {
  if (state.status !== 'playing') return null;
  const r = state.round;
  const n = state.seats.length;
  if (r.stage === 'draw') {
    // Give every AI a chance to declare as its cards arrive.
    for (let seat = 0; seat < n; seat++) {
      if (!isAI(seat)) continue;
      const a = chooseAIAction(state, seat);
      if (a?.type === 'bid') return { action: a, pace: 'ai' };
    }
    if (r.deck.length > 0) return { action: { type: 'draw' }, pace: 'deal' };
  }
  if (r.stage === 'roundOver') {
    const allAI = state.seats.every((_, i) => isAI(i));
    return allAI ? { action: { type: 'nextRound' }, pace: 'linger' } : null;
  }
  const seat = activeSeat(state);
  if (seat === null || !isAI(seat)) return null;
  const a = chooseAIAction(state, seat);
  if (!a) return null;
  return { action: a, pace: r.stage === 'draw' ? 'linger' : 'ai' };
}
