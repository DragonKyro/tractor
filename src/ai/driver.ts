// What happens next without a human click: AI seats drawing and declaring
// during the live deal, AI decisions, and (all-AI tables) moving on to the
// next round. Pure — the UI's driver and the headless simulator both call it
// and dispatch the step it returns. Human seats draw for themselves (auto-draw
// or a click, on their own device); see GameHost.

import { activeSeat } from '@/game/engine';
import type { Action, TractorState } from '@/game/types';
import { chooseAIAction } from './index';

/**
 * How long a step should wait when a human is watching:
 *   deal      — an AI seat drawing its next card
 *   ai        — an AI's move
 *   lead      — the first lead after a trick (time to see who won it)
 *   bidWindow — an AI closing the bidding after the deal (taking the kitty,
 *               turning it over, redealing) while humans might still overcall
 *   roundEnd  — all-AI tables moving on to the next round
 */
export type Pace = 'deal' | 'ai' | 'lead' | 'bidWindow' | 'roundEnd';

export interface DriverStep {
  action: Action;
  pace: Pace;
}

export interface DriverOptions {
  /** Seats this device draws for during a live deal (default: the AI seats). */
  drawsFor?: (seat: number) => boolean;
}

export function nextAutomatedStep(state: TractorState, isAI: (seat: number) => boolean, opts: DriverOptions = {}): DriverStep | null {
  if (state.status !== 'playing') return null;
  const r = state.round;
  const n = state.seats.length;
  if (r.stage === 'draw') {
    // Give every AI a chance to declare as its cards arrive (and after the deal).
    for (let seat = 0; seat < n; seat++) {
      if (!isAI(seat)) continue;
      const a = chooseAIAction(state, seat);
      if (a?.type === 'bid') return { action: a, pace: 'ai' };
    }
    if (r.deck.length > 0) {
      const seat = r.dealTo;
      return (opts.drawsFor ?? isAI)(seat) ? { action: { type: 'draw', seat }, pace: 'deal' } : null;
    }
    const seat = activeSeat(state);
    if (seat === null || !isAI(seat)) return null;
    const a = chooseAIAction(state, seat);
    if (!a) return null;
    // Close the bidding at once only when every human has said they're done.
    const humansDone = state.seats.every((_, i) => isAI(i) || i === seat || r.bidDone.includes(i));
    return { action: a, pace: humansDone ? 'ai' : 'bidWindow' };
  }
  if (r.stage === 'roundOver') {
    const allAI = state.seats.every((_, i) => isAI(i));
    return allAI ? { action: { type: 'nextRound' }, pace: 'roundEnd' } : null;
  }
  const seat = activeSeat(state);
  if (seat === null || !isAI(seat)) return null;
  const a = chooseAIAction(state, seat);
  if (!a) return null;
  const firstLead = r.stage === 'play' && r.trick?.plays.length === 0 && r.tricks.length > 0;
  return { action: a, pace: firstLead ? 'lead' : 'ai' };
}
