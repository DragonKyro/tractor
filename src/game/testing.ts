// Test helpers: build a match with chosen hands, kitty, levels and landlord,
// then drive it with plain actions. Used by the rules tests (scenarios ported
// from shengji's game_state tests and our own). Not imported by app code.

import type { Seat } from '@/core/types';
import { buildDecks, countCards, type Card, type Level } from './cards';
import { defaultConfig, type TractorConfig } from './config';
import { activeSeat, applyAction, createInitialState } from './engine';
import type { Action, TractorState } from './types';

export function plainSeats(n: number): Seat[] {
  return Array.from({ length: n }, (_, i) => ({ id: `p${i + 1}`, name: `P${i + 1}`, color: '#888888', isAI: true, isLocal: true }));
}

/**
 * A round where the deal is already done: `hands[i]` is seat i's hand, the
 * kitty face down. Live bidding by default (bids then "take the kitty");
 * `turns: true` starts turn bidding at `dealer` instead.
 */
export function scenario(opts: {
  hands: Card[][];
  kitty?: Card[];
  config?: Partial<TractorConfig>;
  landlord?: number | null;
  levels?: Level[];
  dealer?: number;
  turns?: boolean;
}): TractorState {
  const n = opts.hands.length;
  const config: TractorConfig = { ...defaultConfig(n), biddingStyle: opts.turns ? 'turns' : 'live', ...(opts.config ?? {}) };
  const s = structuredClone(createInitialState(config, 7, plainSeats(n)));
  const landlord = opts.landlord ?? null;
  if (opts.levels) s.levels = opts.levels.slice();
  s.landlord = landlord;
  const r = s.round;
  r.landlord = landlord;
  r.hands = opts.hands.map((h) => h.slice());
  r.deck = [];
  r.kitty = (opts.kitty ?? []).slice();
  r.dealtKitty = r.kitty.slice();
  r.kittySize = r.kitty.length;
  // Cards not in any hand or the kitty count as removed, so a small
  // scenario is a consistent deck (point totals add up).
  const left = countCards(buildDecks(s.rules.numDecks));
  for (const c of [...r.hands.flat(), ...r.kitty]) left.set(c, (left.get(c) ?? 0) - 1);
  r.removed = [...left].flatMap(([c, k]) => Array<Card>(Math.max(0, k)).fill(c));
  r.dealer = opts.dealer ?? landlord ?? 0;
  r.dealTo = r.dealer;
  r.captured = r.hands.map(() => []);
  r.penalties = r.hands.map(() => 0);
  if (opts.turns) {
    r.stage = 'bid';
    r.bidTurn = r.dealer;
    r.passes = 0;
  } else {
    r.stage = 'draw';
  }
  s.log = [];
  s.logCount = 0;
  const a = activeSeat(s);
  s.activePlayerId = a === null ? null : s.seats[a]!.id;
  return s;
}

export function act(s: TractorState, ...actions: Action[]): TractorState {
  for (const a of actions) s = applyAction(s, a);
  return s;
}

export const bid = (seat: number, card: Card, count = 1): Action => ({ type: 'bid', seat, card, count });
export const pass = (seat: number): Action => ({ type: 'pass', seat });
export const pickUp = (seat: number): Action => ({ type: 'pickUpKitty', seat });
export const play = (seat: number, ...cards: Card[]): Action => ({ type: 'play', seat, cards });
export const exchange = (seat: number, bury: Card[], friends: Array<{ card: Card; ordinal: number }> = []): Action => ({
  type: 'exchange',
  seat,
  bury,
  friends,
});
