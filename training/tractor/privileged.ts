// What the value head may see in training and the policy never does: every
// seat's hand, the kitty (face down or buried) and the undealt deck. A critic
// that knows the cards judges a position far more accurately than one that
// has to guess them, so advantages are less noisy and learning is faster
// ("perfect-information critic, imperfect-information actor"). The policy,
// and the shipped AI, still only ever see an Observation.
//
// Same seat slots as the observation (encode.ts seatSlot), relative to the
// deciding seat; same trump-relative card slots.

import { bidLevelFor } from '@/game/engine';
import type { Trump } from '@/game/trump';
import type { TractorState } from '@/game/types';
import { MAX_SEATS, seatSlot } from '@/ai/nn/encode';
import { CARD_SLOTS, addCards } from '@/ai/nn/slots';

/** Kitty, undealt deck. */
export const PRIV_GLOBAL_DIM = 2 * CARD_SLOTS;
/** That seat's hand. */
export const PRIV_SEAT_DIM = CARD_SLOTS;

export function encodePrivileged(state: TractorState, seat: number, glob: Float32Array, seats: Float32Array): void {
  glob.fill(0);
  seats.fill(0);
  const r = state.round;
  const n = state.seats.length;
  const w = 1 / state.rules.numDecks;
  let trump: Trump | null = r.trump;
  if (!trump) {
    const level = bidLevelFor(state, seat);
    trump = { kind: 'noTrump', number: level === 'NT' ? null : level };
  }
  addCards(glob, 0, trump, r.kitty, w);
  addCards(glob, CARD_SLOTS, trump, r.deck, w);
  for (let s = 0; s < n; s++) {
    const slot = seatSlot((s - seat + n) % n, n);
    if (slot < MAX_SEATS) addCards(seats, slot * PRIV_SEAT_DIM, trump, r.hands[s]!, w);
  }
}
