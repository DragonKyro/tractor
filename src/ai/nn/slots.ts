// Trump-relative card slots: every card maps to one of CARD_SLOTS positions
// by the role it plays under the current trump, not by its printed suit, so
// "my hand with ♥ trump" and "the same hand with ♠ trump" encode the same.
//
//   0..51   four side-suit blocks × 13 ranks (2..A; the trump number's rank
//           stays empty — those cards are trump). Side suits come in a fixed
//           rotation after the trump suit (no trump suit: ♠ ♥ ♦ ♣).
//   52..64  the trump suit by rank (empty under no-trump)
//   65..68  the trump number in side block 0..3 (off-suit numbers)
//   69      the trump number in the trump suit
//   70, 71  small joker, big joker
//
// Ranks keep their own positions so point cards (5, 10, K) always sit in the
// same slots. Before trump is declared, slots use "no trump suit at the
// bidding level": each suit is a side block, and a bid's suit is its block.

import { BIG_JOKER, SMALL_JOKER, SUITS, RANKS, cardRank, cardSuit, cardPoints, type Card, type Rank, type Suit } from '@/game/cards';
import type { Trump } from '@/game/trump';

export const CARD_SLOTS = 72;
export const TRUMP_SUIT_BASE = 52;
export const OFF_NUMBER_BASE = 65;
export const ON_NUMBER = 69;
export const SLOT_SMALL_JOKER = 70;
export const SLOT_BIG_JOKER = 71;
/** Groups for "which suit is this": side blocks 0..3 and trump (4). */
export const GROUPS = 5;
export const TRUMP_GROUP = 4;

/** The printed suits in side-block order under `trump`. */
export function sideSuits(trump: Trump): Suit[] {
  if (trump.kind !== 'standard') return [...SUITS];
  const i = SUITS.indexOf(trump.suit);
  return [1, 2, 3].map((k) => SUITS[(i + k) % 4]!);
}

/** Side block of a printed suit (−1 for the trump suit). */
export function sideBlock(trump: Trump, suit: Suit): number {
  return sideSuits(trump).indexOf(suit);
}

export function slotOf(trump: Trump, card: Card): number {
  if (card === SMALL_JOKER) return SLOT_SMALL_JOKER;
  if (card === BIG_JOKER) return SLOT_BIG_JOKER;
  const rank = cardRank(card)!;
  const suit = cardSuit(card)!;
  const r = RANKS.indexOf(rank);
  if (trump.number !== null && rank === trump.number) {
    if (trump.kind === 'standard' && suit === trump.suit) return ON_NUMBER;
    return OFF_NUMBER_BASE + sideBlock(trump, suit);
  }
  if (trump.kind === 'standard' && suit === trump.suit) return TRUMP_SUIT_BASE + r;
  return sideBlock(trump, suit) * 13 + r;
}

/** The group (side block 0..3, or 4 = trump) of a slot. */
export function groupOfSlot(slot: number): number {
  return slot < TRUMP_SUIT_BASE ? Math.floor(slot / 13) : TRUMP_GROUP;
}

/** The card at a slot under `trump` (null for a slot no card maps to). */
export function cardAtSlot(trump: Trump, slot: number): Card | null {
  if (slot === SLOT_SMALL_JOKER) return SMALL_JOKER;
  if (slot === SLOT_BIG_JOKER) return BIG_JOKER;
  const sides = sideSuits(trump);
  if (slot === ON_NUMBER) return trump.kind === 'standard' && trump.number ? trump.number + trump.suit : null;
  if (slot >= OFF_NUMBER_BASE) {
    const s = sides[slot - OFF_NUMBER_BASE];
    return s && trump.number ? trump.number + s : null;
  }
  const rank = RANKS[slot % 13]! as Rank;
  if (trump.number !== null && rank === trump.number) return null;
  if (slot >= TRUMP_SUIT_BASE) return trump.kind === 'standard' ? rank + trump.suit : null;
  const s = sides[Math.floor(slot / 13)];
  return s ? rank + s : null;
}

/** Add `cards` into a slot plane at `offset`, each copy worth `weight`. */
export function addCards(out: Float32Array, offset: number, trump: Trump, cards: Iterable<Card>, weight: number): void {
  for (const c of cards) out[offset + slotOf(trump, c)]! += weight;
}

/** Points held in slot `slot` per copy (fixed by rank, the same under any trump). */
export function slotPoints(trump: Trump, slot: number): number {
  const c = cardAtSlot(trump, slot);
  return c ? cardPoints(c) : 0;
}
