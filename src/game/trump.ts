// Trump-aware card ordering — a port of shengji's `Trump` (mechanics/src/types.rs).
//
// Under a trump declaration every card has an *effective suit*: the jokers,
// every card of the trump number and every card of the trump suit are
// "trump"; the rest keep their printed suit. Within trump the order is
//   trump-suit cards 2..A (skipping the number) < off-suit number cards
//   (all equal to each other) < the trump-suit number card < small joker
//   < big joker.
// `successor` gives the card(s) one step up, which is what makes tractors
// (consecutive pairs) — e.g. with 4♠ trump, A♠A♠ + 4♥4♥ is a tractor, and the
// number skips out of its suit (3♥3♥5♥5♥ is a tractor when 4 is trump).

import {
  BIG_JOKER,
  SMALL_JOKER,
  SUITS,
  cardRank,
  cardSuit,
  isJoker,
  rankSuccessor,
  rankValue,
  type Card,
  type Rank,
  type Suit,
} from './cards';

export type Trump =
  | { kind: 'standard'; suit: Suit; number: Rank }
  /** No trump suit. `number` is null when the landlord's level is NT (only jokers are trump). */
  | { kind: 'noTrump'; number: Rank | null };

/** A card's suit for following: a printed suit, or 'T' for trump. */
export type EffSuit = Suit | 'T';

export function trumpSuit(trump: Trump): Suit | null {
  return trump.kind === 'standard' ? trump.suit : null;
}

export function trumpNumber(trump: Trump): Rank | null {
  return trump.number;
}

export function effectiveSuit(trump: Trump, card: Card): EffSuit {
  if (isJoker(card)) return 'T';
  const rank = cardRank(card)!;
  const suit = cardSuit(card)!;
  if (trump.number !== null && rank === trump.number) return 'T';
  if (trump.kind === 'standard' && suit === trump.suit) return 'T';
  return suit;
}

export function isTrump(trump: Trump, card: Card): boolean {
  return effectiveSuit(trump, card) === 'T';
}

// Suit ordinals per trump suit, exactly as shengji orders them (so hands sort
// the same way): the trump suit's own slot is never used since its cards are 'T'.
const ORDINALS: Record<Suit | 'none', Record<EffSuit, number>> = {
  C: { D: 1, S: 2, H: 3, C: 4, T: 5 },
  D: { S: 1, H: 2, C: 3, D: 4, T: 5 },
  S: { H: 1, C: 2, D: 3, S: 4, T: 5 },
  H: { C: 1, D: 2, S: 3, H: 4, T: 5 },
  none: { C: 1, D: 2, S: 3, H: 4, T: 5 },
};

export function suitOrdinal(trump: Trump, card: Card): number {
  const table = ORDINALS[trump.kind === 'standard' ? trump.suit : 'none'];
  return table[effectiveSuit(trump, card)];
}

/** Strength within the card's effective suit (equal for off-suit number cards). */
export function strength(trump: Trump, card: Card): number {
  if (card === BIG_JOKER) return 100;
  if (card === SMALL_JOKER) return 99;
  const rank = cardRank(card)!;
  if (trump.number !== null && rank === trump.number) {
    return trump.kind === 'standard' && cardSuit(card) === trump.suit ? 98 : 97;
  }
  return rankValue(rank);
}

/** Order ignoring the tie-break: off-suit trump-number cards compare equal. */
export function compareEffective(trump: Trump, a: Card, b: Card): number {
  if (a === b) return 0;
  const so = suitOrdinal(trump, a) - suitOrdinal(trump, b);
  if (so !== 0) return so;
  return strength(trump, a) - strength(trump, b);
}

/** Total order: effective order, then a fixed suit tie-break (♠ < ♥ < ♦ < ♣, as shengji). */
export function compareCards(trump: Trump, a: Card, b: Card): number {
  const e = compareEffective(trump, a, b);
  if (e !== 0 || a === b) return e;
  return suitTie(a) - suitTie(b);
}

function suitTie(card: Card): number {
  const s = cardSuit(card);
  return s === null ? (card === SMALL_JOKER ? 4 : 5) : SUITS.indexOf(s);
}

export function sortCards(trump: Trump, cards: readonly Card[]): Card[] {
  return cards.slice().sort((a, b) => compareCards(trump, a, b));
}

/** The card(s) one step above `card` for tractor purposes (several only for the off-suit numbers). */
export function successor(trump: Trump, card: Card): Card[] {
  if (card === BIG_JOKER) return [];
  if (card === SMALL_JOKER) return [BIG_JOKER];
  const rank = cardRank(card)!;
  const suit = cardSuit(card)!;
  const num = trump.number;
  if (num !== null && rank === num) {
    if (trump.kind === 'standard') return suit === trump.suit ? [SMALL_JOKER] : [num + trump.suit];
    return [SMALL_JOKER];
  }
  const next = rankSuccessor(rank);
  if (next !== null && next === num) {
    // The next rank is the trump number: skip it.
    const after = rankSuccessor(next);
    if (after !== null) return [after + suit];
    return aboveTop(trump, card);
  }
  if (next !== null) return [next + suit];
  return aboveTop(trump, card);
}

/** Past the ace: in the trump suit, the off-suit number cards (or the small joker); elsewhere nothing. */
function aboveTop(trump: Trump, card: Card): Card[] {
  if (effectiveSuit(trump, card) !== 'T') return [];
  if (trump.number !== null) {
    const ts = trumpSuit(trump);
    return SUITS.filter((s) => s !== ts).map((s) => trump.number + s);
  }
  return [SMALL_JOKER];
}

export function trumpLabel(trump: Trump): string {
  const num = trump.number === null ? '' : trump.number === 'T' ? '10' : trump.number;
  if (trump.kind === 'noTrump') return trump.number === null ? 'No trump' : `No trump · ${num}s`;
  const sym = { S: '♠', H: '♥', D: '♦', C: '♣' }[trump.suit];
  return `${sym} ${num}`;
}
