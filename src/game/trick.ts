// One trick: the lead sets the format, everyone follows in seat order, the
// best matching play wins. Port of shengji's `Trick` (mechanics/src/trick.rs).
// Pure: every function returns new objects.

import { cardPoints, containsAll, removeCards, type Card } from './cards';
import {
  TrickError,
  defeats,
  findThrowBreaker,
  formatFromCards,
  isLegalFollow,
  matchFormat,
  normalizeUnits,
  unitCards,
  unitSize,
  type ThrowEvaluationPolicy,
  type TractorRequirements,
  type TrickDrawPolicy,
  type TrickFormat,
  type TrickUnit,
} from './format';
import { effectiveSuit, sortCards, type Trump } from './trump';

export interface PlayedCards {
  seat: number;
  cards: Card[];
  /** A failed throw: the cards the leader tried to throw but took back. */
  badThrowCards: Card[];
  /** A failed throw: whose hand stopped it. */
  betterPlayer: number | null;
}

export interface TrickState {
  /** Seats in play order, leader first. */
  order: number[];
  plays: PlayedCards[];
  format: TrickFormat | null;
  /** Seat currently winning (null before the lead). */
  winner: number | null;
}

export interface TrickRules {
  trickDrawPolicy: TrickDrawPolicy;
  throwEvaluationPolicy: ThrowEvaluationPolicy;
  tractorRequirements: TractorRequirements;
}

export function newTrick(leader: number, numSeats: number): TrickState {
  const order: number[] = [];
  for (let i = 0; i < numSeats; i++) order.push((leader + i) % numSeats);
  return { order, plays: [], format: null, winner: null };
}

export function nextToPlay(trick: TrickState): number | null {
  return trick.order[trick.plays.length] ?? null;
}

export function trickComplete(trick: TrickState): boolean {
  return trick.plays.length === trick.order.length;
}

/** Why `cards` can't be played by `seat` now, or null when they can (ignoring throw validity). */
export function playProblem(
  trick: TrickState,
  trump: Trump,
  hand: readonly Card[],
  seat: number,
  cards: readonly Card[],
  rules: TrickRules,
): string | null {
  if (nextToPlay(trick) !== seat) return "It isn't your turn.";
  if (cards.length === 0) return 'Pick some cards to play.';
  if (!containsAll(hand, cards)) return "Those cards aren't all in your hand.";
  if (!trick.format) {
    const suit = effectiveSuit(trump, cards[0]!);
    if (cards.some((c) => effectiveSuit(trump, c) !== suit)) return 'A lead must be all one suit (or all trump).';
    return null;
  }
  const need = trick.format.units.reduce((n, u) => n + unitSize(u), 0);
  if (cards.length !== need) return `Play exactly ${need} card${need === 1 ? '' : 's'}.`;
  if (!isLegalFollow(trump, trick.format, hand, cards, rules.trickDrawPolicy)) return 'illegal';
  return null;
}

export interface PlayOutcome {
  trick: TrickState;
  /** The player's hand afterwards. */
  hand: Card[];
  /** Set when a throw failed: what was attempted, and who stopped it. */
  failedThrow: { attempted: Card[]; forced: Card[]; betterPlayer: number } | null;
}

/**
 * Play `cards` for `seat`. `hands` is every seat's hand (used to test a
 * throw against the other players). Throws TrickError when not allowed.
 */
export function playCards(
  trick: TrickState,
  trump: Trump,
  hands: readonly (readonly Card[])[],
  seat: number,
  cards0: readonly Card[],
  rules: TrickRules,
  formatHint?: number,
): PlayOutcome {
  const hand = hands[seat]!;
  const problem = playProblem(trick, trump, hand, seat, cards0, rules);
  if (problem) throw new TrickError('illegalPlay', problem === 'illegal' ? 'That play does not follow the lead.' : problem);
  let cards = sortCards(trump, cards0);
  let format = trick.format;
  let badThrowCards: Card[] = [];
  let betterPlayer: number | null = null;
  let failedThrow: PlayOutcome['failedThrow'] = null;

  if (!format) {
    format = formatFromCards(trump, rules.tractorRequirements, cards, formatHint);
    if (format.units.length > 1) {
      const others = trick.order.slice(1).map((s) => ({ seat: s, hand: hands[s]! }));
      const breaker = findThrowBreaker(trump, format, others);
      if (breaker) {
        const forced = sortCards(trump, unitCards(breaker.unit));
        badThrowCards = removeCards(cards, forced);
        failedThrow = { attempted: cards, forced, betterPlayer: breaker.seat };
        betterPlayer = breaker.seat;
        cards = forced;
        format = { suit: format.suit, units: [breaker.unit] };
      }
    }
  }

  const next: TrickState = {
    order: trick.order,
    plays: [...trick.plays, { seat, cards, badThrowCards, betterPlayer }],
    format,
    winner: null,
  };
  next.winner = computeWinner(next, trump, rules.throwEvaluationPolicy);
  return { trick: next, hand: removeCards(hand, cards), failedThrow };
}

/** The seat whose play is currently best. */
export function computeWinner(trick: TrickState, trump: Trump, policy: ThrowEvaluationPolicy): number | null {
  const format = trick.format;
  const lead = trick.plays[0];
  if (!format || !lead) return null;
  let winnerSeat = lead.seat;
  let winnerUnits: TrickUnit[] = normalizeUnits(trump, format, format.units);
  for (let i = 1; i < trick.plays.length; i++) {
    const play = trick.plays[i]!;
    const m = matchFormat(trump, format, play.cards).find((units) => defeats(trump, units, winnerUnits, policy));
    if (m) {
      winnerSeat = play.seat;
      winnerUnits = m;
    }
  }
  return winnerSeat;
}

export interface TrickResult {
  winner: number;
  /** Point cards played in the trick. */
  pointCards: Card[];
  points: number;
  /** Size of the largest unit in the (final) format — the kitty multiplier is based on it. */
  largestUnitSize: number;
  /** Cards the leader failed to throw (0 when no throw failed). */
  failedThrowSize: number;
}

export function completeTrick(trick: TrickState): TrickResult {
  if (!trickComplete(trick) || !trick.format || trick.winner === null) throw new Error('trick is not finished');
  const pointCards = trick.plays.flatMap((p) => p.cards.filter((c) => cardPoints(c) > 0));
  return {
    winner: trick.winner,
    pointCards,
    points: pointCards.reduce((n, c) => n + cardPoints(c), 0),
    largestUnitSize: Math.max(0, ...trick.format.units.map(unitSize)),
    failedThrowSize: trick.plays[0]!.badThrowCards.length,
  };
}
