// What one seat can legitimately see. Every AI decision is made from an
// Observation, never from the full state, so AI players (and the future
// trained network — see training/README.md) can't peek at hidden hands.
//
// Public: levels, bids, the landlord, trump, every played card, captured
// points, friend calls and who has revealed, hand sizes. Private: your own
// hand, and the kitty when you're the one who picked it up.

import type { Bid } from '@/game/bidding';
import { buildDecks, countCards, type Card, type Level } from '@/game/cards';
import type { ResolvedRules, TractorConfig } from '@/game/config';
import { activeSeat, bidLevelFor, canPickUp, canRedeal, canRevealKitty, landlordToBe, validBidsFor } from '@/game/engine';
import { effectiveSuit, type EffSuit, type Trump } from '@/game/trump';
import type { TrickState } from '@/game/trick';
import type { CompletedTrick, Friend, Stage, TractorState } from '@/game/types';
import { totalPoints } from '@/game/cards';

export interface Observation {
  seat: number;
  numPlayers: number;
  config: TractorConfig;
  rules: ResolvedRules;
  stage: Stage;
  roundNumber: number;
  /** This seat's own cards. */
  hand: Card[];
  handSizes: number[];
  levels: Level[];
  landlord: number | null;
  landlordToBe: number | null;
  bidLevel: Level;
  trump: Trump | null;
  bids: Bid[];
  autobid: Bid | null;
  revealedKitty: Card[];
  /** Cards still to be dealt (live bidding). */
  deckRemaining: number;
  kittySize: number;
  /** The buried kitty / the dealt kitty — only for the landlord. */
  kitty: Card[] | null;
  dealtKitty: Card[] | null;
  friends: Friend[];
  /** Known members of the landlord's team. */
  landlordTeam: number[];
  trick: TrickState | null;
  tricks: CompletedTrick[];
  /** Points each seat has captured. */
  capturedPoints: number[];
  /** Throw penalties per seat. */
  penalties: number[];
  /** Cards taken out of the decks before the deal (public). */
  removed: Card[];
  /** Live bidding after the deal: seats that said they have no more bids. */
  bidDone: number[];
  dealer: number;
  /** Turn bidding: whose turn and how many passes in a row. */
  bidTurn: number | null;
  passes: number;
  /** Whose decision the round waits on. */
  active: number | null;
  /** What this seat may do right now. */
  can: {
    bid: Bid[];
    pass: boolean;
    pickUp: boolean;
    revealKitty: boolean;
    redeal: boolean;
    exchange: boolean;
    play: boolean;
  };
}

export function observe(state: TractorState, seat: number): Observation {
  const r = state.round;
  const isLandlord = r.landlord === seat;
  const active = activeSeat(state);
  return {
    seat,
    numPlayers: state.seats.length,
    config: state.config,
    rules: state.rules,
    stage: r.stage,
    roundNumber: r.number,
    hand: r.hands[seat]!.slice(),
    handSizes: r.hands.map((h) => h.length),
    levels: state.levels.slice(),
    landlord: state.landlord,
    landlordToBe: landlordToBe(state),
    bidLevel: bidLevelFor(state, seat),
    trump: r.trump,
    bids: r.bids.slice(),
    autobid: r.autobid,
    revealedKitty: r.dealtKitty.slice(0, r.revealed),
    deckRemaining: r.deck.length,
    kittySize: r.kittySize,
    kitty: isLandlord && (r.stage === 'play' || r.stage === 'roundOver') ? r.kitty.slice() : null,
    dealtKitty: isLandlord && r.stage !== 'draw' && r.stage !== 'bid' ? r.dealtKitty.slice() : null,
    friends: r.friends.map((f) => ({ ...f })),
    landlordTeam: r.landlordTeam.slice(),
    trick: r.trick,
    tricks: r.tricks,
    capturedPoints: r.captured.map(totalPoints),
    penalties: r.penalties.slice(),
    removed: r.removed.slice(),
    bidDone: r.bidDone.slice(),
    dealer: r.dealer,
    bidTurn: r.bidTurn,
    passes: r.passes,
    active,
    can: {
      bid: validBidsFor(state, seat),
      pass: r.stage === 'bid' && r.bidTurn === seat,
      pickUp: canPickUp(state, seat),
      revealKitty: canRevealKitty(state, seat),
      redeal: canRedeal(state) && active === seat,
      exchange: r.stage === 'exchange' && isLandlord,
      play: r.stage === 'play' && active === seat,
    },
  };
}

/** Cards this seat hasn't seen: every deck card not in its hand, not played, not buried by it. */
export function unseenCounts(obs: Observation): Map<Card, number> {
  const counts = countCards(buildDecks(obs.rules.numDecks));
  const take = (c: Card) => counts.set(c, Math.max(0, (counts.get(c) ?? 0) - 1));
  obs.hand.forEach(take);
  for (const t of obs.tricks) for (const p of t.plays) p.cards.forEach(take);
  if (obs.trick) for (const p of obs.trick.plays) p.cards.forEach(take);
  obs.kitty?.forEach(take);
  obs.revealedKitty.forEach(take);
  obs.removed.forEach(take);
  return counts;
}

/** Suits each seat has shown to be out of (failed to follow), from the tricks so far. */
export function knownVoids(obs: Observation): Array<Set<EffSuit>> {
  const voids = Array.from({ length: obs.numPlayers }, () => new Set<EffSuit>());
  const trump = obs.trump;
  if (!trump) return voids;
  const tricks = [...obs.tricks, ...(obs.trick && obs.trick.format ? [{ suit: obs.trick.format.suit, plays: obs.trick.plays }] : [])];
  for (const t of tricks) {
    for (const p of t.plays.slice(1)) {
      if (p.cards.some((c) => effectiveSuit(trump, c) !== t.suit)) voids[p.seat]!.add(t.suit);
    }
  }
  return voids;
}
