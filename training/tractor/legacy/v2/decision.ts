// FROZEN COPY of src/ai/nn (encoder v2, action space v1, tractor/cand-mlp@1) for benchmarking
// v2 policies (training/benchmarks/) against newer networks. Do not edit: a v2
// policy only plays as trained if this code stays exactly as it was.

// Tractor learned agent — the action space: every decision is a choice
// among a list of candidates, each with a fixed-size feature vector, which
// the network scores (a candidate-scoring policy). Tractor plays are
// combinatorial, so there's no fixed slot list; candidates come from the
// legal-play generator (src/ai/plays.ts).
//
//   bid   declaring trump: "no bid" (live deal) or "pass" (turns), then every legal bid
//   bury  the kitty, one card at a time (kittySize decisions)
//   call  Finding Friends: one friend call at a time (card × which copy)
//   play  a lead (single units, tuples, tractors, boss throws) or a follow
//         (legal matchings of the required shape, low / high / point fills)
//
// Steps with nothing to decide (taking the kitty, turning it over, redeal)
// come back as `forced`. ACTION_SPACE_VERSION goes in every policy file.

import { cardPoints, countCards, isJoker, removeCards, type Card, type Rank } from '@/game/cards';
import { formatFromCards, matchFormat, type TrickUnit } from '@/game/format';
import { effectiveSuit, type Trump } from '@/game/trump';
import type { Action, FriendCall } from '@/game/types';
import { unseenCounts, type Observation } from '@/ai/observation';
import { followCandidates, leadCandidates, tractorReq, unitIsBoss, winnerIfPlayed } from '@/ai/plays';
import { Layout, groupOf, slotTrump, unitKinds } from './encode';
import { CARD_SLOTS, GROUPS, sideSuits, slotOf } from './slots';

export const ACTION_SPACE_VERSION = 1;
export const MAX_CANDIDATES = 64;
/** Follow candidates generated per decision. */
export const FOLLOW_LIMIT = 24;
/** Ranks a friend may be called by (the useful ones), with up to this many copies. */
const CALL_RANKS: Rank[] = ['A', 'K', 'Q', 'J', 'T'];
const CALL_MAX_ORDINAL = 3;

export type DecisionKind = 'bid' | 'bury' | 'call' | 'play';
const KINDS = ['noBid', 'bid', 'bury', 'call', 'lead', 'follow'] as const;

const CAND = new Layout();
const C = {
  kind: CAND.add('kind', KINDS.length),
  cards: CAND.add('cards', CARD_SLOTS),
  nCards: CAND.add('nCards', 1),
  points: CAND.add('points', 1),
  allTrump: CAND.add('allTrump', 1),
  groups: CAND.add('groups', GROUPS),
  units: CAND.add('units', 4),
  isThrow: CAND.add('isThrow', 1),
  winsNow: CAND.add('winsNow', 1),
  boss: CAND.add('boss', 1),
  ruff: CAND.add('ruff', 1),
  bidCount: CAND.add('bidCount', 1),
  bidJoker: CAND.add('bidJoker', 1),
  bidTrumpShare: CAND.add('bidTrumpShare', 1),
  callOrdinal: CAND.add('callOrdinal', 1),
  callHeld: CAND.add('callHeld', 1),
  copies: CAND.add('copies', 1),
  groupShare: CAND.add('groupShare', 1),
};
export const CAND_DIM = CAND.dim;
export const CAND_LAYOUT = CAND;

export interface Draft {
  buried: Card[];
  calls: FriendCall[];
}

export const emptyDraft = (): Draft => ({ buried: [], calls: [] });

/** What choosing a candidate does: an action, one more part of the exchange, or (live deal) nothing for now. */
export type Pick = { action: Action } | { bury: Card } | { call: FriendCall } | { none: true };

export interface Candidate {
  pick: Pick;
  features: Float32Array;
}

export interface Decision {
  kind: DecisionKind;
  seat: number;
  candidates: Candidate[];
}

export type Step = { decision: Decision } | { forced: Action } | null;

/** The observation with an exchange in progress applied: buried cards out of the hand, calls made so far listed. */
export function applyDraft(obs: Observation, draft: Draft): Observation {
  if (draft.buried.length === 0 && draft.calls.length === 0) return obs;
  return {
    ...obs,
    hand: removeCards(obs.hand, draft.buried),
    kitty: draft.buried.slice(),
    friends: [...obs.friends, ...draft.calls.map((c) => ({ ...c, skip: c.ordinal - 1, seat: null }))],
  };
}

function base(kind: (typeof KINDS)[number]): Float32Array {
  const f = new Float32Array(CAND_DIM);
  f[C.kind + KINDS.indexOf(kind)] = 1;
  return f;
}

function cardFeatures(f: Float32Array, trump: Trump, cards: readonly Card[], decks: number): void {
  const w = 1 / decks;
  let pts = 0;
  let trumps = 0;
  for (const c of cards) {
    f[C.cards + slotOf(trump, c)]! += w;
    pts += cardPoints(c);
    if (effectiveSuit(trump, c) === 'T') trumps++;
    f[C.groups + groupOf(trump, c)]! += 1 / Math.max(1, cards.length);
  }
  f[C.nCards] = cards.length / 12;
  f[C.points] = pts / 50;
  f[C.allTrump] = cards.length > 0 && trumps === cards.length ? 1 : 0;
}

function unitFeatures(f: Float32Array, units: readonly TrickUnit[]): void {
  const k = unitKinds(units);
  f[C.units + 0] = k.singles / 6;
  f[C.units + 1] = k.pairs / 6;
  f[C.units + 2] = k.triples / 6;
  f[C.units + 3] = k.tractors / 3;
}

/** What `obs.seat` must decide now (with an exchange draft in progress), or a forced step, or null. */
export function nextStep(obs0: Observation, draft: Draft = emptyDraft()): Step {
  const obs = applyDraft(obs0, draft);
  const seat = obs.seat;
  const c = obs.can;
  const decks = obs.rules.numDecks;
  const trump = slotTrump(obs);

  if (c.exchange) {
    if (draft.buried.length < obs.kittySize) return { decision: { kind: 'bury', seat, candidates: buryCandidates(obs, trump, decks) } };
    const want = obs.rules.gameMode === 'findingFriends' ? obs.rules.numFriends : 0;
    if (draft.calls.length < want) {
      const candidates = callCandidates(obs, trump, decks, draft);
      if (candidates.length > 0) return { decision: { kind: 'call', seat, candidates } };
    }
    return { forced: { type: 'exchange', seat, bury: draft.buried.slice(), friends: draft.calls.slice(0, want) } };
  }
  if (c.play) return { decision: { kind: 'play', seat, candidates: playCandidates(obs, trump, decks) } };
  if (c.pickUp) return { forced: { type: 'pickUpKitty', seat } };
  if (c.revealKitty) return { forced: { type: 'revealKitty', seat } };
  if (c.redeal) return { forced: { type: 'redeal' } };
  if ((obs.stage === 'bid' && c.pass) || (obs.stage === 'draw' && c.bid.length > 0)) {
    const turns = obs.stage === 'bid';
    const none: Candidate = { pick: turns ? { action: { type: 'pass', seat } } : { none: true }, features: base('noBid') };
    const candidates = [none];
    for (const b of c.bid) {
      const f = base('bid');
      const cards = Array<Card>(b.count).fill(b.card);
      cardFeatures(f, trump, cards, decks);
      f[C.bidCount] = b.count / decks;
      f[C.bidJoker] = isJoker(b.card) ? 1 : 0;
      // How much of my hand would be trump with this declaration.
      const level = obs.bidLevel === 'NT' ? null : obs.bidLevel;
      const t: Trump = isJoker(b.card) || level === null ? { kind: 'noTrump', number: level } : { kind: 'standard', suit: b.card[1] as never, number: level };
      f[C.bidTrumpShare] = obs.hand.filter((x) => effectiveSuit(t, x) === 'T').length / Math.max(1, obs.hand.length);
      candidates.push({ pick: { action: { type: 'bid', seat, card: b.card, count: b.count } }, features: f });
    }
    return { decision: { kind: 'bid', seat, candidates: candidates.slice(0, MAX_CANDIDATES) } };
  }
  return null;
}

function buryCandidates(obs: Observation, trump: Trump, decks: number): Candidate[] {
  const counts = countCards(obs.hand);
  const groupLen = new Array<number>(GROUPS).fill(0);
  for (const c of obs.hand) groupLen[groupOf(trump, c)]!++;
  const out: Candidate[] = [];
  for (const [card, k] of counts) {
    const f = base('bury');
    cardFeatures(f, trump, [card], decks);
    f[C.copies] = k / decks;
    f[C.groupShare] = groupLen[groupOf(trump, card)]! / Math.max(1, obs.hand.length);
    out.push({ pick: { bury: card }, features: f });
  }
  // Stable, readable order: by slot.
  out.sort((a, b) => slotOf(trump, (a.pick as { bury: Card }).bury) - slotOf(trump, (b.pick as { bury: Card }).bury));
  return out.slice(0, MAX_CANDIDATES);
}

function callCandidates(obs: Observation, trump: Trump, decks: number, draft: Draft): Candidate[] {
  const policy = obs.config.friendSelectionPolicy;
  const counts = countCards(obs.hand);
  const out: Candidate[] = [];
  const suits = policy === 'trumpsIncluded' ? (['S', 'H', 'D', 'C'] as const) : sideSuits(trump);
  const landlordLevel = obs.levels[obs.seat];
  for (const rank of CALL_RANKS) {
    if (policy !== 'trumpsIncluded' && rank === trump.number) continue;
    if (policy === 'highestCardNotAllowed' && (rank === 'A' || (trump.number === 'A' && rank === 'K'))) continue;
    if (policy === 'pointCardNotAllowed' && cardPoints(rank + 'S') > 0 && !(landlordLevel === 'A' && rank === 'K')) continue;
    for (const suit of suits) {
      const card = rank + suit;
      for (let ordinal = 1; ordinal <= Math.min(decks, CALL_MAX_ORDINAL); ordinal++) {
        if (draft.calls.some((x) => x.card === card && x.ordinal === ordinal)) continue;
        const f = base('call');
        cardFeatures(f, trump, [card], decks);
        f[C.callOrdinal] = ordinal / decks;
        f[C.callHeld] = (counts.get(card) ?? 0) / decks;
        out.push({ pick: { call: { card, ordinal } }, features: f });
      }
    }
  }
  return out.slice(0, MAX_CANDIDATES);
}

function playCandidates(obs: Observation, trump: Trump, decks: number): Candidate[] {
  const trick = obs.trick!;
  const realTrump = obs.trump!;
  const seat = obs.seat;
  const unseen = unseenCounts(obs);
  const leading = !trick.format;
  const plays = leading ? leadCandidates(obs, unseen) : followCandidates(obs, trick.format!, FOLLOW_LIMIT);
  const out: Candidate[] = [];
  for (const cards of plays.slice(0, MAX_CANDIDATES)) {
    const f = base(leading ? 'lead' : 'follow');
    cardFeatures(f, trump, cards, decks);
    if (leading) {
      try {
        const units = formatFromCards(realTrump, tractorReq(obs), cards).units;
        unitFeatures(f, units);
        f[C.isThrow] = units.length > 1 ? 1 : 0;
        f[C.boss] = units.every((u) => unitIsBoss(realTrump, u, unseen)) ? 1 : 0;
      } catch {
        /* not a lead after all — leave unit features empty */
      }
      f[C.winsNow] = 1;
    } else {
      const m = matchFormat(realTrump, trick.format!, cards)[0];
      if (m) unitFeatures(f, m);
      f[C.winsNow] = winnerIfPlayed(trick, realTrump, seat, cards, obs) === seat ? 1 : 0;
      f[C.ruff] = trick.format!.suit !== 'T' && cards.every((c) => effectiveSuit(realTrump, c) === 'T') ? 1 : 0;
    }
    out.push({ pick: { action: { type: 'play', seat, cards } }, features: f });
  }
  return out;
}

/** Same multiset of cards. */
export function sameCards(a: readonly Card[], b: readonly Card[]): boolean {
  if (a.length !== b.length) return false;
  const x = a.slice().sort();
  const y = b.slice().sort();
  return x.every((c, i) => c === y[i]);
}
