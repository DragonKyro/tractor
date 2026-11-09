// Tractor learned agent — the action space: every decision is a choice
// among a list of candidates, each with a fixed-size feature vector, which
// the network scores (a candidate-scoring policy). Tractor plays are
// combinatorial, so there's no fixed slot list; candidates come from the
// legal-play generator (src/ai/plays.ts).
//
//   bid   declaring trump: "no bid" (live deal) or "pass" (turns), then every legal bid
//   bury  the kitty, one card at a time (kittySize decisions)
//   call  Finding Friends: one friend call at a time (card × which copy)
//   play  a lead (single units, tuples, tractors, boss throws and their
//         variants: part of the boss units, or one near-boss card added) or a follow:
//         the legal matchings of the required shape with every choice the
//         rules leave open (any discard when void, any card of the suit when
//         following a single, per-suit and pair-keeping fills; plays.ts)
//
// Steps with nothing to decide (taking the kitty, turning it over, redeal)
// come back as `forced`. ACTION_SPACE_VERSION goes in every policy file.

import { cardPoints, countCards, isJoker, removeCards, totalPoints, type Card, type Rank } from '@/game/cards';
import { formatFromCards, formatSize, matchFormat, unitCards, unitSize, type TrickUnit } from '@/game/format';
import { effectiveSuit, type Trump } from '@/game/trump';
import type { Action, FriendCall } from '@/game/types';
import { knownVoids, unseenCounts, type Observation } from '../observation';
import { followCandidates, leadCandidates, leadUnits, tractorReq, unitIsBoss, unseenAbove, winnerIfPlayed } from '../plays';
import { Layout, groupOf, knownSides, slotTrump, unitKinds } from './encode';
import { CARD_SLOTS, GROUPS, sideSuits, slotOf } from './slots';

export const ACTION_SPACE_VERSION = 3;
export const MAX_CANDIDATES = 64;
/** Follow candidates generated per decision. */
export const FOLLOW_LIMIT = MAX_CANDIDATES;
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
  // what the cards cost me / do for the side (plays and burying)
  discardShare: CAND.add('discardShare', 1),
  voidsAfter: CAND.add('voidsAfter', GROUPS),
  pairsBroken: CAND.add('pairsBroken', 1),
  rankHigh: CAND.add('rankHigh', 1),
  rankLow: CAND.add('rankLow', 1),
  groupLenBefore: CAND.add('groupLenBefore', 1),
  pointsToUs: CAND.add('pointsToUs', 1),
  pointsToThem: CAND.add('pointsToThem', 1),
  partnerVoidLed: CAND.add('partnerVoidLed', 1),
  opponentVoidLed: CAND.add('opponentVoidLed', 1),
  // v4 (appended to v3's)
  killRisk: CAND.add('killRisk', 1),
  ruffRisk: CAND.add('ruffRisk', 1),
  trumpsLeftAfter: CAND.add('trumpsLeftAfter', 1),
  bossLeftAfter: CAND.add('bossLeftAfter', 1),
  drawsTrump: CAND.add('drawsTrump', 1),
  lastTrick: CAND.add('lastTrick', 1),
  lastTrickMultiplier: CAND.add('lastTrickMultiplier', 1),
  bossHeldBack: CAND.add('bossHeldBack', 1),
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

/** Rank within its group, 0 (lowest) .. 1 (highest), by slot order. */
function rankInGroup(trump: Trump, card: Card): number {
  const s = slotOf(trump, card);
  if (s < 52) return (s % 13) / 12;
  return (s - 52) / (CARD_SLOTS - 1 - 52);
}

/** What taking `cards` out of my hand does to it: voids made, pairs broken, how high / low they are. */
function handFeatures(f: Float32Array, trump: Trump, hand: readonly Card[], cards: readonly Card[]): void {
  const before = new Array<number>(GROUPS).fill(0);
  const taken = new Array<number>(GROUPS).fill(0);
  for (const c of hand) before[groupOf(trump, c)]!++;
  for (const c of cards) taken[groupOf(trump, c)]!++;
  for (let g = 0; g < GROUPS; g++) if (before[g]! > 0 && before[g] === taken[g]) f[C.voidsAfter + g] = 1;
  const have = countCards(hand);
  const use = countCards(cards);
  let broken = 0;
  for (const [c, k] of use) if ((have.get(c) ?? 0) >= 2 && (have.get(c) ?? 0) - k === 1) broken++;
  f[C.pairsBroken] = broken / 3;
  let hi = 0;
  let lo = 1;
  for (const c of cards) {
    const r = rankInGroup(trump, c);
    hi = Math.max(hi, r);
    lo = Math.min(lo, r);
  }
  f[C.rankHigh] = cards.length ? hi : 0;
  f[C.rankLow] = cards.length ? lo : 0;
  const g0 = cards.length ? groupOf(trump, cards[0]!) : 0;
  f[C.groupLenBefore] = before[g0]! / Math.max(1, hand.length);
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
    handFeatures(f, trump, obs.hand, [card]);
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
  const sides = knownSides(obs);
  const mine = sides[seat]!;
  const voids = knownVoids(obs);
  const trickPoints = totalPoints(trick.plays.flatMap((p) => p.cards));
  // v4: who is still to play, and what they're known to lack.
  const after = leading ? trick.order.slice(1) : trick.order.slice(trick.plays.length + 1);
  const oppsAfter = after.filter((s) => mine !== '?' && sides[s] !== '?' && sides[s] !== mine);
  const allOpps = obs.landlord === null ? [] : [...Array(obs.numPlayers).keys()].filter((s) => s !== seat && mine !== '?' && sides[s] !== '?' && sides[s] !== mine);
  // Computed once per decision: boss cards in hand, boss units per suit.
  const bossCards = countCards(obs.hand.filter((c) => unseenAbove(realTrump, c, unseen) === 0));
  const bossTotal = [...bossCards.values()].reduce((a, k) => a + k, 0);
  const bossUnitsBySuit = new Map<string, number>();
  if (leading) {
    for (const u of leadUnits(obs)) {
      if (!unitIsBoss(realTrump, u, unseen)) continue;
      const su = effectiveSuit(realTrump, unitCards(u)[0]!);
      bossUnitsBySuit.set(su, (bossUnitsBySuit.get(su) ?? 0) + 1);
    }
  }
  const lastTrick = obs.hand.length > 0 && obs.hand.length <= (leading ? obs.hand.length : formatSize(trick.format!));
  const out: Candidate[] = [];
  for (const cards of plays.slice(0, MAX_CANDIDATES)) {
    const f = base(leading ? 'lead' : 'follow');
    cardFeatures(f, trump, cards, decks);
    handFeatures(f, trump, obs.hand, cards);
    if (leading) {
      // Who can't follow this suit: a void partner can ruff for us, a void opponent against us.
      const suit = effectiveSuit(realTrump, cards[0]!);
      let partners = 0;
      let opponents = 0;
      let opp = 0;
      for (let s = 0; s < obs.numPlayers; s++) {
        if (s === seat || mine === '?' || sides[s] === '?') continue;
        if (sides[s] === mine) partners += voids[s]!.has(suit) ? 1 : 0;
        else {
          opp++;
          opponents += voids[s]!.has(suit) ? 1 : 0;
        }
      }
      f[C.partnerVoidLed] = partners > 0 ? 1 : 0;
      f[C.opponentVoidLed] = opp > 0 ? opponents / opp : 0;
    }
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
      const winner = winnerIfPlayed(trick, realTrump, seat, cards, obs);
      f[C.winsNow] = winner === seat ? 1 : 0;
      f[C.ruff] = trick.format!.suit !== 'T' && cards.every((c) => effectiveSuit(realTrump, c) === 'T') ? 1 : 0;
      const led = trick.format!.suit;
      f[C.discardShare] = cards.filter((c) => {
        const es = effectiveSuit(realTrump, c);
        return es !== led && es !== 'T';
      }).length / Math.max(1, cards.length);
      // Where the trick's points go if it ended now.
      const pts = (trickPoints + totalPoints(cards)) / 50;
      if (winner !== null && mine !== '?') {
        if (winner === seat || sides[winner] === mine) f[C.pointsToUs] = pts;
        else if (sides[winner] !== '?') f[C.pointsToThem] = pts;
      }
    }
    // v4: the risk of being beaten, what control the play leaves, and the final trick.
    const units = leading ? (() => {
      try {
        return formatFromCards(realTrump, tractorReq(obs), cards).units;
      } catch {
        return [];
      }
    })() : (matchFormat(realTrump, trick.format!, cards)[0] ?? []);
    const suit = effectiveSuit(realTrump, cards[0]!);
    if (leading || f[C.winsNow]) {
      let beaters = 0;
      for (const u of units) {
        // Unseen cards (with enough copies) above the unit's top card: who could kill it.
        const uc = unitCards(u);
        beaters = Math.max(beaters, unseenAbove(realTrump, uc[uc.length - 1]!, unseen, u.count));
      }
      f[C.killRisk] = Math.min(1, beaters / 4);
      const against = leading ? allOpps : oppsAfter;
      const ledSuit = leading ? suit : trick.format!.suit;
      if (ledSuit !== 'T' && against.length) {
        f[C.ruffRisk] = against.filter((s) => voids[s]!.has(ledSuit) && !voids[s]!.has('T')).length / against.length;
      }
    }
    const left = removeCards(obs.hand, cards);
    f[C.trumpsLeftAfter] = left.filter((c) => effectiveSuit(realTrump, c) === 'T').length / Math.max(1, obs.hand.length);
    const used = countCards(cards);
    let bossUsed = 0;
    for (const [c, k] of used) bossUsed += Math.min(k, bossCards.get(c) ?? 0);
    f[C.bossLeftAfter] = Math.min(1, (bossTotal - bossUsed) / 6);
    if (leading && suit === 'T' && allOpps.length) f[C.drawsTrump] = allOpps.filter((s) => !voids[s]!.has('T')).length / allOpps.length;
    if (lastTrick) {
      f[C.lastTrick] = 1;
      const largest = units.reduce((m, u) => Math.max(m, unitSize(u)), 0);
      f[C.lastTrickMultiplier] = Math.min(1, (obs.config.kittyPenalty === 'power' ? 2 ** largest : 2 * largest) / 16);
    }
    if (leading) {
      const bossInSuit = bossUnitsBySuit.get(suit) ?? 0;
      const bossPlayed = units.filter((u) => unitIsBoss(realTrump, u, unseen)).length;
      f[C.bossHeldBack] = Math.min(1, Math.max(0, bossInSuit - bossPlayed) / 4);
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
