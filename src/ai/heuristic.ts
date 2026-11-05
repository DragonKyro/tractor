// Hand-written Tractor AI. Decides from an Observation only (no peeking).
// Levels: 'easy' plays a random sensible-ish legal move; 'normal' plays the
// heuristics below. Trained levels slot in beside these (training/README.md).

import { cardPoints, cardRank, cardSuit, countCards, isJoker, SUITS, type Card, type Rank, type Suit } from '@/game/cards';
import { formatFromCards } from '@/game/format';
import { compareCards, effectiveSuit, strength, type EffSuit } from '@/game/trump';
import type { Action, FriendCall } from '@/game/types';
import type { Bid } from '@/game/bidding';
import { isPartner, knownVoids, unseenCounts, type Observation } from './observation';
import { cardCost, followCandidates, leadCandidates, playCost, tractorReq, unitIsBoss, winnerIfPlayed } from './plays';

export type HeuristicLevel = 'easy' | 'normal';

/** Deterministic per-decision randomness (the AI must not use Math.random: replays must agree). */
function hashRandom(obs: Observation, salt: number): () => number {
  let h = 2166136261 ^ salt;
  const feed = (s: string) => {
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  };
  feed(obs.hand.join(''));
  feed(String(obs.seat));
  feed(String(obs.tricks.length));
  feed(String(obs.bids.length));
  feed(String(obs.deckRemaining));
  return () => {
    h = Math.imul(h ^ (h >>> 15), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
}

export function chooseHeuristicAction(obs: Observation, level: HeuristicLevel): Action | null {
  const seat = obs.seat;
  const rnd = hashRandom(obs, level === 'easy' ? 7 : 3);
  const c = obs.can;
  if (c.exchange) return exchange(obs, level, rnd);
  if (c.play) return { type: 'play', seat, cards: choosePlay(obs, level, rnd) };
  if (c.pickUp) return { type: 'pickUpKitty', seat };
  if (c.revealKitty) return { type: 'revealKitty', seat };
  if (c.redeal) return { type: 'redeal' };
  if (obs.stage === 'bid' && c.pass) {
    const bid = chooseBid(obs, level, rnd, true);
    return bid ? { type: 'bid', seat, card: bid.card, count: bid.count } : { type: 'pass', seat };
  }
  if (obs.stage === 'draw' && c.bid.length > 0) {
    const bid = chooseBid(obs, level, rnd, false);
    return bid ? { type: 'bid', seat, card: bid.card, count: bid.count } : null;
  }
  return null;
}

// ------------------------------------------------------------------ bidding

/** How many cards would be trump if `suit` (or null = no-trump) were declared at the bid level. */
function trumpCount(obs: Observation, suit: Suit | null): number {
  const number = obs.bidLevel === 'NT' ? null : obs.bidLevel;
  return obs.hand.filter((c) => isJoker(c) || cardRank(c) === number || (suit !== null && cardSuit(c) === suit)).length;
}

function chooseBid(obs: Observation, level: HeuristicLevel, rnd: () => number, turns: boolean): Bid | null {
  const options = obs.can.bid;
  if (options.length === 0) return null;
  const last = obs.bids[obs.bids.length - 1];
  // Don't overcall a known partner unless it's our own bid getting reinforced.
  if (last && last.seat !== obs.seat && obs.landlord !== null && isPartner(obs, last.seat) && level === 'normal') return null;
  const totalDeal = Math.max(1, obs.hand.length + Math.floor(obs.deckRemaining / obs.numPlayers));
  const dealt = obs.hand.length / totalDeal;
  let best: { bid: Bid; score: number } | null = null;
  for (const bid of options) {
    const suit = isJoker(bid.card) ? null : cardSuit(bid.card);
    const tc = trumpCount(obs, suit);
    // Expected share of trump if dealt evenly is ~ (13/54+…) — normalize by hand size.
    let score = tc / Math.max(1, obs.hand.length);
    if (suit === null) score = obs.hand.filter((c) => cardRank(c) === 'A' || isJoker(c)).length / Math.max(1, obs.hand.length) + 0.12;
    score += (bid.count - 1) * 0.05;
    if (!best || score > best.score) best = { bid, score };
  }
  if (!best) return null;
  if (level === 'easy') return rnd() < 0.35 ? options[Math.floor(rnd() * options.length)]! : null;
  const reinforcing = last?.seat === obs.seat;
  if (reinforcing) {
    // Only reinforce to fend off a likely overcall late in the deal.
    return dealt > 0.8 && best.bid.count > last!.count && rnd() < 0.5 ? best.bid : null;
  }
  const threshold = turns ? 0.3 : dealt < 0.35 ? 0.42 : dealt < 0.7 ? 0.34 : 0.3;
  return best.score >= threshold ? best.bid : null;
}

// ------------------------------------------------------------------ exchange

function exchange(obs: Observation, level: HeuristicLevel, rnd: () => number): Action {
  const trump = obs.trump!;
  const hand = obs.hand.slice();
  const counts = countCards(hand);
  const suitLen = new Map<EffSuit, number>();
  for (const c of hand) suitLen.set(effectiveSuit(trump, c), (suitLen.get(effectiveSuit(trump, c)) ?? 0) + 1);
  const keep = (c: Card) => {
    const s = effectiveSuit(trump, c);
    if (s === 'T') return 200 + strength(trump, c);
    let v = strength(trump, c);
    if ((counts.get(c) ?? 0) >= 2) v += 18;
    if (cardRank(c) === 'A') v += 12;
    v += cardPoints(c) * 1.2; // points in the kitty are a liability
    v += (suitLen.get(s) ?? 0) * 1.5; // void short suits
    if (level === 'easy') v += rnd() * 10;
    return v;
  };
  const sorted = hand.slice().sort((a, b) => keep(a) - keep(b));
  const bury = sorted.slice(0, obs.kittySize);
  const after = hand.slice();
  for (const c of bury) after.splice(after.indexOf(c), 1);
  return { type: 'exchange', seat: obs.seat, bury, friends: callFriends(obs, after) };
}

function callFriends(obs: Observation, hand: Card[]): FriendCall[] {
  const n = obs.rules.numFriends;
  if (obs.rules.gameMode !== 'findingFriends' || n === 0) return [];
  const trump = obs.trump!;
  const policy = obs.config.friendSelectionPolicy;
  const decks = obs.rules.numDecks;
  const ranks: Rank[] =
    policy === 'highestCardNotAllowed' ? ['K', 'Q', 'J'] : policy === 'pointCardNotAllowed' ? ['A', 'Q', 'J'] : ['A', 'K', 'Q'];
  const calls: FriendCall[] = [];
  const counts = countCards(hand);
  for (const rank of ranks) {
    if (rank === trump.number) continue;
    const suits = SUITS.filter((s) => !(trump.kind === 'standard' && trump.suit === s))
      .map((s) => ({ s, held: counts.get(rank + s) ?? 0 }))
      .filter((x) => x.held < decks)
      .sort((a, b) => a.held - b.held);
    for (const { s, held } of suits) {
      if (calls.length >= n) break;
      calls.push({ card: rank + s, ordinal: held + 1 });
    }
    if (calls.length >= n) break;
  }
  // Out of good calls (tiny decks): fall back to anything legal-looking.
  for (const s of SUITS) {
    for (const rank of ['J', 'T', '9', '8', '7', '6'] as Rank[]) {
      if (calls.length >= n) break;
      if (rank === trump.number || (trump.kind === 'standard' && trump.suit === s)) continue;
      const card = rank + s;
      if (calls.some((x) => x.card === card)) continue;
      const held = counts.get(card) ?? 0;
      if (held < decks) calls.push({ card, ordinal: held + 1 });
    }
  }
  return calls.slice(0, n);
}

// ------------------------------------------------------------------ play

function choosePlay(obs: Observation, level: HeuristicLevel, rnd: () => number): Card[] {
  const trick = obs.trick!;
  const unseen = unseenCounts(obs);
  if (!trick.format) {
    const leads = leadCandidates(obs, unseen);
    if (level === 'easy') return leads[Math.floor(rnd() * leads.length)]!;
    return chooseLead(obs, leads, unseen);
  }
  const options = followCandidates(obs, trick.format);
  if (level === 'easy' || options.length === 1) return options[Math.floor(rnd() * options.length)]!;
  return chooseFollow(obs, options, unseen);
}

function chooseLead(obs: Observation, leads: Card[][], unseen: Map<Card, number>): Card[] {
  const trump = obs.trump!;
  const voids = knownVoids(obs);
  const opponents = Array.from({ length: obs.numPlayers }, (_, i) => i).filter((i) => !isPartner(obs, i));
  const trumpLeft = [...unseen].filter(([c, n]) => n > 0 && effectiveSuit(trump, c) === 'T').reduce((a, [, n]) => a + n, 0);
  const myTrump = obs.hand.filter((c) => effectiveSuit(trump, c) === 'T').length;
  let best: { cards: Card[]; score: number } | null = null;
  for (const cards of leads) {
    const suit = effectiveSuit(trump, cards[0]!);
    let units;
    try {
      units = formatFromCards(trump, tractorReq(obs), cards).units;
    } catch {
      continue;
    }
    const boss = units.every((u) => unitIsBoss(trump, u, unseen));
    const ruffRisk = suit !== 'T' && opponents.some((o) => voids[o]!.has(suit)) && trumpLeft > 0;
    const pts = cards.reduce((n, c) => n + cardPoints(c), 0);
    let score = 0;
    if (boss && !ruffRisk) score += 40 + cards.length * 6 + pts;
    else if (boss) score += 8 + cards.length;
    else score -= pts * 1.5 + cardCost(trump, cards[cards.length - 1]!) * 0.5;
    if (units.some((u) => u.kind === 'tractor')) score += boss ? 10 : 2;
    if (suit === 'T') score += myTrump > trumpLeft ? 6 : -6;
    // Probe with low singles of a long side suit when nothing is safe.
    if (!boss && cards.length === 1 && suit !== 'T') score += obs.hand.filter((c) => effectiveSuit(trump, c) === suit).length * 0.6;
    if (!best || score > best.score) best = { cards, score };
  }
  return best?.cards ?? leads[0]!;
}

function chooseFollow(obs: Observation, options: Card[][], unseen: Map<Card, number>): Card[] {
  const trump = obs.trump!;
  const trick = obs.trick!;
  const seat = obs.seat;
  const current = trick.winner;
  const remainingAfterMe = trick.order.length - trick.plays.length - 1;
  const partnerWinning = current !== null && current !== seat && isPartner(obs, current);
  const trickPoints = trick.plays.reduce((n, p) => n + p.cards.reduce((m, c) => m + cardPoints(c), 0), 0);
  const winnerPlay = trick.plays.find((p) => p.seat === current);
  // Is the partner's winning play safe from the players still to come?
  const partnerSafe =
    partnerWinning
    && (remainingAfterMe === 0
      || (trick.format!.units.length === 1 && winnerPlay !== undefined && winnerPlay.cards.every((c) => effectiveSuit(trump, c) === effectiveSuit(trump, winnerPlay.cards[0]!))
        && unitIsBoss(trump, { kind: 'repeated', card: winnerPlay.cards[winnerPlay.cards.length - 1]!, count: 1 }, unseen)));

  type Scored = { cards: Card[]; score: number };
  let best: Scored | null = null;
  for (const cards of options) {
    const winner = winnerIfPlayed(trick, trump, seat, cards, obs);
    const iWin = winner === seat;
    const pts = cards.reduce((n, c) => n + cardPoints(c), 0);
    const cost = playCost(trump, cards);
    let score: number;
    if (partnerWinning) {
      // Help the partner: don't overtake them; feed points when it's safe.
      score = (iWin ? -15 : 0) + (partnerSafe ? pts * 2 : -pts * 0.5) - cost * 0.3;
    } else if (iWin) {
      const lastToPlay = remainingAfterMe === 0;
      const stake = trickPoints + pts;
      score = 10 + stake * 1.5 - cost * (lastToPlay ? 0.6 : 0.25);
      if (!lastToPlay) {
        // Will it hold? Prefer winning with something unbeatable when points are at stake.
        const top = cards.slice().sort((a, b) => compareCards(trump, a, b)).pop()!;
        const holds = unitIsBoss(trump, { kind: 'repeated', card: top, count: 1 }, unseen);
        score += holds ? stake * 0.8 : -pts * 1.2;
      }
    } else {
      score = -pts * 2.5 - cost * 0.5;
    }
    if (!best || score > best.score) best = { cards, score };
  }
  return best?.cards ?? options[0]!;
}
