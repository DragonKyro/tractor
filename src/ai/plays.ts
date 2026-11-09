// Legal-play candidates for a seat, built from its own hand only: leads
// (single units, plus "boss" throws) and follows (matches of the required
// shapes with a choice of fill). The learned AI scores these (src/ai/nn/decision.ts).

import { cardPoints, countCards, removeCards, type Card } from '@/game/cards';
import {
  findFormatMatches,
  findTractorsFromStart,
  formatSize,
  isLegalFollow,
  requiredShapes,
  unitCards,
  unitShape,
  type TractorRequirements,
  type TrickFormat,
  type TrickUnit,
} from '@/game/format';
import { computeWinner, type TrickState } from '@/game/trick';
import { compareCards, compareEffective, effectiveSuit, strength, type EffSuit, type Trump } from '@/game/trump';
import type { Observation } from './observation';

export function tractorReq(obs: Observation): TractorRequirements {
  return { minCount: obs.config.tractorMinCount, minLength: obs.config.tractorMinLength };
}

/** Rough "how much would I hate to spend this card" value. */
export function cardCost(trump: Trump, card: Card): number {
  const t = effectiveSuit(trump, card) === 'T';
  return (t ? 20 : 0) + strength(trump, card) + cardPoints(card) * 0.6;
}

function bySuit(trump: Trump, hand: readonly Card[]): Map<EffSuit, Card[]> {
  const m = new Map<EffSuit, Card[]>();
  for (const c of hand) {
    const s = effectiveSuit(trump, c);
    const list = m.get(s) ?? [];
    list.push(c);
    m.set(s, list);
  }
  for (const list of m.values()) list.sort((a, b) => compareCards(trump, a, b));
  return m;
}

function key(cards: readonly Card[]): string {
  return cards.slice().sort().join(',');
}

/** Single-unit leads: every single, tuple and tractor in the hand. */
export function leadUnits(obs: Observation): TrickUnit[] {
  const trump = obs.trump!;
  const out: TrickUnit[] = [];
  for (const cards of bySuit(trump, obs.hand).values()) {
    const counts = countCards(cards);
    for (const [card, n] of counts) {
      out.push({ kind: 'repeated', card, count: 1 });
      for (let k = 2; k <= n; k++) out.push({ kind: 'repeated', card, count: k });
      for (const t of findTractorsFromStart(trump, card, n, counts, tractorReq(obs))) out.push(t);
    }
  }
  return out;
}

/** Can anyone (from the cards this seat hasn't seen) beat this unit in its own suit? */
export function unitIsBoss(trump: Trump, unit: TrickUnit, unseen: Map<Card, number>): boolean {
  const suit = effectiveSuit(trump, unitCards(unit)[0]!);
  if (unit.kind === 'repeated') {
    for (const [c, n] of unseen) {
      if (n >= unit.count && effectiveSuit(trump, c) === suit && compareEffective(trump, c, unit.card) > 0) return false;
    }
    return true;
  }
  const inSuit = new Map([...unseen].filter(([c, n]) => n > 0 && effectiveSuit(trump, c) === suit));
  for (const [c, n] of inSuit) {
    if (compareEffective(trump, c, unit.members[0]!) <= 0) continue;
    if (findTractorsFromStart(trump, c, n, inSuit, { minCount: unit.count, minLength: unit.members.length }).length) return false;
  }
  return true;
}

/** Legal leads to consider: single units, and per suit a throw of all its boss units. */
export function leadCandidates(obs: Observation, unseen: Map<Card, number>): Card[][] {
  const trump = obs.trump!;
  const units = leadUnits(obs);
  const out = new Map<string, Card[]>();
  for (const u of units) {
    const cards = unitCards(u);
    out.set(key(cards), cards);
  }
  // Boss throws: combine the unbeatable units of a suit (largest first, disjoint).
  const bySuitUnits = new Map<EffSuit, TrickUnit[]>();
  for (const u of units) {
    if (!unitIsBoss(trump, u, unseen)) continue;
    const s = effectiveSuit(trump, unitCards(u)[0]!);
    bySuitUnits.set(s, [...(bySuitUnits.get(s) ?? []), u]);
  }
  for (const list of bySuitUnits.values()) {
    list.sort((a, b) => unitCards(b).length - unitCards(a).length);
    let pool = obs.hand.slice();
    const picked: Card[] = [];
    let parts = 0;
    for (const u of list) {
      const cards = unitCards(u);
      try {
        pool = removeCards(pool, cards);
      } catch {
        continue;
      }
      picked.push(...cards);
      parts++;
    }
    if (parts >= 2) out.set(key(picked), picked);
  }
  return [...out.values()];
}

/**
 * Legal follows to `format`: several matchings of the required shapes, each
 * with low / high / point fills. With `wide` (the learned AI's action space)
 * also, wherever the rules leave a choice: every distinct card when one card
 * is free to choose (any discard when void, any card of the suit when
 * following a single), a cheapest fill taken from each suit first, and a fill
 * that keeps pairs. The narrow list always comes first and is unchanged, so
 * the frozen v2 benchmark (training/tractor/legacy/v2) sees exactly what it always did.
 */
export function followCandidates(obs: Observation, format: TrickFormat, limit = 24, opts: { wide?: boolean } = {}): Card[][] {
  const trump = obs.trump!;
  const hand = obs.hand;
  const need = formatSize(format);
  const inSuit = hand.filter((c) => effectiveSuit(trump, c) === format.suit).sort((a, b) => compareCards(trump, a, b));
  const others = hand.filter((c) => effectiveSuit(trump, c) !== format.suit);
  const out = new Map<string, Card[]>();
  const add = (cards: Card[]) => {
    if (cards.length === need) out.set(key(cards), cards);
  };
  const cost = (a: Card, b: Card) => cardCost(trump, a) - cardCost(trump, b);

  const narrowFills = (pool: Card[], n: number): Card[][] => {
    if (n <= 0) return [[]];
    const low = pool.slice().sort(cost);
    const pts = pool.slice().sort((a, b) => cardPoints(b) - cardPoints(a) || cost(a, b));
    const high = pool.slice().sort((a, b) => cost(b, a));
    return [low.slice(0, n), pts.slice(0, n), high.slice(0, n)];
  };
  const wideFills = (pool: Card[], n: number): Card[][] => {
    if (n <= 0) return [[]];
    if (n === 1) return [...new Set(pool)].sort(cost).map((c) => [c]);
    const low = pool.slice().sort(cost);
    const res: Card[][] = [];
    const groups = bySuit(trump, pool);
    for (const cards of groups.values()) {
      const first = cards.slice().sort(cost).slice(0, n);
      let rest = pool.slice();
      try {
        rest = removeCards(rest, first);
      } catch {
        continue;
      }
      res.push([...first, ...rest.sort(cost).slice(0, n - first.length)]);
    }
    const counts = countCards(pool);
    const loose = (c: Card) => ((counts.get(c) ?? 0) >= 2 ? 1 : 0);
    res.push(low.slice().sort((a, b) => loose(a) - loose(b) || cost(a, b)).slice(0, n));
    return res;
  };

  const generate = (fills: (pool: Card[], n: number) => Card[][]) => {
    if (inSuit.length <= need) {
      const rest = need - inSuit.length;
      for (const f of fills(others, rest)) add([...inSuit, ...f]);
      // Ruffing: when void, trump in the lead's shape can win.
      if (inSuit.length === 0 && format.suit !== 'T') {
        const trumps = countCards(others.filter((c) => effectiveSuit(trump, c) === 'T'));
        const shapes = format.units.map(unitShape);
        for (const m of findFormatMatches(trump, trumps, shapes, { limit: 8 })) {
          add(m.flatMap((p) => p.flatMap(([c, k]) => Array<Card>(k).fill(c))));
        }
      }
      return;
    }
    const req = requiredShapes(trump, format, hand, obs.config.trickDrawPolicy);
    if (!req) {
      for (const f of fills(inSuit, need)) add(f);
      return;
    }
    const counts = countCards(inSuit);
    const filter =
      obs.config.trickDrawPolicy === 'longerTuplesProtected' || obs.config.trickDrawPolicy === 'longerTuplesProtectedAndOnlyDrawTractorOnTractor'
        ? (c: Card, k: number) => (counts.get(c) ?? 0) <= k
        : undefined;
    const matches = findFormatMatches(trump, counts, req, { filter, limit: 40 });
    // Highest-first comes first; also try the last (lowest) few.
    const picks = [...matches.slice(0, 4), ...matches.slice(-4)];
    for (const m of picks) {
      const core = m.flatMap((p) => p.flatMap(([c, k]) => Array<Card>(k).fill(c)));
      let rest: Card[];
      try {
        rest = removeCards(inSuit, core);
      } catch {
        continue;
      }
      for (const f of fills(rest, need - core.length)) add([...core, ...f]);
    }
  };

  generate(narrowFills);
  if (opts.wide) generate(wideFills);
  const legal = [...out.values()].filter((cards) => isLegalFollow(trump, format, hand, cards, obs.config.trickDrawPolicy));
  if (legal.length > 0) return legal.slice(0, limit);
  return [bruteForceFollow(obs, format, inSuit, others, need)];
}

/** Last resort: search in-suit subsets for a legal follow (should be unreachable). */
function bruteForceFollow(obs: Observation, format: TrickFormat, inSuit: Card[], others: Card[], need: number): Card[] {
  const trump = obs.trump!;
  if (inSuit.length <= need) return [...inSuit, ...others.slice(0, need - inSuit.length)];
  const n = inSuit.length;
  let found: Card[] | null = null;
  const pick: Card[] = [];
  let budget = 20000;
  const rec = (i: number) => {
    if (found || budget-- <= 0) return;
    if (pick.length === need) {
      if (isLegalFollow(trump, format, obs.hand, pick, obs.config.trickDrawPolicy)) found = pick.slice();
      return;
    }
    if (n - i < need - pick.length) return;
    pick.push(inSuit[i]!);
    rec(i + 1);
    pick.pop();
    rec(i + 1);
  };
  rec(0);
  return found ?? inSuit.slice(0, need);
}

/** Who would be winning if `seat` played `cards` into the trick now. */
export function winnerIfPlayed(trick: TrickState, trump: Trump, seat: number, cards: Card[], obs: Observation): number | null {
  if (!trick.format) return seat;
  const next: TrickState = { ...trick, plays: [...trick.plays, { seat, cards, badThrowCards: [], betterPlayer: null }] };
  return computeWinner(next, trump, obs.config.throwEvaluationPolicy);
}
