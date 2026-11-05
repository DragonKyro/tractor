// Trick formats: how a lead breaks into units (singles, tuples, tractors),
// whether a follow is legal, and how plays match a format — a port of
// shengji's trick.rs / ordered_card.rs / format_match.rs (no bombs or
// rainbows yet; see the README checklist).
//
// Vocabulary:
//   unit        — one component of a lead: `repeated` (n copies of one card:
//                 single, pair, triple…) or `tractor` (consecutive tuples of
//                 equal width, e.g. 7♥7♥8♥8♥).
//   shape       — a unit with the cards forgotten: its adjacent tuple sizes,
//                 [1] single, [2] pair, [2,2] tractor of pairs, [3,3,3] …
//   requirement — a list of shapes a follow is checked against. A follower
//                 must match the lead's own shapes if their hand can; failing
//                 that, the next requirement in the decomposition order
//                 (a tractor of pairs falls back to two pairs, then a pair +
//                 two singles, …), and so on. See `decomposition`.

import { countCards, type Card } from './cards';
import { compareCards, compareEffective, effectiveSuit, successor, type EffSuit, type Trump } from './trump';

export type TrickUnit =
  | { kind: 'repeated'; count: number; card: Card }
  | { kind: 'tractor'; count: number; members: Card[] };

export interface TractorRequirements {
  /** Minimum width of each tuple in a tractor (2 = pairs). */
  minCount: number;
  /** Minimum number of consecutive tuples. */
  minLength: number;
}

export const DEFAULT_TRACTOR_REQUIREMENTS: TractorRequirements = { minCount: 2, minLength: 2 };

export type TrickDrawPolicy =
  /** Followers must match the lead's shape as far as their hand allows, breaking longer tuples if needed. */
  | 'noProtections'
  /** Don't force a longer tuple (a triple) to be broken to follow a shorter one (a pair). */
  | 'longerTuplesProtected'
  /** Only a tractor lead can force a tractor to be drawn out. */
  | 'onlyDrawTractorOnTractor'
  | 'longerTuplesProtectedAndOnlyDrawTractorOnTractor'
  /** Followers only have to follow suit, never shape. */
  | 'noFormatBasedDraw';

export type ThrowEvaluationPolicy =
  /** A throw is beaten only when every unit is beaten. */
  | 'all'
  /** A throw is beaten by a matching play whose highest card is higher. */
  | 'highest'
  /** Compare the longest units first (singles ignored when longer units exist). */
  | 'trickUnitLength';

export interface TrickFormat {
  suit: EffSuit;
  /** Sorted by (size, first card) ascending. */
  units: TrickUnit[];
}

export type CardCounts = Map<Card, number>;
/** One unit's assignment in a match: (card, copies) per tuple, low card first for tractors. */
export type Placement = Array<[Card, number]>;
export type Shape = number[];

// ---------------------------------------------------------------- units

export function unitSize(u: TrickUnit): number {
  return u.kind === 'repeated' ? u.count : u.count * u.members.length;
}

export function unitFirst(u: TrickUnit): Card {
  return u.kind === 'repeated' ? u.card : u.members[0]!;
}

export function unitLast(u: TrickUnit): Card {
  return u.kind === 'repeated' ? u.card : u.members[u.members.length - 1]!;
}

export function unitCards(u: TrickUnit): Card[] {
  const out: Card[] = [];
  if (u.kind === 'repeated') {
    for (let i = 0; i < u.count; i++) out.push(u.card);
  } else {
    for (const m of u.members) for (let i = 0; i < u.count; i++) out.push(m);
  }
  return out;
}

export function unitShape(u: TrickUnit): Shape {
  return u.kind === 'repeated' ? [u.count] : u.members.map(() => u.count);
}

export function formatSize(f: TrickFormat): number {
  return f.units.reduce((n, u) => n + unitSize(u), 0);
}

function unitsEqual(a: TrickUnit, b: TrickUnit): boolean {
  if (a.kind !== b.kind || a.count !== b.count) return false;
  if (a.kind === 'repeated') return a.card === (b as typeof a).card;
  const bm = (b as typeof a).members;
  return a.members.length === bm.length && a.members.every((m, i) => m === bm[i]);
}

function sortUnits(trump: Trump, units: TrickUnit[]): TrickUnit[] {
  return units
    .slice()
    .sort((a, b) => unitSize(a) - unitSize(b) || compareCards(trump, unitFirst(a), unitFirst(b)));
}

function sortedKeys(trump: Trump, counts: CardCounts): Card[] {
  return [...counts.keys()].filter((c) => (counts.get(c) ?? 0) > 0).sort((a, b) => compareCards(trump, a, b));
}

// ---------------------------------------------------------------- lead decomposition

/** Tractors starting at `card` (ported verbatim, including its shared running minimum). */
export function findTractorsFromStart(
  trump: Trump,
  card: Card,
  count: number,
  counts: CardCounts,
  req: TractorRequirements,
): TrickUnit[] {
  const out: TrickUnit[] = [];
  if (count < req.minCount) return out;
  let frontier: Array<[Card, Card[]]> = successor(trump, card).map((c) => [c, [card]]);
  let minCount = count;
  while (frontier.length > 0) {
    const next: Array<[Card, Card[]]> = [];
    for (const [nc, path0] of frontier) {
      const ncount = counts.get(nc) ?? 0;
      if (ncount >= 2) {
        minCount = Math.min(minCount, ncount);
        const path = [...path0, nc];
        if (minCount >= req.minCount && path.length >= req.minLength) {
          out.push({ kind: 'tractor', members: path, count: minCount });
        }
        for (const s of successor(trump, nc)) next.push([s, path]);
      }
    }
    frontier = next;
  }
  return out;
}

function subtractUnit(counts: CardCounts, u: TrickUnit): CardCounts {
  const out = new Map(counts);
  const take = (c: Card, n: number) => {
    const left = (out.get(c) ?? 0) - n;
    if (left <= 0) out.delete(c);
    else out.set(c, left);
  };
  if (u.kind === 'repeated') take(u.card, u.count);
  else for (const m of u.members) take(m, u.count);
  return out;
}

function findPlaysInner(
  trump: Trump,
  counts: CardCounts,
  numCards: number,
  req: TractorRequirements,
  minStart: Card | null,
): TrickUnit[][] {
  if (numCards === 0) return [];
  const keys = sortedKeys(trump, counts).filter((c) => minStart === null || compareCards(trump, c, minStart) >= 0);
  const starts: TrickUnit[] = [];
  const first = keys[0];
  if (first !== undefined) {
    const count = counts.get(first)!;
    const tractors = findTractorsFromStart(trump, first, count, counts, req);
    const allConsumed =
      tractors.length > 0
      && tractors.every((t) => t.kind === 'tractor' && t.members.every((c) => (counts.get(c) ?? 0) === t.count));
    starts.push(...tractors);
    if (!allConsumed) starts.push({ kind: 'repeated', card: first, count });
  }
  const whole = starts.find((u) => unitSize(u) === numCards);
  if (whole) return [[whole]];
  const plays: TrickUnit[][] = [];
  for (const start of starts) {
    const sub = subtractUnit(counts, start);
    for (const play of findPlaysInner(trump, sub, numCards - unitSize(start), req, unitFirst(start))) {
      plays.push([...play, start]);
    }
  }
  return plays;
}

/** Every way to read `cards` (one effective suit) as a set of units. */
export function findPlays(trump: Trump, req: TractorRequirements, cards: readonly Card[]): TrickUnit[][] {
  return findPlaysInner(trump, countCards(cards), cards.length, req, null);
}

/** The distinct readings of a lead, each sorted, in the order the default choice ranks them (best last). */
export function possibleFormats(trump: Trump, req: TractorRequirements, cards: readonly Card[]): TrickUnit[][] {
  const plays = findPlays(trump, req, cards).map((u) => sortUnits(trump, u));
  const key = (units: TrickUnit[]) => {
    let best: [number, number] = [-1, -1];
    for (const u of units) {
      const k: [number, number] = [unitSize(u), u.kind === 'tractor' ? 1 : 0];
      if (k[0] > best[0] || (k[0] === best[0] && k[1] > best[1])) best = k;
    }
    return best;
  };
  const ranked = plays.slice().sort((a, b) => {
    const ka = key(a);
    const kb = key(b);
    return ka[0] - kb[0] || ka[1] - kb[1];
  });
  // Keep the last copy of each reading, so the last entry is the default choice.
  const out: TrickUnit[][] = [];
  for (let i = ranked.length - 1; i >= 0; i--) {
    const p = ranked[i]!;
    if (!out.some((q) => q.length === p.length && q.every((u, j) => unitsEqual(u, p[j]!)))) out.unshift(p);
  }
  return out;
}

export class TrickError extends Error {
  constructor(
    readonly code: 'wrongNumberOfSuits' | 'nonMatchingProposal' | 'illegalPlay' | 'nonMatchingPlay',
    message: string,
  ) {
    super(message);
    this.name = 'TrickError';
  }
}

/**
 * Read a lead as a trick format. With `hint` (an index into `possibleFormats`)
 * the leader picks the reading; otherwise the one with the largest unit wins
 * (shengji's rule: ties go to the reading found last).
 */
export function formatFromCards(
  trump: Trump,
  req: TractorRequirements,
  cards: readonly Card[],
  hint?: number,
): TrickFormat {
  if (cards.length === 0) throw new TrickError('wrongNumberOfSuits', 'No cards.');
  const suit = effectiveSuit(trump, cards[0]!);
  if (cards.some((c) => effectiveSuit(trump, c) !== suit)) {
    throw new TrickError('wrongNumberOfSuits', 'A lead must be all one suit (or all trump).');
  }
  if (hint !== undefined) {
    const options = possibleFormats(trump, req, cards);
    const chosen = options[hint];
    if (!chosen) throw new TrickError('nonMatchingProposal', 'That reading of the cards is not possible.');
    return { suit, units: chosen };
  }
  const plays = findPlays(trump, req, cards);
  const key = (units: TrickUnit[]) => {
    let best: [number, number] | null = null;
    for (const u of units) {
      const k: [number, number] = [unitSize(u), u.kind === 'tractor' ? 1 : 0];
      if (!best || k[0] > best[0] || (k[0] === best[0] && k[1] > best[1])) best = k;
    }
    return best ?? [-1, -1];
  };
  const ranked = plays.slice().sort((a, b) => {
    const ka = key(a);
    const kb = key(b);
    return ka[0] - kb[0] || ka[1] - kb[1];
  });
  const units = ranked.pop();
  if (!units) throw new TrickError('illegalPlay', 'Those cards cannot be led together.');
  return { suit, units: sortUnits(trump, units) };
}

// ---------------------------------------------------------------- decomposition ordering

/** Rust's Vec<usize> ordering: element-wise, then shorter first. */
function lexCmp(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i]! - b[i]!;
  return a.length - b.length;
}

function lexCmp2(a: readonly number[][], b: readonly number[][]): boolean {
  return a.length === b.length && a.every((x, i) => lexCmp(x, b[i]!) === 0);
}

function uniqueBy<T>(items: T[]): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const it of items) {
    const k = JSON.stringify(it);
    if (!seen.has(k)) {
      seen.add(k);
      out.push(it);
    }
  }
  return out;
}

const partitionCache = new Map<number, number[][]>();

/** All ways to split `num` cards into tuple sizes, most complex first: 4 → [4] [3,1] [2,2] [2,1,1] [1,1,1,1]. */
export function findTuplePartitions(num: number): number[][] {
  const cached = partitionCache.get(num);
  if (cached) return cached;
  let groupings: number[][] = [];
  if (num === 1) {
    groupings.push([1]);
  } else {
    for (const g of findTuplePartitions(num - 1)) {
      const incremented = new Set<number>();
      for (const v of g) {
        if (incremented.has(v)) continue;
        incremented.add(v);
        let found = false;
        groupings.push(g.map((vv) => (vv === v && !found ? ((found = true), vv + 1) : vv)));
      }
      groupings.push([...g, 1]);
    }
  }
  groupings.sort((a, b) => lexCmp(b, a));
  groupings = groupings.filter((g, i) => i === 0 || lexCmp(g, groupings[i - 1]!) !== 0);
  partitionCache.set(num, groupings);
  return groupings;
}

const assignmentCache = new Map<number, number[][][]>();

/** Set partitions of indices 0..length-1 into blocks, biggest blocks first. */
function computeAdjacentAssignments(length: number): number[][][] {
  if (length === 1) return [[[0]]];
  const cached = assignmentCache.get(length);
  if (cached) return cached;
  const elem = length - 1;
  const out: number[][][] = [];
  for (const part of computeAdjacentAssignments(length - 1)) {
    for (let i = 0; i < part.length; i++) out.push(part.map((b, j) => (j === i ? [...b, elem] : b.slice())));
    out.push([...part.map((b) => b.slice()), [elem]]);
  }
  const maxLen = (a: number[][]) => Math.max(...a.map((b) => b.length));
  out.sort((a, b) => maxLen(b) - maxLen(a) || a.length - b.length);
  const deduped = out.filter((a, i) => i === 0 || !lexCmp2(a, out[i - 1]!));
  assignmentCache.set(length, deduped);
  return deduped;
}

/** Distinct orderings of `values`, in the order itertools' permutations().unique() yields them. */
function uniquePermutations(values: readonly number[]): number[][] {
  const out: number[][] = [];
  const used = new Array<boolean>(values.length).fill(false);
  const cur: number[] = [];
  const rec = () => {
    if (cur.length === values.length) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < values.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(values[i]!);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return uniqueBy(out);
}

function cartesian<T>(lists: T[][]): T[][] {
  let acc: T[][] = [[]];
  for (const list of lists) {
    const next: T[][] = [];
    for (const prefix of acc) for (const item of list) next.push([...prefix, item]);
    acc = next;
  }
  return acc;
}

/** Ways to arrange tuple sizes into runs of adjacent tuples, e.g. [3,2,2] → [[3,2,2]], [[2,3,2]], …, [[3],[2],[2]]. */
function groupIntoSequentialTuples(values: readonly number[]): number[][][] {
  const out: number[][][] = [];
  for (const assignment of computeAdjacentAssignments(values.length)) {
    const mapped = assignment.map((sub) => sub.map((i) => values[i]!));
    if (mapped.every((p) => p.every((v) => v === p[0]))) out.push(mapped);
    else for (const combo of cartesian(mapped.map(uniquePermutations))) out.push(combo);
  }
  return uniqueBy(out);
}

/** Past this many cards in one unit the full ordering explodes; fall back to plain tuples. */
const MAX_FULL_DECOMPOSITION = 12;
const fullDecompositionCache = new Map<number, number[][][]>();

/** Every requirement `num` cards could fall back to, most demanding first. Singles never need adjacency. */
export function fullDecompositionOrdering(num: number): number[][][] {
  const cached = fullDecompositionCache.get(num);
  if (cached) return cached;
  const full: number[][][] = [];
  for (const group of findTuplePartitions(num)) {
    const one = group.indexOf(1);
    const oneIdx = one < 0 ? group.length : one;
    const gt1 = group.slice(0, oneIdx);
    const eq1 = group.slice(oneIdx);
    if (gt1.length === 0) {
      full.push(eq1.map((v) => [v]));
    } else if (num > MAX_FULL_DECOMPOSITION) {
      full.push([...gt1.map((v) => [v]), ...eq1.map((v) => [v])]);
    } else {
      for (const d of groupIntoSequentialTuples(gt1)) {
        const dd = [...d.map((x) => x.slice()), ...eq1.map((v) => [v])];
        dd.sort((a, b) => lexCmp(b, a));
        full.push(dd);
      }
    }
  }
  const result = uniqueBy(full);
  fullDecompositionCache.set(num, result);
  return result;
}

/** The fallback requirements after a format's own shapes, in order (shengji's subsequent_decomposition_ordering). */
export function subsequentDecompositionOrdering(adjReqs0: readonly Shape[], includeNewAdjacency: boolean): Shape[][] {
  if (!adjReqs0.every((a) => a.length > 0)) return [];
  const adjReqs = adjReqs0.map((a) => a.slice().sort((x, y) => y - x));
  const decompositions = adjReqs.map((req) => {
    const len = req.reduce((n, v) => n + v, 0);
    const decomp = fullDecompositionOrdering(len).slice().reverse();
    while (decomp.length > 0) {
      const v = decomp.pop()!;
      if (v.length === 1 && lexCmp(v[0]!, req) === 0) break;
    }
    return decomp;
  });
  const current = new Map<number, Shape[]>();
  adjReqs.forEach((r, i) => current.set(i, [r]));
  const canInclude = adjReqs.map((a) => includeNewAdjacency || a.length > 1);
  const h = adjReqs.map((_, i) => i);
  const out: Shape[][] = [];
  for (;;) {
    h.sort((a, b) => decompositions[b]!.length - decompositions[a]!.length);
    const idx = h[0];
    if (idx === undefined) break;
    const v = decompositions[idx]!.pop();
    if (!v) break;
    current.set(idx, v);
    const include = h.every((i) => current.get(i)!.every((a) => a.length === 1 || canInclude[i]));
    if (include) {
      const full = h.flatMap((i) => current.get(i)!.map((a) => a.slice()));
      full.sort((a, b) => lexCmp(b, a));
      out.push(full);
    }
  }
  return out;
}

/** The requirements a follow is checked against, in order: the lead's own shapes, then the fallbacks. */
export function* decomposition(format: TrickFormat, policy: TrickDrawPolicy): Generator<Shape[]> {
  const shapes = format.units.map(unitShape);
  yield shapes;
  const includeNew = policy !== 'onlyDrawTractorOnTractor' && policy !== 'longerTuplesProtectedAndOnlyDrawTractorOnTractor';
  yield* subsequentDecompositionOrdering(shapes, includeNew);
}

// ---------------------------------------------------------------- format matching

interface MatchOptions {
  /** Under a protection policy: may (card, copies) be used? Checked against the counts passed in. */
  filter?: (card: Card, copies: number) => boolean;
  /** Stop after this many matches. */
  limit?: number;
}

function protectionFilter(counts: CardCounts, policy: TrickDrawPolicy): MatchOptions['filter'] {
  if (policy === 'longerTuplesProtected' || policy === 'longerTuplesProtectedAndOnlyDrawTractorOnTractor') {
    return (card, copies) => (counts.get(card) ?? 0) <= copies;
  }
  return undefined;
}

function compareShapes(a: Shape, b: Shape): number {
  // Longer adjacency first, then larger tuples, so identical shapes sit together.
  return b.length - a.length || lexCmp(b, a);
}

function comparePlacements(trump: Trump, a: Placement, b: Placement): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const c = compareCards(trump, a[i]![0], b[i]![0]);
    if (c !== 0) return c;
    if (a[i]![1] !== b[i]![1]) return a[i]![1] - b[i]![1];
  }
  return a.length - b.length;
}

/** Placements for one shape from the counts, highest first (feasibility vs. what's left is checked later). */
function candidatePlacements(trump: Trump, counts: CardCounts, shape: Shape, filter: MatchOptions['filter']): Placement[] {
  const keys = sortedKeys(trump, counts).reverse();
  const ok = (card: Card, n: number) => (counts.get(card) ?? 0) >= n && (!filter || filter(card, n));
  if (shape.length === 1) {
    const n = shape[0]!;
    return keys.filter((c) => ok(c, n)).map((c) => [[c, n]] as Placement);
  }
  const out: Placement[] = [];
  for (const start of keys) {
    if (!ok(start, shape[0]!)) continue;
    const stack: Array<{ card: Card; i: number; path: Placement }> = [{ card: start, i: 0, path: [] }];
    while (stack.length > 0) {
      const { card, i, path } = stack.pop()!;
      const n = shape[i]!;
      if (!ok(card, n)) continue;
      const next: Placement = [...path, [card, n]];
      if (i === shape.length - 1) out.push(next);
      else for (const s of successor(trump, card)) stack.push({ card: s, i: i + 1, path: next });
    }
  }
  out.sort((a, b) => comparePlacements(trump, b, a));
  return out;
}

/**
 * Assignments of `counts` to the shapes (each match aligned with `shapes`;
 * within one shape the placements run highest first). Backtracking with
 * canonical ordering for repeated shapes, so each assignment appears once.
 */
export function findFormatMatches(trump: Trump, counts: CardCounts, shapes: readonly Shape[], opts: MatchOptions = {}): Placement[][] {
  const limit = opts.limit ?? Infinity;
  const order = shapes.map((_, i) => i).sort((a, b) => compareShapes(shapes[a]!, shapes[b]!) || a - b);
  const candByKey = new Map<string, Placement[]>();
  const cands = order.map((i) => {
    const key = shapes[i]!.join(',');
    let c = candByKey.get(key);
    if (!c) {
      c = candidatePlacements(trump, counts, shapes[i]!, opts.filter);
      candByKey.set(key, c);
    }
    return c;
  });
  const sameAsPrev = order.map((i, k) => k > 0 && compareShapes(shapes[i]!, shapes[order[k - 1]!]!) === 0);
  // How many slots of the same tuple shape follow position k (inclusive), for pruning.
  const runLeft = order.map((_, k) => {
    let n = 0;
    for (let j = k; j < order.length && (j === k || sameAsPrev[j]); j++) n++;
    return n;
  });
  const remaining = new Map(counts);
  const chosen: Array<{ idx: number; p: Placement }> = [];
  const results: Placement[][] = [];

  const fits = (p: Placement) => p.every(([c, n]) => (remaining.get(c) ?? 0) >= n);
  const apply = (p: Placement, sign: 1 | -1) => {
    for (const [c, n] of p) remaining.set(c, (remaining.get(c) ?? 0) - sign * n);
  };

  const rec = (k: number): boolean => {
    if (k === order.length) {
      const match: Placement[] = new Array(shapes.length);
      // Within each shape, give the original units the placements highest first.
      const byShape = new Map<string, Placement[]>();
      for (let j = 0; j < order.length; j++) {
        const key = shapes[order[j]!]!.join(',');
        const list = byShape.get(key) ?? [];
        list.push(chosen[j]!.p);
        byShape.set(key, list);
      }
      for (const list of byShape.values()) list.sort((a, b) => comparePlacements(trump, b, a));
      const taken = new Map<string, number>();
      shapes.forEach((s, i) => {
        const key = s.join(',');
        const t = taken.get(key) ?? 0;
        taken.set(key, t + 1);
        match[i] = byShape.get(key)![t]!;
      });
      results.push(match);
      return results.length >= limit;
    }
    const list = cands[k]!;
    const start = sameAsPrev[k] ? chosen[k - 1]!.idx : 0;
    const shape = shapes[order[k]!]!;
    for (let idx = start; idx < list.length; idx++) {
      const p = list[idx]!;
      if (!fits(p)) continue;
      if (shape.length === 1 && runLeft[k]! > 1) {
        // Prune: enough copies left among candidates idx.. for the rest of this run?
        const n = shape[0]!;
        let room = 0;
        for (let j = idx; j < list.length && room < runLeft[k]!; j++) room += Math.floor((remaining.get(list[j]![0]![0]) ?? 0) / n);
        if (room < runLeft[k]!) break;
      }
      apply(p, 1);
      chosen.push({ idx, p });
      const stop = rec(k + 1);
      chosen.pop();
      apply(p, -1);
      if (stop) return true;
    }
    return false;
  };
  rec(0);
  return results;
}

export function canMatch(trump: Trump, counts: CardCounts, shapes: readonly Shape[], policy: TrickDrawPolicy = 'noProtections'): boolean {
  return findFormatMatches(trump, counts, shapes, { filter: protectionFilter(counts, policy), limit: 1 }).length > 0;
}

function placementToUnit(p: Placement): TrickUnit {
  if (p.length === 1) return { kind: 'repeated', card: p[0]![0], count: p[0]![1] };
  return { kind: 'tractor', count: Math.min(...p.map(([, n]) => n)), members: p.map(([c]) => c) };
}

/** Ways `cards` match the format (all one suit, the led one or trump), as units aligned with the format's. */
export function matchFormat(trump: Trump, format: TrickFormat, cards: readonly Card[], limit = 256): TrickUnit[][] {
  if (cards.length === 0 || cards.length !== formatSize(format)) return [];
  const suit = effectiveSuit(trump, cards[0]!);
  if (cards.some((c) => effectiveSuit(trump, c) !== suit)) return [];
  if (suit !== format.suit && suit !== 'T') return [];
  return findFormatMatches(trump, countCards(cards), format.units.map(unitShape), { limit }).map((m) => m.map(placementToUnit));
}

// ---------------------------------------------------------------- legality

/**
 * Whether `proposed` is a legal follow to `format` from `hand` (shengji's
 * TrickFormat::is_legal_play, without bombs/rainbows). The hand includes the
 * proposed cards.
 */
export function isLegalFollow(
  trump: Trump,
  format: TrickFormat,
  hand: readonly Card[],
  proposed: readonly Card[],
  policy: TrickDrawPolicy,
): boolean {
  const required = formatSize(format);
  if (proposed.length !== required) return false;
  const inSuit = (c: Card) => effectiveSuit(trump, c) === format.suit;
  const proposedInSuit = proposed.filter(inSuit).length;
  const available = hand.filter(inSuit);
  if (proposedInSuit < required) return available.length === proposedInSuit;
  if (policy === 'noFormatBasedDraw') return true;

  const playCounts = countCards(proposed);
  const availableCounts = countCards(available);
  const availableMinusPlay = new Map(availableCounts);
  for (const [c, n] of playCounts) availableMinusPlay.set(c, Math.max(0, (availableMinusPlay.get(c) ?? 0) - n));
  for (const [c, n] of [...availableMinusPlay]) if (n === 0) availableMinusPlay.delete(c);
  const protectedPolicy = policy === 'longerTuplesProtected' || policy === 'longerTuplesProtectedAndOnlyDrawTractorOnTractor';

  for (const requirement of decomposition(format, policy)) {
    if (canMatch(trump, playCounts, requirement, 'noProtections')) {
      if (protectedPolicy) {
        // A longer tuple in the play can stand in for a shorter slot; that's
        // only fine when the rest of the hand has no genuine tuple for it.
        const sizes = new Set(requirement.filter((s) => s.length === 1 && s[0]! >= 2).map((s) => s[0]!));
        for (const n of sizes) {
          const slots = requirement.filter((s) => s.length === 1 && s[0] === n).length;
          let covered = 0;
          for (const [c, pct] of playCounts) if (pct === n && inSuit(c)) covered++;
          const left = slots - covered;
          if (left <= 0) continue;
          const units: Shape[] = Array.from({ length: left }, () => [n]);
          if (canMatch(trump, availableMinusPlay, units, policy)) return false;
        }
      }
      return true;
    }
    if (canMatch(trump, availableCounts, requirement, policy)) return false;
  }
  return true;
}

/** The first requirement this hand can (and so must) meet when following, or null when only suit matters. */
export function requiredShapes(trump: Trump, format: TrickFormat, hand: readonly Card[], policy: TrickDrawPolicy): Shape[] | null {
  if (policy === 'noFormatBasedDraw') return null;
  const available = countCards(hand.filter((c) => effectiveSuit(trump, c) === format.suit));
  for (const requirement of decomposition(format, policy)) {
    if (canMatch(trump, available, requirement, policy)) return requirement;
  }
  return null;
}

// ---------------------------------------------------------------- comparing plays

/** Does `m` (a follower's matched units) beat `winner` (the current winner's)? Both aligned with the format. */
export function defeats(trump: Trump, m: TrickUnit[], winner: TrickUnit[], policy: ThrowEvaluationPolicy): boolean {
  const gt = (a: Card, b: Card) => compareEffective(trump, a, b) > 0;
  if (policy === 'all') return m.every((u, i) => gt(unitFirst(u), unitFirst(winner[i]!)));
  if (policy === 'highest') {
    const max = (units: TrickUnit[]) =>
      units.map(unitLast).reduce((best, c) => (compareCards(trump, c, best) > 0 ? c : best));
    return gt(max(m), max(winner));
  }
  const skipSingles = m.length > 1 && m.some((u) => unitSize(u) > 1);
  const comparisons = m
    .map((u, i) => ({ size: unitSize(u), cmp: Math.sign(compareEffective(trump, unitFirst(u), unitFirst(winner[i]!))) }))
    .filter((c) => !skipSingles || c.size > 1);
  comparisons.sort((a, b) => b.size - a.size || (a.cmp === 0 ? 1 : 0) - (b.cmp === 0 ? 1 : 0));
  for (const c of comparisons) {
    if (c.cmp > 0) return true;
    if (c.cmp < 0) return false;
  }
  return false;
}

/** Within each shape, units highest first — the order matches come in, so comparisons line up. */
export function normalizeUnits(trump: Trump, format: TrickFormat, units: TrickUnit[]): TrickUnit[] {
  const shapes = format.units.map((u) => unitShape(u).join(','));
  const byShape = new Map<string, TrickUnit[]>();
  units.forEach((u, i) => {
    const list = byShape.get(shapes[i]!) ?? [];
    list.push(u);
    byShape.set(shapes[i]!, list);
  });
  for (const list of byShape.values()) {
    list.sort((a, b) => comparePlacements(trump, unitPlacement(b), unitPlacement(a)));
  }
  const taken = new Map<string, number>();
  return shapes.map((s) => {
    const t = taken.get(s) ?? 0;
    taken.set(s, t + 1);
    return byShape.get(s)![t]!;
  });
}

function unitPlacement(u: TrickUnit): Placement {
  return u.kind === 'repeated' ? [[u.card, u.count]] : u.members.map((m) => [m, u.count] as [Card, number]);
}

// ---------------------------------------------------------------- throws

/**
 * For a multi-unit lead (a throw): the first other player, in play order,
 * holding something that beats one of its units, and that unit. Null when
 * the throw stands.
 */
export function findThrowBreaker(
  trump: Trump,
  format: TrickFormat,
  others: ReadonlyArray<{ seat: number; hand: readonly Card[] }>,
): { seat: number; unit: TrickUnit } | null {
  if (format.units.length <= 1) return null;
  for (const { seat, hand } of others) {
    const inSuit = countCards(hand.filter((c) => effectiveSuit(trump, c) === format.suit));
    for (const unit of format.units) {
      if (unit.kind === 'repeated') {
        for (const [c, ct] of inSuit) {
          if (ct >= unit.count && compareEffective(trump, c, unit.card) > 0) return { seat, unit };
        }
      } else {
        const from = unit.members[1]!;
        for (const c of sortedKeys(trump, inSuit)) {
          if (compareCards(trump, c, from) < 0) continue;
          const found = findTractorsFromStart(trump, c, inSuit.get(c)!, inSuit, {
            minCount: unit.count,
            minLength: unit.members.length,
          });
          if (found.length > 0) return { seat, unit };
        }
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------- descriptions

export function tupleName(n: number): string {
  return ['', 'single', 'pair', 'triple', 'quadruple', 'quintuple'][n] ?? `${n}-tuple`;
}

export function shapeName(shape: Shape): string {
  if (shape.length === 1) return tupleName(shape[0]!);
  const rect = shape.every((v) => v === shape[0]);
  if (rect) {
    const w = shape[0]!;
    if (shape.length === 2) return w === 2 ? 'tractor' : `tractor of ${tupleName(w)}s`;
    return `${shape.length}-tractor of ${tupleName(w)}s`;
  }
  return `${tupleName(shape[0]!)} followed immediately by a ${shape.slice(1).map(tupleName).join(', ')}`;
}

/** "a pair", "2 pairs", "1 tractor, 2 singles". */
export function describeShapes(shapes: readonly Shape[]): string {
  const counts = new Map<string, number>();
  for (const s of shapes) counts.set(shapeName(s), (counts.get(shapeName(s)) ?? 0) + 1);
  const entries = [...counts.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
  if (entries.length === 1) {
    const [desc, n] = entries[0]!;
    return n === 1 ? `a ${desc}` : `${n} ${desc}s`;
  }
  return entries.map(([d, n]) => `${n} ${d}${n > 1 ? 's' : ''}`).join(', ');
}

export function describeUnit(u: TrickUnit): string {
  return shapeName(unitShape(u));
}
