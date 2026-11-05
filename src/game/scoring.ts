// End-of-round scoring and level advancement — a port of shengji's
// mechanics/src/scoring.rs and PlayPhase::compute_player_level_deltas.
//
// With the defaults (step = 20 points per deck, 2 steps to turn over, a
// one-step dead zone) two decks score the classic "80 points" way:
//   0 → landlord team +3 · 5–35 → +2 · 40–75 → +1 · 80–115 → attackers take
//   over, nobody moves · 120–155 → attackers +1 · 160–195 → +2 · 200+ → +3 …

import { levelSuccessor, type Level, type Rank } from './cards';

export interface ScoringParams {
  /** Points per step, per deck. */
  stepSizePerDeck: number;
  /** Steps (from 0) the attackers need to take the landlord seat. */
  stepsToTurnover: number;
  /** Steps past the turnover where nobody levels up. */
  deadzoneSize: number;
  /** The window containing 0 is narrower and worth one extra level (shengji default). */
  truncateZeroCrossingWindow: boolean;
  /** A landlord team that wins short-handed (a friend never showed) gets a bonus level. */
  bonusLevelForSmallerTeam: boolean;
}

export const DEFAULT_SCORING: ScoringParams = {
  stepSizePerDeck: 20,
  stepsToTurnover: 2,
  deadzoneSize: 1,
  truncateZeroCrossingWindow: true,
  bonusLevelForSmallerTeam: true,
};

interface Segment {
  start: number; // inclusive
  end: number; // exclusive
  delta: number;
}

export interface Materialized {
  landlordWins: Segment[]; // sorted by start
  landlordLoses: Segment[];
  totalPoints: number;
  stepSize: number;
}

export function stepSize(params: ScoringParams, numDecks: number): number {
  return params.stepSizePerDeck * numDecks;
}

export function materialize(params: ScoringParams, numDecks: number, totalPoints: number): Materialized {
  const s = stepSize(params, numDecks);
  if (s <= 0 || s > totalPoints || s % 5 !== 0) throw new Error(`Step size ${s} must be a multiple of 5 between 5 and ${totalPoints}.`);
  if (params.stepsToTurnover < 1) throw new Error('The landlord team must be able to win.');
  const wins: Segment[] = [];
  if (params.truncateZeroCrossingWindow) {
    let delta = 1;
    for (let offset = params.stepsToTurnover; offset >= 2; offset--) {
      wins.push({ start: (offset - 1) * s, end: offset * s, delta });
      delta++;
    }
    wins.push({ start: 5, end: s, delta });
    wins.push({ start: 5 - s, end: 5, delta: delta + 1 });
  } else {
    wins.push({ start: (params.stepsToTurnover - 1) * s, end: params.stepsToTurnover * s, delta: 1 });
  }
  const loses: Segment[] = [];
  const turnover = params.stepsToTurnover * s;
  if (params.deadzoneSize > 0) loses.push({ start: turnover, end: turnover + params.deadzoneSize * s, delta: 0 });
  const after = turnover + params.deadzoneSize * s;
  loses.push({ start: after, end: after + s, delta: 1 });
  wins.sort((a, b) => a.start - b.start);
  return { landlordWins: wins, landlordLoses: loses, totalPoints, stepSize: s };
}

export interface PartialScore {
  landlordWon: boolean;
  landlordDelta: number;
  attackerDelta: number;
}

/** Levels gained for `attackerPoints` (may be negative after penalties). */
export function scoreRound(m: Materialized, attackerPoints: number): PartialScore {
  const threshold = m.landlordWins[m.landlordWins.length - 1]!.end;
  if (attackerPoints < threshold) {
    // Walk down from the top window, extending below the lowest one as needed.
    const segs = m.landlordWins.slice().reverse();
    let seg = segs[0]!;
    for (let i = 0; i < 60; i++) {
      seg = i < segs.length ? segs[i]! : { start: seg.start - (seg.end - seg.start), end: seg.start, delta: seg.delta + 1 };
      if (seg.start <= attackerPoints && attackerPoints < seg.end) return { landlordWon: true, landlordDelta: seg.delta, attackerDelta: 0 };
    }
  } else {
    const segs = m.landlordLoses;
    let seg = segs[0]!;
    for (let i = 0; i < 60; i++) {
      seg = i < segs.length ? segs[i]! : { start: seg.end, end: seg.end + (seg.end - seg.start), delta: seg.delta + 1 };
      if (seg.start <= attackerPoints && attackerPoints < seg.end) return { landlordWon: false, landlordDelta: 0, attackerDelta: seg.delta };
    }
  }
  throw new Error('Failed to score the round.');
}

/** The scoring table as (points from, outcome) rows, for the rules / scoreboard. */
export function explainScoring(m: Materialized): Array<{ from: number; score: PartialScore }> {
  const rows: Array<{ from: number; score: PartialScore }> = [];
  let prev: string | null = null;
  for (let pts = 0; pts <= m.totalPoints; pts += 5) {
    const score = scoreRound(m, pts);
    const key = JSON.stringify(score);
    if (key !== prev) rows.push({ from: pts, score });
    prev = key;
  }
  return rows;
}

/** The fewest attacker points that change the outcome, or null if none. */
export function nextThreshold(m: Materialized, attackerPoints: number): number | null {
  const now = JSON.stringify(scoreRound(m, attackerPoints));
  for (let p = attackerPoints + 5; p <= m.totalPoints + 5; p += 5) {
    if (JSON.stringify(scoreRound(m, p)) !== now) return p;
  }
  return null;
}

export type AdvancementPolicy =
  /** Players must defend (win as landlord team) their max rank to pass it. */
  | 'unrestricted'
  /** No defending needed anywhere. */
  | 'fullyUnrestricted'
  /** Also defend the point ranks (5, 10, K). */
  | 'defendPoints';

export interface Advancement {
  from: Level;
  to: Level;
  steps: number;
  blocked: boolean;
  /** Advanced past the max rank: this player has won the game. */
  finished: boolean;
}

/**
 * Advance one player by `bump` levels, stopping where they must defend
 * (shengji's rules). `landlordLevel` is the round's landlord level.
 */
export function advancePlayer(
  level: Level,
  bump: number,
  isDefending: boolean,
  landlordLevel: Level,
  maxRank: Level,
  policy: AdvancementPolicy,
): Advancement {
  let cur = level;
  let steps = 0;
  let blocked = false;
  let finished = false;
  for (let i = 0; i < bump; i++) {
    let mustDefend = false;
    if (policy !== 'fullyUnrestricted') {
      if (cur === maxRank || (levelSuccessor(cur) === maxRank && maxRank === 'NT')) mustDefend = true;
    }
    if (policy === 'defendPoints' && (cur === '5' || cur === 'T' || cur === 'K')) mustDefend = true;
    const landlordMustDefend = mustDefend && cur === 'NT';
    if (mustDefend && (!isDefending || i > 0 || (landlordMustDefend && landlordLevel !== 'NT'))) {
      blocked = true;
      break;
    }
    if (cur === maxRank) {
      finished = true;
      steps++;
      break;
    }
    cur = levelSuccessor(cur) ?? cur;
    steps++;
  }
  return { from: level, to: cur, steps, blocked, finished };
}

export function isRank(level: Level): level is Rank {
  return level !== 'NT';
}
