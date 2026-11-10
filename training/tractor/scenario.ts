// The training curriculum: which rounds the network practises. Every
// episode is one round, started mid-match from sampled levels (createMatchAt)
// so the network meets every rank situation without playing up from 2:
//
//   table      the focus table, 4-player Tractor (2 decks), 85% of rounds;
//              the other 15% are a variety mix so the network still plays
//              every form of the game: 2-12 players (weighted toward 5, 6 and
//              8), Finding Friends on every odd table and half the even ones,
//              decks auto or one off (15%), NT as the final rank (10%)
//   levels     the landlord's level — the trump number — is where play changes
//              most: point ranks (5, 10, K: the trump numbers are point cards)
//              and A (the side suits' top card is the K; the final rank must be
//              defended) are oversampled; the rest share what's left. Attackers'
//              levels are sampled separately (attackers at A play differently).
//              Finding Friends players each get their own level near the landlord's.
//   first      15% are a match's first round: everyone at 2, no landlord, the
//   rounds     bidding decides who leads
//   bidding    live (90%), or in turns (10%: hot-seat tables bid in turns)
//   rules      the standard rules only (no house rules)
//
// `focus: 1` gives only the focus table; `focus: 0` only the variety mix.

import type { Level, Rank } from '@/game/cards';
import { autoNumDecks, defaultConfig, type TractorConfig } from '@/game/config';

export interface ScenarioWeights {
  /** Share of rounds at the focus table (4-player Tractor, standard decks, final rank A). */
  focus: number;
  /** The variety rounds: table sizes, Finding Friends on even tables, deck counts off the default, NT final rank. */
  players: Record<number, number>;
  ffOnEvenTables: number;
  deckJitter: number;
  ntMaxRank: number;
  /** Every round: first rounds, turn bidding, the landlord's level class. */
  firstRound: number;
  turnBidding: number;
  /** Landlord level classes. */
  pointRanks: number;
  ace: number;
  other: number;
}

export const DEFAULT_SCENARIO: ScenarioWeights = {
  focus: 0.85,
  players: { 2: 0.5, 3: 1, 4: 1, 5: 2, 6: 2, 7: 1, 8: 1.5, 9: 0.5, 10: 0.5, 11: 0.25, 12: 0.25 },
  ffOnEvenTables: 0.5,
  deckJitter: 0.15,
  ntMaxRank: 0.1,
  firstRound: 0.15,
  turnBidding: 0.1,
  pointRanks: 0.35,
  ace: 0.15,
  other: 0.5,
};

export interface Scenario {
  players: number;
  config: TractorConfig;
  levels: Level[];
  landlord: number | null;
  /** Label for stats: 'point' | 'ace' | 'other' | 'first'. */
  levelClass: string;
  /** A focus-table round (vs the variety mix). */
  focus: boolean;
}

const POINT_RANKS: Rank[] = ['5', 'T', 'K'];
const OTHER_RANKS: Rank[] = ['2', '3', '4', '6', '7', '8', '9', 'J', 'Q'];
const ALL: Rank[] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];

function pick<T>(rnd: () => number, xs: readonly T[]): T {
  return xs[Math.floor(rnd() * xs.length)]!;
}

function weighted<K extends string | number>(rnd: () => number, w: Record<K, number>): K {
  const entries = Object.entries(w) as Array<[string, number]>;
  const total = entries.reduce((a, [, x]) => a + x, 0);
  let r = rnd() * total;
  for (const [k, x] of entries) {
    r -= x;
    if (r <= 0) return (isNaN(Number(k)) ? k : Number(k)) as K;
  }
  return (isNaN(Number(entries[0]![0])) ? entries[0]![0] : Number(entries[0]![0])) as K;
}

function landlordLevel(rnd: () => number, w: ScenarioWeights, nt: boolean): { level: Level; cls: string } {
  const cls = weighted(rnd, { point: w.pointRanks, ace: w.ace, other: w.other });
  if (cls === 'point') return { level: pick(rnd, POINT_RANKS), cls };
  if (cls === 'ace') return { level: nt && rnd() < 0.3 ? 'NT' : 'A', cls };
  return { level: pick(rnd, OTHER_RANKS), cls };
}

function near(rnd: () => number, level: Level, maxRank: Level): Level {
  const ranks: Level[] = maxRank === 'NT' ? [...ALL, 'NT'] : [...ALL];
  const i = Math.max(0, ranks.indexOf(level));
  const j = Math.max(0, Math.min(ranks.length - 1, i + Math.round((rnd() - 0.5) * 6)));
  return ranks[j]!;
}

export function sampleScenario(rnd: () => number, w: ScenarioWeights = DEFAULT_SCENARIO): Scenario {
  const focus = rnd() < w.focus;
  const n = focus ? 4 : weighted(rnd, w.players as Record<number, number>);
  const ff = !focus && (n % 2 === 1 || (n >= 4 && rnd() < w.ffOnEvenTables));
  const auto = autoNumDecks(n);
  let decks = auto;
  if (!focus && rnd() < w.deckJitter) decks = Math.max(1, Math.min(2 * n, auto + (rnd() < 0.5 ? -1 : 1)));
  const maxRank: Level = !focus && rnd() < w.ntMaxRank ? 'NT' : 'A';
  const config: TractorConfig = {
    ...defaultConfig(n),
    gameMode: ff ? 'findingFriends' : 'tractor',
    numDecks: decks === auto ? null : decks,
    maxRank,
    biddingStyle: rnd() < w.turnBidding ? 'turns' : 'live',
  };
  if (rnd() < w.firstRound) {
    return { players: n, config, levels: Array<Level>(n).fill('2'), landlord: null, levelClass: 'first', focus };
  }
  const landlord = Math.floor(rnd() * n);
  const ll = landlordLevel(rnd, w, maxRank === 'NT');
  const levels: Level[] = new Array(n);
  if (!ff) {
    const other = landlordLevel(rnd, w, maxRank === 'NT').level;
    for (let i = 0; i < n; i++) levels[i] = i % 2 === landlord % 2 ? ll.level : other;
  } else {
    for (let i = 0; i < n; i++) levels[i] = i === landlord ? ll.level : near(rnd, ll.level, maxRank);
  }
  return { players: n, config, levels, landlord, levelClass: ll.cls, focus };
}
