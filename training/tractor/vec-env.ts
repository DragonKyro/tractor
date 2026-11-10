// A batch of independent Tractor rounds for a learner to drive.
//
// Every episode is one round of a match, set up by the curriculum
// (scenario.ts: table size, mode, decks, levels, landlord), so a network meets
// every rank situation without playing a match up from 2. Every seat is played
// by a network in the learner process, in one of two roles:
//   learner    the network being trained
//   opponent   a frozen network (a past snapshot, or the anchor)
// Which seats are which comes from a weighted list of lineups, sampled per
// episode.
//
// Each env runs on its own — the deal, steps with nothing to choose — until a
// seat has a decision, then reports that seat's encoded observation and its
// candidate list; the learner process answers every env with one candidate
// index. Decisions:
//
//   bid   live deal: a network seat is asked whenever its legal bids change
//         (a card arrives, someone bids); "no bid" means "not now". Turn
//         bidding: the seat whose turn it is (pass or a bid)
//   bury  the exchange, one kitty card at a time …
//   call  … then (Finding Friends) one friend call at a time; nothing changes
//         in the game until the whole exchange is chosen
//   play  a lead or a follow
//
// Steps with nothing to choose (taking the kitty, turning it over, redeal,
// single-candidate follows) are played automatically. A finished round
// auto-resets: `done[i]` marks that env i's episode ended during the last
// advance, and the row returned is already its next episode's first decision.
// With `quota` set, an env stops after that many episodes and reports role −1
// (no candidates; any answer is ignored).
//
// Observations go out packed: each row's global block, then only its table's
// seat blocks (with their slot), in env order — 4 seat blocks for a 4-player
// table rather than 12. With `privileged`, each row also carries what only the
// critic may see (privileged.ts), in the same packing.
//
// Rewards (reward.ts) are per seat, numEnvs × MAX_SEATS, paid at the round's
// end: credit `reward[i·MAX_SEATS + s]` to seat s's latest decision in env i.
// With `pointsShaping` w > 0, each seat is also paid as points are captured:
// potential-based shaping on Φ = ±w × attackers' points / threshold (+ for the
// attackers, − for the landlord's team as currently known), with Φ taken back
// at the round's end, so the shaping sums to zero over a round and only moves
// credit earlier, toward the tricks that earned it.
//
// With `duplicate`, episodes come in pairs (2k, 2k+1) on the same env: the
// same deal (seed, scenario, lineup), the second with every role moved one
// seat on — in a 4-player table the two teams swap cards. Comparing a pair
// cancels the luck of the deal (evaluation).
//
// With `matches` on (evaluation), an episode is a whole match instead: it
// starts with everyone at 2 and plays round after round (rewards still paid
// per round) until someone passes the final rank, or `maxRounds`.
//
// Fault tolerance: an error abandons just that round (`Episode.aborted`, no
// reward) and starts a new one; it goes to stderr, throttled.

import type { Level } from '@/game/cards';
import { defaultConfig } from '@/game/config';
import { activeSeat, applyAction, attackerPointsSoFar, createMatchAt, validBidsFor } from '@/game/engine';
import type { Trump } from '@/game/trump';
import type { Action, TractorState } from '@/game/types';
import { observe, type Observation } from '@/ai/observation';
import { aiSeats } from '@/ai/simulate';
import { repairExchange } from '@/ai/nn/agent';
import { CAND_DIM, MAX_CANDIDATES, applyDraft, emptyDraft, nextStep, type Decision, type DecisionKind, type Draft } from '@/ai/nn/decision';
import { GLOBAL_DIM, MAX_SEATS, OBS_DIM, SEAT_DIM, encodeObservation, threshold } from '@/ai/nn/encode';
import type { TractorPolicyNet } from '@/ai/nn/runtime';
import { PRIV_GLOBAL_DIM, PRIV_SEAT_DIM, encodePrivileged } from './privileged';
import { skillStats, type Skill } from './skills';
import { DEFAULT_REWARD, roundRewards, sideGain, type RewardWeights } from './reward';
import { DEFAULT_SCENARIO, sampleScenario, type Scenario, type ScenarioWeights } from './scenario';

export type SeatRole = 'learner' | 'opponent';
const ROLE_CODE: Record<SeatRole, number> = { learner: 0, opponent: 1 };
export const KIND_CODE: Record<DecisionKind, number> = { bid: 0, bury: 1, call: 2, play: 3 };

/**
 * Which seats are the learner, for one episode (the rest are `opponent`):
 * 'all' every seat · 'one' a random seat · 'team' every other seat from a
 * random start (in Tractor that is one fixed-partner team).
 */
export interface Lineup {
  weight: number;
  learner: 'all' | 'one' | 'team';
}

export interface VecEnvOptions {
  numEnvs: number;
  seed: number;
  lineups: Lineup[];
  /** Curriculum weights (merged over DEFAULT_SCENARIO). */
  scenario: Partial<ScenarioWeights>;
  reward: RewardWeights;
  /** Episodes each env plays before going idle (0 = forever). */
  quota: number;
  /** Rows also carry the critic's privileged features. */
  privileged: boolean;
  /** Potential-based shaping weight on the attackers' points (0 = off). */
  pointsShaping: number;
  /** Episodes in pairs: the same deal, roles moved one seat on. */
  duplicate: boolean;
  /** Actions (any seat, draws included) before a round is cut off. */
  maxSteps: number;
  /** Episodes are whole matches from 2 (evaluation), not single rounds. */
  matches: boolean;
  /** Rounds before a match is cut off. */
  maxRounds: number;
}

export const DEFAULT_VEC_ENV_OPTIONS: VecEnvOptions = {
  numEnvs: 16,
  seed: 1,
  lineups: [{ weight: 1, learner: 'all' }],
  scenario: {},
  reward: DEFAULT_REWARD,
  quota: 0,
  privileged: false,
  pointsShaping: 0,
  duplicate: false,
  maxSteps: 20_000,
  matches: false,
  maxRounds: 200,
};

export interface Episode {
  env: number;
  seed: number;
  /** Duplicate pairs: the pair's index and which half (0 / 1); otherwise the episode index and 0. */
  pair: number;
  flip: number;
  players: number;
  mode: 'tractor' | 'findingFriends';
  decks: number;
  /** 'point' | 'ace' | 'other' | 'first' | 'match' (scenario.ts). */
  levelClass: string;
  /** A focus-table round (4-player Tractor) rather than the variety mix. */
  focus: boolean;
  landlordLevel: string;
  /** Its lineup's `learner` ('all' | 'one' | 'team'). */
  lineup: string;
  roles: SeatRole[];
  landlord: number | null;
  landlordTeam: number[];
  landlordWon: boolean | null;
  attackerPoints: number | null;
  threshold: number;
  /** Per seat: the reward paid at the end (shaping excluded), and levels my side gained minus the other side's. */
  reward: number[];
  gain: number[];
  /** Network decisions per seat (single-candidate ones excluded: the env plays those itself, `autoplays`). */
  decisions: number[];
  steps: number;
  autoplays: number;
  truncated: boolean;
  aborted: boolean;
  /** Rounds played (1 unless `matches`). */
  rounds: number;
  /** Each seat's level at the end, and who passed the final rank (the match winners). */
  finalLevels: string[];
  winners: number[];
  /** Per-seat skill measures of the (last) round (skills.ts). */
  skills: Record<Skill, number[]>;
}

/** Buffers are reused by the next reset / step — copy anything you keep. */
export interface VecBatch {
  /** Seat deciding row i. */
  seat: Int8Array;
  /** 0 learner, 1 opponent, −1 idle (quota reached). */
  role: Int8Array;
  /** KIND_CODE, −1 idle. */
  kind: Int8Array;
  /** Players at env i's table. */
  players: Int8Array;
  /** numEnvs × GLOBAL_DIM. */
  glob: Float32Array;
  /** Seat blocks per row (the table's players; 0 when idle). */
  seatCount: Int16Array;
  /** Their slots (0..MAX_SEATS-1), packed in row order (a view; `seatRows` long). */
  seatSlot: Int8Array;
  /** seatRows × SEAT_DIM (a view). */
  seats: Float32Array;
  seatRows: number;
  /** Privileged (critic-only) features: numEnvs × PRIV_GLOBAL_DIM, seatRows × PRIV_SEAT_DIM (empty unless `privileged`). */
  privGlob: Float32Array;
  privSeats: Float32Array;
  /** Candidates per row (0 when idle). */
  count: Int16Array;
  /** sum(count) × CAND_DIM, rows in env order (a view; see `candRows`). */
  cand: Float32Array;
  candRows: number;
  /** numEnvs × MAX_SEATS. */
  reward: Float32Array;
  done: Uint8Array;
  episodes: Episode[];
}

interface Pending {
  seat: number;
  obs0: Observation;
  draft: Draft;
  decision: Decision;
}

interface EnvSlot {
  state: TractorState;
  seed: number;
  episode: number;
  pair: number;
  flip: number;
  scenario: Scenario;
  lineup: string;
  roles: SeatRole[];
  steps: number;
  /** Steps in the current round (`maxSteps` is per round). */
  roundSteps: number;
  rounds: number;
  decisions: number[];
  autoplays: number;
  /** Live deal: the legal-bid set each seat was last asked about. */
  bidKey: string[];
  /** Shaping: each seat's current potential, and the round's threshold. */
  phi: number[];
  thr: number;
  pending: Pending | null;
  /** The pending decision's encoding (dense, OBS_DIM) and privileged features. */
  obs: Float32Array;
  privGlob: Float32Array;
  privSeats: Float32Array;
  idle: boolean;
}

/** stderr, at most a few lines per distinct problem (then every 100th). */
const reported = new Map<string, number>();
function report(kind: string, message: string): void {
  const n = (reported.get(kind) ?? 0) + 1;
  reported.set(kind, n);
  if (n <= 3 || n % 100 === 0) process.stderr.write(`[tractor-env] ${kind} (×${n}): ${message}\n`);
}

export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function lineupRoles(rnd: () => number, n: number, lineup: Lineup): SeatRole[] {
  const roles: SeatRole[] = [];
  const start = Math.floor(rnd() * n);
  for (let s = 0; s < n; s++) {
    const learner = lineup.learner === 'all' || (lineup.learner === 'one' ? s === start : (s - start + n) % 2 === 0);
    roles.push(learner ? 'learner' : 'opponent');
  }
  return roles;
}

function validateLineups(lineups: Lineup[]): void {
  if (!Array.isArray(lineups) || lineups.length === 0) throw new Error('lineups must be a non-empty list');
  for (const l of lineups) {
    if (!(l.weight >= 0)) throw new Error(`lineup weight ${l.weight}`);
    if (!['all', 'one', 'team'].includes(l.learner)) throw new Error(`lineup learner ${JSON.stringify(l.learner)}`);
  }
  if (!lineups.some((l) => l.weight > 0)) throw new Error('every lineup has weight 0');
}

/** The round's trump number ('NT' for a no-trump level), or the scenario's landlord level before trump is set. */
function trumpNumber(t: Trump | null, sc: Scenario): string {
  if (t) return t.number ?? 'NT';
  return String(sc.landlord !== null ? sc.levels[sc.landlord] : '2');
}

function roundThreshold(state: TractorState): number {
  return threshold(observe(state, 0));
}

const bidKeyOf = (state: TractorState, seat: number) =>
  validBidsFor(state, seat).map((b) => `${b.card}${b.count}`).join(',');

export class TractorVecEnv {
  readonly opts: VecEnvOptions;
  private readonly envs: EnvSlot[] = [];
  private readonly batch: VecBatch;
  private readonly candBuf: Float32Array;
  private readonly seatBuf: Float32Array;
  private readonly slotBuf: Int8Array;
  private readonly privSeatBuf: Float32Array;

  constructor(options: Partial<VecEnvOptions> = {}) {
    this.opts = { ...DEFAULT_VEC_ENV_OPTIONS, ...options };
    const n = this.opts.numEnvs;
    if (!(Number.isInteger(n) && n >= 1)) throw new Error('numEnvs must be a positive integer');
    validateLineups(this.opts.lineups);
    this.candBuf = new Float32Array(n * MAX_CANDIDATES * CAND_DIM);
    this.seatBuf = new Float32Array(n * MAX_SEATS * SEAT_DIM);
    this.slotBuf = new Int8Array(n * MAX_SEATS);
    const priv = this.opts.privileged;
    this.privSeatBuf = new Float32Array(priv ? n * MAX_SEATS * PRIV_SEAT_DIM : 0);
    this.batch = {
      seat: new Int8Array(n),
      role: new Int8Array(n),
      kind: new Int8Array(n),
      players: new Int8Array(n),
      glob: new Float32Array(n * GLOBAL_DIM),
      seatCount: new Int16Array(n),
      seatSlot: this.slotBuf.subarray(0, 0),
      seats: this.seatBuf.subarray(0, 0),
      seatRows: 0,
      privGlob: new Float32Array(priv ? n * PRIV_GLOBAL_DIM : 0),
      privSeats: this.privSeatBuf.subarray(0, 0),
      count: new Int16Array(n),
      cand: this.candBuf.subarray(0, 0),
      candRows: 0,
      reward: new Float32Array(n * MAX_SEATS),
      done: new Uint8Array(n),
      episodes: [],
    };
  }

  /** New lineups / curriculum for episodes that start from now on. */
  configure(update: { lineups?: Lineup[]; scenario?: Partial<ScenarioWeights> }): void {
    if (update.lineups) {
      validateLineups(update.lineups);
      this.opts.lineups = update.lineups.map((l) => ({ ...l }));
    }
    if (update.scenario) this.opts.scenario = { ...update.scenario };
  }

  reset(): VecBatch {
    this.beginBatch();
    this.envs.length = 0;
    for (let i = 0; i < this.opts.numEnvs; i++) {
      this.envs.push(this.startEpisode(i, 0));
      this.guarded(i, () => this.advance(i));
    }
    return this.finishBatch();
  }

  /** One candidate index per env, for the decision reported in the previous batch (ignored for idle envs). */
  step(choices: ArrayLike<number>): VecBatch {
    if (this.envs.length === 0) throw new Error('step before reset');
    if (choices.length !== this.envs.length) throw new Error(`expected ${this.envs.length} choices, got ${choices.length}`);
    this.beginBatch();
    for (let i = 0; i < this.envs.length; i++) {
      const env = this.envs[i]!;
      if (env.idle) continue;
      this.guarded(i, () => {
        const p = env.pending;
        if (!p) throw new Error('no pending decision');
        const k = Math.max(0, Math.min(p.decision.candidates.length - 1, Math.floor(Number(choices[i]) || 0)));
        env.decisions[p.seat]!++;
        if (this.choose(i, p, k)) return;
        this.advance(i);
      });
    }
    return this.finishBatch();
  }

  /** The TypeScript runtime's logits for every env's pending decision ([] if idle) — parity checks against PyTorch. */
  policyLogits(net: TractorPolicyNet): number[][] {
    return this.envs.map((env) => {
      const p = env.pending;
      if (env.idle || !p) return [];
      return Array.from(net.logits(env.obs, p.decision.candidates.map((c) => c.features)));
    });
  }

  // ------------------------------------------------------------ episodes

  private guarded(i: number, f: () => void, recover: () => void = () => this.advance(i)): void {
    for (let attempt = 0; ; attempt++) {
      try {
        f();
        return;
      } catch (e) {
        if (attempt >= 3) throw e; // a fresh round failing over and over is a real bug
        const env = this.envs[i]!;
        report('round aborted', `env ${i} seed ${env.seed} stage ${env.state.round.stage}: ${(e as Error).stack ?? e}`);
        this.endEpisode(i, { aborted: true, truncated: false });
        this.envs[i] = this.startEpisode(i, env.episode + 1);
        f = recover;
      }
    }
  }

  private startEpisode(i: number, episode: number): EnvSlot {
    const dup = this.opts.duplicate;
    const pair = dup ? episode >> 1 : episode;
    const flip = dup ? episode & 1 : 0;
    const seed = (Math.imul(this.opts.seed, 1_000_003) + Math.imul(i, 7919) + Math.imul(pair, 104_729)) >>> 0;
    const rng = mulberry32(seed ^ 0x9e3779b9);
    const weights = { ...DEFAULT_SCENARIO, ...this.opts.scenario };
    for (let tries = 0; ; tries++) {
      let scenario = sampleScenario(rng, weights);
      const n = scenario.players;
      if (this.opts.matches) scenario = { ...scenario, levels: Array<Level>(n).fill('2'), landlord: null, levelClass: 'match' };
      try {
        const state = createMatchAt({ ...defaultConfig(n), ...scenario.config }, seed + tries, aiSeats(n), {
          levels: scenario.levels,
          landlord: scenario.landlord,
        });
        const lineup = this.pickLineup(rng);
        let roles = lineupRoles(rng, n, lineup);
        // The pair's second half: every role one seat on (the same cards, the other team holding them).
        if (flip) roles = roles.map((_, s) => roles[(s - 1 + n) % n]!);
        return {
          state,
          seed: seed + tries,
          episode,
          pair,
          flip,
          scenario,
          lineup: lineup.learner,
          roles,
          steps: 0,
          roundSteps: 0,
          rounds: 1,
          decisions: new Array<number>(n).fill(0),
          autoplays: 0,
          bidKey: new Array<string>(n).fill(''),
          phi: new Array<number>(n).fill(0),
          thr: roundThreshold(state),
          pending: null,
          obs: this.envs[i]?.obs ?? new Float32Array(OBS_DIM),
          privGlob: this.envs[i]?.privGlob ?? new Float32Array(this.opts.privileged ? PRIV_GLOBAL_DIM : 0),
          privSeats: this.envs[i]?.privSeats ?? new Float32Array(this.opts.privileged ? MAX_SEATS * PRIV_SEAT_DIM : 0),
          idle: false,
        };
      } catch (e) {
        if (tries >= 20) throw e;
        report('scenario rejected', `${n} players ${JSON.stringify(scenario.config.gameMode)}: ${(e as Error).message}`);
      }
    }
  }

  private pickLineup(rnd: () => number): Lineup {
    const ls = this.opts.lineups;
    const total = ls.reduce((a, l) => a + l.weight, 0);
    let r = rnd() * total;
    for (const l of ls) {
      r -= l.weight;
      if (r <= 0 && l.weight > 0) return l;
    }
    return ls.find((l) => l.weight > 0)!;
  }

  /** Shaping: take back each seat's potential (end of a round). */
  private settleShaping(i: number): void {
    const env = this.envs[i]!;
    for (let k = 0; k < env.phi.length; k++) {
      this.batch.reward[i * MAX_SEATS + k]! -= env.phi[k]!;
      env.phi[k] = 0;
    }
  }

  private endEpisode(i: number, how: { aborted: boolean; truncated: boolean }): void {
    const env = this.envs[i]!;
    const s = env.state;
    const n = s.seats.length;
    const result = s.round.result;
    const paid = !how.aborted && !how.truncated && result ? roundRewards(s, this.opts.reward) : new Array<number>(n).fill(0);
    for (let k = 0; k < n; k++) this.batch.reward[i * MAX_SEATS + k]! += paid[k]!;
    this.settleShaping(i);
    this.batch.done[i] = 1;
    const sc = env.scenario;
    this.batch.episodes.push({
      env: i,
      seed: env.seed,
      pair: env.pair,
      flip: env.flip,
      players: n,
      mode: s.rules.gameMode,
      decks: s.rules.numDecks,
      levelClass: sc.levelClass,
      focus: sc.focus,
      landlordLevel: trumpNumber(result?.trump ?? s.round.trump, sc),
      lineup: env.lineup,
      roles: env.roles.slice(),
      landlord: result?.landlord ?? s.round.landlord,
      landlordTeam: result?.landlordTeam.slice() ?? s.round.landlordTeam.slice(),
      landlordWon: result ? result.landlordWon : null,
      attackerPoints: result ? result.attackerPoints : null,
      threshold: env.thr,
      reward: paid,
      gain: s.seats.map((_, k) => (result && !how.aborted ? sideGain(s, k) : 0)),
      decisions: env.decisions.slice(),
      steps: env.steps,
      autoplays: env.autoplays,
      truncated: how.truncated,
      aborted: how.aborted,
      rounds: env.rounds,
      finalLevels: s.levels.map(String),
      winners: result && !how.aborted ? result.winners.slice() : [],
      skills: skillStats(s),
    });
  }

  // ------------------------------------------------------------ playing

  private apply(i: number, action: Action): void {
    const env = this.envs[i]!;
    env.state = applyAction(env.state, action);
    env.steps++;
    env.roundSteps++;
    if (action.type === 'play' && this.opts.pointsShaping > 0) this.shape(i);
  }

  /** Pay each seat the change in its potential ±w × attackers' points / threshold. */
  private shape(i: number): void {
    const env = this.envs[i]!;
    const s = env.state;
    const pts = attackerPointsSoFar(s) / env.thr;
    const team = s.round.landlordTeam;
    for (let k = 0; k < env.phi.length; k++) {
      const phi = this.opts.pointsShaping * (team.includes(k) ? -pts : pts);
      this.batch.reward[i * MAX_SEATS + k]! += phi - env.phi[k]!;
      env.phi[k] = phi;
    }
  }

  /** A step with nothing to choose; a finished exchange gets legal friend calls if its own break a rule. */
  private applyForced(i: number, a: Action): void {
    this.apply(i, repairExchange(this.envs[i]!.state, a));
  }

  /**
   * Play candidate k of decision p. A bury / call is one part of the exchange:
   * the next part becomes pending (true), or the finished exchange is applied.
   * "No bid" does nothing (the bid key is already recorded).
   */
  private choose(i: number, p: Pending, k: number): boolean {
    const env = this.envs[i]!;
    const pick = p.decision.candidates[k]!.pick;
    env.pending = null;
    if ('action' in pick) {
      this.apply(i, pick.action);
      return false;
    }
    if ('none' in pick) return false;
    if ('bury' in pick) p.draft.buried.push(pick.bury);
    else p.draft.calls.push(pick.call);
    const st = nextStep(p.obs0, p.draft);
    if (!st) throw new Error('the exchange stopped short');
    if ('forced' in st) {
      this.applyForced(i, st.forced);
      return false;
    }
    return this.offer(i, { ...p, decision: st.decision });
  }

  /** Make p env i's pending decision — or, with a single candidate (a forced follow), just play it. True when one is pending. */
  private offer(i: number, p: Pending): boolean {
    if (p.decision.candidates.length === 1 && p.decision.kind !== 'bid') {
      this.envs[i]!.autoplays++;
      return this.choose(i, p, 0);
    }
    this.envs[i]!.pending = p;
    return true;
  }

  /** Live deal: ask the first seat whose legal bids changed since it was last asked. */
  private askBid(i: number): boolean {
    const env = this.envs[i]!;
    const s = env.state;
    for (let seat = 0; seat < s.seats.length; seat++) {
      const key = bidKeyOf(s, seat);
      if (key === env.bidKey[seat]) continue;
      env.bidKey[seat] = key;
      if (!key) continue;
      const obs0 = observe(s, seat);
      const st = nextStep(obs0);
      if (st && 'decision' in st && st.decision.kind === 'bid') {
        return this.offer(i, { seat, obs0, draft: emptyDraft(), decision: st.decision });
      }
    }
    return false;
  }

  /** Run env i forward to the next decision, resetting finished rounds. */
  private advance(i: number): void {
    for (;;) {
      const env = this.envs[i]!;
      if (env.idle || env.pending) return;
      const s = env.state;
      const over = s.round.stage === 'roundOver' || s.status !== 'playing';
      if (over && this.opts.matches && s.status === 'playing' && env.rounds < this.opts.maxRounds) {
        const paid = roundRewards(s, this.opts.reward);
        for (let k = 0; k < paid.length; k++) this.batch.reward[i * MAX_SEATS + k]! += paid[k]!;
        this.settleShaping(i);
        this.apply(i, { type: 'nextRound' });
        env.rounds++;
        env.roundSteps = 0;
        env.bidKey.fill('');
        env.thr = roundThreshold(env.state);
        continue;
      }
      if (over || env.roundSteps >= this.opts.maxSteps) {
        const cut = !over || (this.opts.matches && s.status === 'playing');
        if (!over) report('round truncated', `env ${i} seed ${env.seed} at ${s.round.stage} after ${env.roundSteps} steps`);
        this.endEpisode(i, { aborted: false, truncated: cut });
        if (this.opts.quota > 0 && env.episode + 1 >= this.opts.quota) {
          env.idle = true;
          return;
        }
        this.envs[i] = this.startEpisode(i, env.episode + 1);
        continue;
      }
      const r = s.round;
      if (r.stage === 'draw') {
        if (this.askBid(i)) return;
        if (r.deck.length > 0) {
          this.apply(i, { type: 'draw', seat: r.dealTo });
          continue;
        }
      }
      const seat = activeSeat(s);
      if (seat === null) throw new Error(`nobody to act at ${r.stage}`);
      const obs0 = observe(s, seat);
      const st = nextStep(obs0);
      if (!st) throw new Error(`seat ${seat} has nothing to do at ${r.stage}`);
      if ('forced' in st) {
        this.applyForced(i, st.forced);
        continue;
      }
      if (this.offer(i, { seat, obs0, draft: emptyDraft(), decision: st.decision })) return;
    }
  }

  // ------------------------------------------------------------ batches

  private beginBatch(): void {
    this.batch.reward.fill(0);
    this.batch.done.fill(0);
    this.batch.episodes = [];
  }

  private finishBatch(): VecBatch {
    const b = this.batch;
    for (let i = 0; i < this.envs.length; i++) {
      this.guarded(i, () => this.writeRow(i), () => {
        this.advance(i);
        this.writeRow(i);
      });
    }
    // Seat blocks and candidates go in env order, packed.
    let seatRows = 0;
    let candRows = 0;
    const priv = this.opts.privileged;
    for (let i = 0; i < this.envs.length; i++) {
      const env = this.envs[i]!;
      const p = env.pending;
      b.seatCount[i] = 0;
      if (env.idle || !p) continue;
      for (let slot = 0; slot < MAX_SEATS; slot++) {
        const o = GLOBAL_DIM + slot * SEAT_DIM;
        if (env.obs[o]! < 0.5) continue; // absent seat
        this.seatBuf.set(env.obs.subarray(o, o + SEAT_DIM), seatRows * SEAT_DIM);
        this.slotBuf[seatRows] = slot;
        if (priv) this.privSeatBuf.set(env.privSeats.subarray(slot * PRIV_SEAT_DIM, (slot + 1) * PRIV_SEAT_DIM), seatRows * PRIV_SEAT_DIM);
        seatRows++;
        b.seatCount[i]!++;
      }
      for (const c of p.decision.candidates) this.candBuf.set(c.features, (candRows++) * CAND_DIM);
    }
    b.seatRows = seatRows;
    b.seatSlot = this.slotBuf.subarray(0, seatRows);
    b.seats = this.seatBuf.subarray(0, seatRows * SEAT_DIM);
    b.privSeats = this.privSeatBuf.subarray(0, priv ? seatRows * PRIV_SEAT_DIM : 0);
    b.candRows = candRows;
    b.cand = this.candBuf.subarray(0, candRows * CAND_DIM);
    return b;
  }

  /** Encode env i's pending decision into row i of the batch. */
  private writeRow(i: number): void {
    const b = this.batch;
    const env = this.envs[i]!;
    b.players[i] = env.state.seats.length;
    const p = env.pending;
    if (env.idle || !p) {
      if (!env.idle) throw new Error('no pending decision after advance');
      b.seat[i] = 0;
      b.role[i] = -1;
      b.kind[i] = -1;
      b.count[i] = 0;
      b.glob.fill(0, i * GLOBAL_DIM, (i + 1) * GLOBAL_DIM);
      if (this.opts.privileged) b.privGlob.fill(0, i * PRIV_GLOBAL_DIM, (i + 1) * PRIV_GLOBAL_DIM);
      return;
    }
    const cands = p.decision.candidates;
    if (cands.length === 0 || cands.length > MAX_CANDIDATES) throw new Error(`${cands.length} candidates`);
    b.seat[i] = p.seat;
    b.role[i] = ROLE_CODE[env.roles[p.seat]!];
    b.kind[i] = KIND_CODE[p.decision.kind];
    b.count[i] = cands.length;
    encodeObservation(applyDraft(p.obs0, p.draft), env.obs);
    b.glob.set(env.obs.subarray(0, GLOBAL_DIM), i * GLOBAL_DIM);
    if (this.opts.privileged) {
      encodePrivileged(env.state, p.seat, env.privGlob, env.privSeats);
      b.privGlob.set(env.privGlob, i * PRIV_GLOBAL_DIM);
    }
  }
}
