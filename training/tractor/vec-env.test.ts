// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { CAND_DIM } from '@/ai/nn/decision';
import { GLOBAL_DIM, GLOBAL_LAYOUT, MAX_SEATS, SEAT_DIM } from '@/ai/nn/encode';
import { TractorPolicyNet } from '@/ai/nn/runtime';
import { randomPolicyFile } from '@/ai/nn/testSupport';
import { PRIV_GLOBAL_DIM, PRIV_SEAT_DIM } from './privileged';
import { TractorVecEnv, lineupRoles, mulberry32, type Episode, type VecBatch } from './vec-env';
import { DEFAULT_SCENARIO, sampleScenario } from './scenario';

function randomChoices(b: VecBatch, rnd: () => number): number[] {
  return Array.from(b.count, (k) => Math.floor(rnd() * Math.max(1, k)));
}

function checkBatch(b: VecBatch, privileged = false): void {
  const n = b.seat.length;
  let rows = 0;
  let seatRows = 0;
  for (let i = 0; i < n; i++) {
    if (b.role[i] === -1) {
      expect(b.count[i]).toBe(0);
      expect(b.seatCount[i]).toBe(0);
      continue;
    }
    expect(b.count[i]).toBeGreaterThan(0);
    expect(b.seat[i]).toBeLessThan(b.players[i]!);
    expect(b.seatCount[i]).toBe(b.players[i]);
    rows += b.count[i]!;
    seatRows += b.seatCount[i]!;
  }
  expect(b.candRows).toBe(rows);
  expect(b.cand.length).toBe(rows * CAND_DIM);
  expect(b.seatRows).toBe(seatRows);
  expect(b.seats.length).toBe(seatRows * SEAT_DIM);
  expect(b.seatSlot.length).toBe(seatRows);
  expect(b.glob.length).toBe(n * GLOBAL_DIM);
  expect(b.reward.length).toBe(n * MAX_SEATS);
  expect(b.privGlob.length).toBe(privileged ? n * PRIV_GLOBAL_DIM : 0);
  expect(b.privSeats.length).toBe(privileged ? seatRows * PRIV_SEAT_DIM : 0);
  const ok = (a: Float32Array) => a.every((x) => Number.isFinite(x) && Math.abs(x) <= 4);
  expect(ok(b.glob) && ok(b.seats) && ok(b.cand)).toBe(true);
  for (let k = 0; k < seatRows; k++) expect(b.seats[k * SEAT_DIM]).toBe(1); // present
}

function run(env: TractorVecEnv, episodes: number, opts: { maxSteps?: number; privileged?: boolean } = {}) {
  const rnd = mulberry32(5);
  let b = env.reset();
  checkBatch(b, opts.privileged);
  const eps: Episode[] = [];
  const rows = [0, 0]; // learner, opponent
  // Every reward paid to each env's seats since its episode began.
  const paid = Array.from({ length: b.seat.length }, () => new Array<number>(MAX_SEATS).fill(0));
  const totals: Array<{ ep: Episode; total: number[] }> = [];
  for (let t = 0; t < (opts.maxSteps ?? 50_000) && eps.length < episodes; t++) {
    for (let i = 0; i < b.role.length; i++) if (b.role[i]! >= 0) rows[b.role[i]!]!++;
    b = env.step(randomChoices(b, rnd));
    checkBatch(b, opts.privileged);
    for (let i = 0; i < b.done.length; i++) {
      for (let s = 0; s < MAX_SEATS; s++) paid[i]![s]! += b.reward[i * MAX_SEATS + s]!;
    }
    for (const e of b.episodes) {
      totals.push({ ep: e, total: paid[e.env]!.slice(0, e.players) });
      paid[e.env]!.fill(0);
    }
    eps.push(...b.episodes);
  }
  return { eps, rows, totals };
}

describe('scenario sampler', () => {
  it('plays mostly 4-player Tractor, plus a variety of every other table', () => {
    const rnd = mulberry32(1);
    const seen = new Set<string>();
    let focus = 0;
    const N = 6000;
    for (let k = 0; k < N; k++) {
      const s = sampleScenario(rnd);
      expect(s.levels.length).toBe(s.players);
      if (s.focus) {
        focus++;
        expect(s.players).toBe(4);
        expect(s.config.gameMode).toBe('tractor');
        expect(s.config.numDecks).toBeNull();
        expect(s.config.maxRank).toBe('A');
      }
      if (s.players % 2 === 1) expect(s.config.gameMode).toBe('findingFriends');
      seen.add(`p${s.players}`);
      seen.add(`${s.players}${s.config.gameMode}`);
      seen.add(s.levelClass);
      seen.add(s.config.biddingStyle);
      if (s.config.gameMode === 'tractor' && s.landlord !== null) {
        for (let i = 0; i < s.players; i++) expect(s.levels[i]).toBe(s.levels[(i + 2) % s.players]);
      }
    }
    expect(focus / N).toBeGreaterThan(0.82);
    expect(focus / N).toBeLessThan(0.88);
    for (let p = 2; p <= 12; p++) expect(seen.has(`p${p}`)).toBe(true);
    for (const k of ['4findingFriends', '6findingFriends', '6tractor', 'point', 'ace', 'other', 'first', 'live', 'turns']) expect(seen.has(k)).toBe(true);
  });

  it('can give only the focus table, or only the variety', () => {
    const rnd = mulberry32(4);
    for (let k = 0; k < 300; k++) expect(sampleScenario(rnd, { ...DEFAULT_SCENARIO, focus: 1 }).focus).toBe(true);
    for (let k = 0; k < 300; k++) expect(sampleScenario(rnd, { ...DEFAULT_SCENARIO, focus: 0 }).focus).toBe(false);
  });

  it('seats lineups as asked', () => {
    const rnd = mulberry32(2);
    expect(lineupRoles(rnd, 6, { weight: 1, learner: 'all' })).toEqual(Array(6).fill('learner'));
    const one = lineupRoles(rnd, 5, { weight: 1, learner: 'one' });
    expect(one.filter((r) => r === 'learner').length).toBe(1);
    const team = lineupRoles(rnd, 6, { weight: 1, learner: 'team' });
    expect(team.filter((r) => r === 'learner').length).toBe(3);
    const at = team.indexOf('learner');
    expect(team[(at + 2) % 6]).toBe('learner');
  });
});

describe('TractorVecEnv', () => {
  it('plays rounds to the end with random choices, asking learner and opponent seats', () => {
    const env = new TractorVecEnv({ numEnvs: 6, seed: 3, lineups: [{ weight: 1, learner: 'team' }], scenario: { focus: 0.5 } });
    const { eps, rows } = run(env, 24);
    expect(rows[0]).toBeGreaterThan(0);
    expect(rows[1]).toBeGreaterThan(0);
    expect(eps.length).toBeGreaterThanOrEqual(24);
    for (const e of eps) {
      expect(e.aborted).toBe(false);
      expect(e.truncated).toBe(false);
      expect(e.landlordWon).not.toBeNull();
      expect(e.decisions.reduce((a, x) => a + x, 0)).toBeGreaterThan(0);
      // Zero-sum between the sides, seat by seat.
      const sides = e.reward.map((r, s) => (e.landlordTeam.includes(s) ? r : -r));
      for (const x of sides) expect(x).toBeCloseTo(sides[0]!, 5);
    }
    expect(new Set(eps.map((e) => e.players)).size).toBeGreaterThan(2);
  }, 120_000);

  it('pays point shaping that sums to zero over each round', () => {
    const env = new TractorVecEnv({ numEnvs: 4, seed: 8, pointsShaping: 0.5, lineups: [{ weight: 1, learner: 'all' }] });
    const { totals } = run(env, 12);
    expect(totals.length).toBeGreaterThanOrEqual(12);
    for (const { ep, total } of totals) total.forEach((x, s) => expect(x).toBeCloseTo(ep.reward[s]!, 4));
  }, 120_000);

  it('pairs duplicate deals with the teams swapped', () => {
    const env = new TractorVecEnv({
      numEnvs: 2, seed: 4, quota: 4, duplicate: true,
      lineups: [{ weight: 1, learner: 'team' }], scenario: { focus: 1 },
    });
    const { eps } = run(env, 100);
    expect(eps.length).toBe(8);
    for (let env = 0; env < 2; env++) {
      const mine = eps.filter((e) => e.env === env).sort((a, b) => a.pair - b.pair || a.flip - b.flip);
      for (let k = 0; k < mine.length; k += 2) {
        const [a, b] = [mine[k]!, mine[k + 1]!];
        expect([a.pair, a.flip, b.pair, b.flip]).toEqual([b.pair, 0, a.pair, 1]);
        expect(a.seed).toBe(b.seed);
        expect(a.roles.includes('opponent')).toBe(true);
        expect(b.roles).toEqual(a.roles.map((_, s) => a.roles[(s + 3) % 4]));
      }
    }
  }, 120_000);

  it('sends privileged features that agree with the observation', () => {
    const env = new TractorVecEnv({ numEnvs: 4, seed: 6, privileged: true, lineups: [{ weight: 1, learner: 'all' }], scenario: { focus: 1 } });
    let b = env.reset();
    const rnd = mulberry32(3);
    const hand = GLOBAL_LAYOUT.segments.find((s) => s.name === 'hand')!.offset;
    let checked = 0;
    for (let t = 0; t < 400; t++) {
      checkBatch(b, true);
      let at = 0;
      for (let i = 0; i < b.seat.length; i++) {
        const k = b.seatCount[i]!;
        // Outside the exchange, my own hand (slot 0) is the observation's hand plane.
        if (k > 0 && b.kind[i] === 3) {
          const slot0 = Array.from(b.seatSlot.subarray(at, at + k)).indexOf(0);
          const priv = b.privSeats.subarray((at + slot0) * PRIV_SEAT_DIM, (at + slot0 + 1) * PRIV_SEAT_DIM);
          const obs = b.glob.subarray(i * GLOBAL_DIM + hand, i * GLOBAL_DIM + hand + PRIV_SEAT_DIM);
          expect(Array.from(priv)).toEqual(Array.from(obs));
          checked++;
        }
        at += k;
      }
      b = env.step(randomChoices(b, rnd));
    }
    expect(checked).toBeGreaterThan(100);
  }, 120_000);

  it('runs turn bidding, then idles on quota', () => {
    const env = new TractorVecEnv({
      numEnvs: 3, seed: 7, quota: 2,
      lineups: [{ weight: 1, learner: 'one' }],
      scenario: { turnBidding: 1, focus: 0.5 },
    });
    const { eps } = run(env, 100, { maxSteps: 20_000 });
    expect(eps.length).toBe(6);
    expect(eps.some((e) => e.roles.includes('opponent'))).toBe(true);
  }, 120_000);

  it('plays whole matches from 2 in match mode', () => {
    const env = new TractorVecEnv({
      numEnvs: 2, seed: 5, quota: 1, matches: true,
      lineups: [{ weight: 1, learner: 'team' }], scenario: { focus: 1 },
    });
    const { eps } = run(env, 2, { maxSteps: 200_000 });
    expect(eps.length).toBe(2);
    for (const e of eps) {
      expect(e.aborted).toBe(false);
      expect(e.truncated).toBe(false);
      expect(e.levelClass).toBe('match');
      expect(e.rounds).toBeGreaterThan(5);
      expect(e.finalLevels.length).toBe(4);
      expect(e.winners.length).toBe(2);
      expect(e.winners[0]! % 2).toBe(e.winners[1]! % 2); // a Tractor team wins together
    }
  }, 120_000);

  it('measures each seat’s skills at the end of a round', () => {
    const env = new TractorVecEnv({ numEnvs: 4, seed: 13, lineups: [{ weight: 1, learner: 'all' }], scenario: { focus: 1 } });
    const { eps } = run(env, 8);
    expect(eps.length).toBeGreaterThanOrEqual(8);
    for (const e of eps) {
      const k = e.skills;
      const side = (s: number) => (e.landlordTeam.includes(s) ? 'D' : 'A');
      for (let s = 0; s < e.players; s++) {
        expect(k.throwFails[s]).toBeLessThanOrEqual(k.throws[s]!);
        expect(k.kittyBuried[s]! > 0 ? s === e.landlord : true).toBe(true);
      }
      // One side won the last trick: same value across a side, the other side the opposite.
      const bySide = new Map<string, Set<number>>();
      k.lastTrick.forEach((x, s) => bySide.set(side(s), (bySide.get(side(s)) ?? new Set()).add(x)));
      for (const v of bySide.values()) expect(v.size).toBe(1);
      expect(new Set([...bySide.values()].map((v) => [...v][0])).size).toBe(bySide.size);
    }
  }, 120_000);

  it('reports the TypeScript network’s logits for every pending decision', () => {
    const env = new TractorVecEnv({ numEnvs: 3, seed: 9 });
    const b = env.reset();
    const logits = env.policyLogits(new TractorPolicyNet(randomPolicyFile(3)));
    expect(logits.map((l) => l.length)).toEqual(Array.from(b.count));
  });
});
