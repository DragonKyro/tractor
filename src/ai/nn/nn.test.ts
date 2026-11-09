import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { RANKS, SUITS, buildDecks, type Card, type Suit } from '@/game/cards';
import { defaultConfig, type TractorConfig } from '@/game/config';
import { activeSeat, applyAction, createMatchAt, validBidsFor } from '@/game/engine';
import type { Trump } from '@/game/trump';
import type { TractorState } from '@/game/types';
import { nextAutomatedStep } from '../driver';
import { chooseAIAction } from '../index';
import { NN_LEVELS, setNnLevel } from './levels';
import { observe } from '../observation';
import { aiSeats } from '../simulate';
import { actWithPolicy, legalFriendCalls, netPolicy, repairExchange, teacherIndex, type BidMemo, type CandidatePolicy } from './agent';
import { CAND_DIM, emptyDraft, nextStep } from './decision';
import { OBS_DIM, encodeObservation } from './encode';
import { TractorPolicyNet, decodeTensor, halfToFloat, type TractorPolicyFile } from './runtime';
import { CARD_SLOTS, cardAtSlot, slotOf } from './slots';
import { floatToHalf, randomPolicyFile, toHalfPolicyFile } from './testSupport';

const trumps: Trump[] = [
  { kind: 'standard', suit: 'H', number: '7' },
  { kind: 'standard', suit: 'S', number: 'A' },
  { kind: 'noTrump', number: '5' },
  { kind: 'noTrump', number: null },
];

describe('card slots', () => {
  it('give every distinct card its own slot, and map back', () => {
    for (const t of trumps) {
      const deck = [...new Set(buildDecks(1))];
      const slots = deck.map((c) => slotOf(t, c));
      expect(new Set(slots).size).toBe(54);
      for (const s of slots) expect(s).toBeGreaterThanOrEqual(0);
      for (const s of slots) expect(s).toBeLessThan(CARD_SLOTS);
      deck.forEach((c, i) => expect(cardAtSlot(t, slots[i]!)).toBe(c));
    }
  });

  it('are the same under any trump suit (suits rotate together)', () => {
    const rotate = (k: number) => (c: Card) => (c.length === 2 && c !== 'LJ' && c !== 'BJ' ? c[0]! + SUITS[(SUITS.indexOf(c[1] as Suit) + k) % 4]! : c);
    for (const number of ['2', '5', 'K'] as const) {
      const base: Trump = { kind: 'standard', suit: 'S', number };
      for (let k = 1; k < 4; k++) {
        const t: Trump = { kind: 'standard', suit: SUITS[k]!, number };
        for (const r of RANKS) for (const s of SUITS) {
          const c = r + s;
          expect(slotOf(t, rotate(k)(c))).toBe(slotOf(base, c));
        }
      }
    }
  });
});

/** Walk AI matches; at every network-style decision, call `visit`. */
function walk(n: number, config: Partial<TractorConfig>, seed: number, visit: (s: TractorState, seat: number) => void, maxSteps = 3000) {
  let s = createMatchAt({ ...defaultConfig(n), ...config }, seed, aiSeats(n), {});
  for (let i = 0; i < maxSteps && s.status === 'playing' && s.history.length < 1; i++) {
    for (let seat = 0; seat < n; seat++) {
      const st = nextStep(observe(s, seat), emptyDraft());
      if (st && 'decision' in st) visit(s, seat);
    }
    const step = nextAutomatedStep(s, () => true);
    if (!step) break;
    s = applyAction(s, step.action);
  }
  return s;
}

describe('encoder and candidates', () => {
  it('encode finite, bounded vectors at every stage', () => {
    let seen = 0;
    walk(4, {}, 3, (s, seat) => {
      const v = encodeObservation(observe(s, seat));
      expect(v.length).toBe(OBS_DIM);
      expect(v.every((x) => Number.isFinite(x) && Math.abs(x) <= 4)).toBe(true);
      seen++;
    });
    expect(seen).toBeGreaterThan(50);
  });

  it('only offer legal plays', () => {
    let plays = 0;
    walk(5, {}, 9, (s, seat) => {
      const st = nextStep(observe(s, seat))!;
      if (!('decision' in st) || st.decision.kind !== 'play') return;
      for (const c of st.decision.candidates) {
        expect(c.features.length).toBe(CAND_DIM);
        const pick = c.pick;
        if ('action' in pick) {
          const a = pick.action;
          expect(() => applyAction(s, a)).not.toThrow();
        }
        plays++;
      }
    });
    expect(plays).toBeGreaterThan(100);
  });

  it('find a teacher’s choice among the candidates (the distillation label)', () => {
    // A teacher is any other player's action; candidates come from the same generator, so its pick is among them.
    const teacher = netPolicy(new TractorPolicyNet(randomPolicyFile(31, { seat: [8], trunk: [16], cand: [8], joint: 8 })));
    let found = 0;
    let total = 0;
    walk(4, {}, 21, (s, seat) => {
      const st = nextStep(observe(s, seat))!;
      if (!('decision' in st)) return;
      total++;
      if (teacherIndex(st.decision, emptyDraft(), actWithPolicy(s, seat, teacher)) >= 0) found++;
    });
    expect(found / total).toBeGreaterThan(0.95);
  });
});

describe('a network plays whole rounds', () => {
  const net = new TractorPolicyNet(randomPolicyFile(7));
  const policy = netPolicy(net);
  const cases: Array<[number, Partial<TractorConfig>]> = [
    [4, {}],
    [5, {}],
    [6, { gameMode: 'findingFriends' }],
    [8, {}],
    [10, { gameMode: 'tractor' }],
    [12, {}],
    [3, {}],
    [2, { gameMode: 'tractor' }],
    [4, { biddingStyle: 'turns' }],
  ];
  for (const [n, config] of cases) {
    it(`${n} players ${config.gameMode ?? ''} ${config.biddingStyle ?? ''}`.trim(), () => {
      const levels = Array.from({ length: n }, (_, i) => (['5', 'T', 'K', 'A', '2', '9'] as const)[i % 6]);
      let s = createMatchAt({ ...defaultConfig(n), ...config }, n * 13, aiSeats(n), { levels, landlord: 1 % n });
      const nnSeats = new Set([0, n > 2 ? 2 : 1]);
      for (let i = 0; i < 6000 && s.status === 'playing' && s.history.length < 2; i++) {
        let acted = false;
        for (const seat of nnSeats) {
          const a = actWithPolicy(s, seat, policy);
          if (a && a.type !== 'draw') {
            s = applyAction(s, a);
            acted = true;
            break;
          }
        }
        if (acted) continue;
        const step = nextAutomatedStep(s, (seat) => !nnSeats.has(seat), { drawsFor: () => true });
        if (!step) {
          if (s.round.stage === 'roundOver') s = applyAction(s, { type: 'nextRound' });
          else throw new Error(`stuck at ${s.round.stage}`);
          continue;
        }
        s = applyAction(s, step.action);
      }
      expect(s.history.length).toBeGreaterThanOrEqual(1);
    }, 60_000);
  }
});

describe('live-deal bid memo', () => {
  /** Deal a live round, polling every seat at every card like the app's driver. */
  function deal(policy: CandidatePolicy, memo?: BidMemo) {
    let s = createMatchAt({ ...defaultConfig(4), biddingStyle: 'live' }, 5, aiSeats(4), { levels: ['2', '2', '2', '2'], landlord: 0 });
    const asked = new Set<string>();
    let bids = 0;
    while (s.round.stage === 'draw' && s.round.deck.length > 0) {
      for (let seat = 0; seat < 4; seat++) {
        const options = validBidsFor(s, seat);
        if (options.length) asked.add(`${seat}:${options.map((b) => b.card + b.count).join(',')}`);
        const a = actWithPolicy(s, seat, policy, memo);
        if (a?.type === 'bid') {
          s = applyAction(s, a);
          bids++;
        }
      }
      s = applyAction(s, { type: 'draw', seat: s.round.dealTo });
    }
    return { asked: asked.size, bids };
  }

  it('asks a declining seat once per set of legal bids', () => {
    let calls = 0;
    const never: CandidatePolicy = (_o, d) => {
      if (d.kind === 'bid') calls++;
      return 0;
    };
    const polled = deal(never);
    const pollCalls = calls;
    calls = 0;
    const memoed = deal(never, new Map());
    expect(memoed.asked).toBe(polled.asked);
    expect(calls).toBe(memoed.asked);
    expect(calls).toBeLessThan(pollCalls / 3);
  });

  it('still lets a seat bid', () => {
    const eager: CandidatePolicy = (_o, d) => (d.kind === 'bid' ? d.candidates.length - 1 : 0);
    expect(deal(eager, new Map()).bids).toBeGreaterThan(0);
  });
});

describe('float16 policy files', () => {
  it('decode halves exactly', () => {
    expect(halfToFloat(0x0000)).toBe(0);
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x3555)).toBe(0.333251953125);
    expect(halfToFloat(0x7bff)).toBe(65504);
    expect(halfToFloat(0x0001)).toBe(2 ** -24);
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    for (const x of [0.1, -0.73, 3.5e-3, 1234.5, -6.2e-4]) {
      expect(Math.abs(halfToFloat(floatToHalf(x)) - x)).toBeLessThanOrEqual(Math.abs(x) * 2 ** -11);
    }
  });

  it('are half the size and play like the float32 network', () => {
    const f32 = randomPolicyFile(11);
    const f16 = toHalfPolicyFile(f32);
    expect(JSON.stringify(f16).length).toBeLessThan(JSON.stringify(f32).length * 0.55);
    const a = new TractorPolicyNet(f32);
    const b = new TractorPolicyNet(f16);
    let checked = 0;
    walk(4, {}, 5, (s, seat) => {
      if (checked >= 20) return;
      const st = nextStep(observe(s, seat));
      if (!st || !('decision' in st)) return;
      const obs = encodeObservation(observe(s, seat));
      const feats = st.decision.candidates.map((c) => c.features);
      const la = a.logits(obs, feats);
      const lb = b.logits(obs, feats);
      const scale = Math.max(1, ...Array.from(la, Math.abs));
      la.forEach((x, i) => expect(Math.abs(x - lb[i]!)).toBeLessThan(0.01 * scale));
      checked++;
    });
    expect(checked).toBe(20);
  });

  it('reject a tensor of the wrong size', () => {
    const t = randomPolicyFile(2).tensors['score.bias']!;
    expect(() => decodeTensor({ ...t, dtype: 'f16' })).toThrow(/needs 2 bytes/);
  });
});

describe('network levels', () => {
  it('play every table, both modes, up to 12 players', () => {
    const net = new TractorPolicyNet(randomPolicyFile(13));
    setNnLevel({ id: 'nn-test', label: 'Test', url: '' }, net);
    const atFirstPlay = (n: number, mode: 'tractor' | 'findingFriends') => {
      let s = createMatchAt({ ...defaultConfig(n), gameMode: mode }, 21, aiSeats(n), { levels: Array(n).fill('7'), landlord: 0 });
      while (s.round.stage !== 'play') s = applyAction(s, nextAutomatedStep(s, () => true)!.action);
      return s;
    };
    for (const [n, mode] of [[4, 'tractor'], [4, 'findingFriends'], [9, 'findingFriends'], [12, 'tractor']] as const) {
      const s = atFirstPlay(n, mode);
      const seat = s.round.trick!.order[0]!;
      expect(chooseAIAction(s, seat, 'nn-test')).toEqual(actWithPolicy(s, seat, netPolicy(net)));
    }
  });
});

describe('friend calls', () => {
  it('can always be made legal, whatever the friend-selection policy', () => {
    for (const policy of ['unrestricted', 'highestCardNotAllowed', 'pointCardNotAllowed', 'trumpsIncluded'] as const) {
      let s = createMatchAt({ ...defaultConfig(6), gameMode: 'findingFriends', friendSelectionPolicy: policy }, 8, aiSeats(6), { levels: Array(6).fill('A'), landlord: 0 });
      while (s.round.stage !== 'exchange') s = applyAction(s, nextAutomatedStep(s, () => true)!.action);
      const calls = legalFriendCalls(s, s.rules.numFriends);
      expect(calls.length).toBe(s.rules.numFriends);
      // A finished exchange with illegal calls is repaired, a legal one left alone.
      const bury = s.round.hands[s.round.landlord!]!.slice(0, s.round.kittySize);
      const bad = repairExchange(s, { type: 'exchange', seat: s.round.landlord!, bury, friends: [] });
      expect(bad.type === 'exchange' && bad.friends).toEqual(calls);
      const ok = { type: 'exchange' as const, seat: s.round.landlord!, bury, friends: calls };
      expect(repairExchange(s, ok)).toBe(ok);
      expect(() => applyAction(s, bad)).not.toThrow();
    }
  });
});

describe('the shipped difficulty levels', () => {
  const read = (id: string) => JSON.parse(readFileSync(join(process.cwd(), 'src/ai/nn/policies', `${id}.json`), 'utf8')) as TractorPolicyFile;
  const shipped = () => NN_LEVELS.slice(0, 3); // tests may register more levels after these

  it('are Easy, Normal and Hard: three valid policy files from one run, in order of training', () => {
    expect(shipped().map((l) => [l.id, l.label])).toEqual([['easy', 'Easy'], ['normal', 'Normal'], ['hard', 'Hard']]);
    expect(shipped().map((l) => read(l.id).meta?.update)).toEqual([100, 500, 2000]);
    for (const l of shipped()) {
      const f = read(l.id);
      expect(f.format).toBe('tractor/seat-mlp@3');
      expect(new Set(Object.values(f.tensors).map((t) => t.dtype))).toEqual(new Set(['f16']));
      expect(() => new TractorPolicyNet(f)).not.toThrow();
    }
  });

  it('are fetched from their URLs, built once, and a failed fetch can be retried', async () => {
    vi.resetModules();
    let calls = 0;
    let failNext = true;
    vi.stubGlobal('fetch', async (url: string) => {
      calls++;
      if (failNext) {
        failNext = false;
        return { ok: false, status: 503, json: async () => ({}) };
      }
      return { ok: true, status: 200, json: async () => read(url.split('/').pop()!.replace(/\.json.*$/, '').replace(/-[\w-]{8}$/, '')) };
    });
    try {
      const levels = await import('./levels');
      expect(levels.loadedNnLevel('hard')).toBeNull();
      await expect(levels.loadNnLevel('hard')).rejects.toThrow(/HTTP 503/);
      const net = await levels.loadNnLevel('hard');
      expect(levels.loadedNnLevel('hard')).toBe(net);
      expect(await levels.loadNnLevel('hard')).toBe(net); // cached: no second fetch
      expect(calls).toBe(2);
      await expect(levels.loadNnLevel('no-such-level')).rejects.toThrow(/no AI level/);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('play a whole round at a 4-player table and a 5-player Finding Friends table', () => {
    const nets = shipped().map((l) => new TractorPolicyNet(read(l.id)));
    for (const [n, mode] of [[4, 'tractor'], [5, 'findingFriends']] as const) {
      let s = createMatchAt({ ...defaultConfig(n), gameMode: mode }, 5, aiSeats(n), { levels: Array(n).fill('7'), landlord: 1 });
      for (let i = 0; i < 4000 && s.round.stage !== 'roundOver'; i++) {
        if (s.round.stage === 'draw' && s.round.deck.length > 0) {
          s = applyAction(s, { type: 'draw', seat: s.round.dealTo });
          continue;
        }
        const seat = activeSeat(s);
        const a = seat === null ? null : actWithPolicy(s, seat, netPolicy(nets[seat % nets.length]!));
        if (!a) throw new Error(`stuck at ${s.round.stage}`);
        s = applyAction(s, a);
      }
      expect(s.round.stage).toBe('roundOver');
    }
  }, 120_000);
});
