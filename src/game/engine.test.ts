import { describe, expect, it } from 'vitest';
import { aiSeats, simulateMatch } from '@/ai/simulate';
import { nextAutomatedStep } from '@/ai/driver';
import { validBids, type BidRules } from './bidding';
import { DECK_SIZE } from './cards';
import { autoKittySize, defaultConfig, resolveRules, validateConfig, type TractorConfig } from './config';
import { applyAction, createInitialState, IllegalAction, validBidsFor } from './engine';
import { advancePlayer, explainScoring, materialize, scoreRound, DEFAULT_SCORING } from './scoring';
import type { TractorState } from './types';

const RULES: BidRules = { bidPolicy: 'jokerOrGreaterLength', jokerBidPolicy: 'bothTwoOrMore', bidReinforcementPolicy: 'reinforceWhileWinning', numDecks: 2 };

describe('config', () => {
  it('sizes the kitty the classic way', () => {
    expect(autoKittySize(108, 4)).toBe(8);
    expect(autoKittySize(108, 5)).toBe(8);
    expect(autoKittySize(162, 6)).toBe(6);
    expect(autoKittySize(54, 3)).toBe(6);
    expect(autoKittySize(216, 8)).toBe(8);
  });
  it('requires an even table for classic tractor', () => {
    const cfg = { ...defaultConfig(5), gameMode: 'tractor' as const };
    expect(validateConfig(cfg, aiSeats(5)).length).toBeGreaterThan(0);
    expect(validateConfig(defaultConfig(5), aiSeats(5))).toEqual([]);
  });
  it('picks friends and decks from the player count', () => {
    const r = resolveRules(defaultConfig(7), aiSeats(7));
    expect(r.gameMode).toBe('findingFriends');
    expect(r.numFriends).toBe(2);
    expect(r.numDecks).toBe(3);
  });
  it('removes cards so an explicit kitty size deals evenly', () => {
    const r = resolveRules({ ...defaultConfig(4), kittySize: 6 }, aiSeats(4));
    expect((2 * DECK_SIZE - r.cardsToRemove - 6) % 4).toBe(0);
    const s = createInitialState({ ...defaultConfig(4), kittySize: 6 }, 3, aiSeats(4));
    expect(s.round.removed.length).toBe(r.cardsToRemove);
    expect(s.round.kitty.length).toBe(6);
  });
});

describe('bidding', () => {
  it('needs a level card, and more of them to overcall', () => {
    expect(validBids(0, [], ['2H', '3S'], '2', false, 0, RULES).map((b) => b.card)).toEqual(['2H']);
    const first = { seat: 0, card: '2H', count: 1, epoch: 0 };
    // One 2♠ can't overcall one 2♥; a pair can.
    expect(validBids(1, [first], ['2S', '9C'], '2', false, 0, RULES)).toEqual([]);
    expect(validBids(1, [first], ['2S', '2S'], '2', false, 0, RULES)).toEqual([{ seat: 1, card: '2S', count: 2, epoch: 0 }]);
  });
  it('jokers need a pair, and outrank a suit pair', () => {
    expect(validBids(0, [], ['BJ', '5C'], '2', false, 0, RULES)).toEqual([]);
    const pair = { seat: 1, card: '2S', count: 2, epoch: 0 };
    expect(validBids(0, [pair], ['LJ', 'LJ'], '2', false, 0, RULES).map((b) => b.card)).toEqual(['LJ']);
  });
  it('the winning bidder may only reinforce', () => {
    const mine = { seat: 0, card: '2H', count: 1, epoch: 0 };
    expect(validBids(0, [mine], ['2H', '2H', '2S', '2S'], '2', false, 0, RULES)).toEqual([{ seat: 0, card: '2H', count: 2, epoch: 0 }]);
  });
  it('a no-trump landlord has nothing to bid for', () => {
    expect(validBids(0, [], ['BJ', 'BJ'], 'NT', true, 0, RULES)).toEqual([]);
  });
});

describe('scoring', () => {
  const m = materialize(DEFAULT_SCORING, 2, 200);
  it('scores two decks the classic 80-point way', () => {
    expect(scoreRound(m, 0)).toEqual({ landlordWon: true, landlordDelta: 3, attackerDelta: 0 });
    expect(scoreRound(m, 35)).toEqual({ landlordWon: true, landlordDelta: 2, attackerDelta: 0 });
    expect(scoreRound(m, 40)).toEqual({ landlordWon: true, landlordDelta: 1, attackerDelta: 0 });
    expect(scoreRound(m, 75)).toEqual({ landlordWon: true, landlordDelta: 1, attackerDelta: 0 });
    expect(scoreRound(m, 80)).toEqual({ landlordWon: false, landlordDelta: 0, attackerDelta: 0 });
    expect(scoreRound(m, 120)).toEqual({ landlordWon: false, landlordDelta: 0, attackerDelta: 1 });
    expect(scoreRound(m, 160)).toEqual({ landlordWon: false, landlordDelta: 0, attackerDelta: 2 });
    expect(scoreRound(m, 205)).toEqual({ landlordWon: false, landlordDelta: 0, attackerDelta: 3 });
    expect(scoreRound(m, -40)).toEqual({ landlordWon: true, landlordDelta: 4, attackerDelta: 0 });
    expect(explainScoring(m).map((r) => r.from)).toEqual([0, 5, 40, 80, 120, 160, 200]);
  });
  it('stops at the final rank until it is defended', () => {
    expect(advancePlayer('K', 3, true, 'K', 'A', 'unrestricted')).toMatchObject({ to: 'A', blocked: true, finished: false });
    expect(advancePlayer('A', 1, false, 'A', 'A', 'unrestricted')).toMatchObject({ to: 'A', blocked: true });
    expect(advancePlayer('A', 2, true, 'A', 'A', 'unrestricted')).toMatchObject({ finished: true });
    expect(advancePlayer('K', 3, false, 'Q', 'A', 'fullyUnrestricted')).toMatchObject({ finished: true });
    expect(advancePlayer('4', 3, true, '4', 'A', 'defendPoints')).toMatchObject({ to: '5', blocked: true });
  });
});

function cardsInPlay(s: TractorState): number {
  const r = s.round;
  const played = r.tricks.reduce((n, t) => n + t.plays.reduce((m, p) => m + p.cards.length, 0), 0)
    + (r.trick?.plays.reduce((m, p) => m + p.cards.length, 0) ?? 0);
  return r.deck.length + r.kitty.length + r.hands.reduce((n, h) => n + h.length, 0) + played;
}

describe('the round flow', () => {
  it('rejects moves out of turn and keeps every card accounted for', () => {
    let s = createInitialState(defaultConfig(4), 11, aiSeats(4));
    const total = s.rules.numDecks * DECK_SIZE - s.round.removed.length;
    expect(cardsInPlay(s)).toBe(total);
    expect(() => applyAction(s, { type: 'pickUpKitty', seat: 0 })).toThrow(IllegalAction);
    for (let i = 0; i < 3000 && s.status === 'playing' && s.history.length === 0; i++) {
      const step = nextAutomatedStep(s, () => true)!;
      s = applyAction(s, step.action);
      if (s.round.stage !== 'roundOver') expect(cardsInPlay(s)).toBe(total);
    }
    expect(s.history.length).toBe(1);
    expect(s.round.stage).toBe('roundOver');
  });

  it('is deterministic: the same seed and moves give the same state', () => {
    const a = simulateMatch({ seats: aiSeats(4), seed: 99, maxRounds: 2 });
    const b = simulateMatch({ seats: aiSeats(4), seed: 99, maxRounds: 2 });
    expect(JSON.stringify(a.state)).toBe(JSON.stringify(b.state));
  });

  it('turn bidding deals everything up front and asks each seat in turn', () => {
    const s = createInitialState({ ...defaultConfig(4), biddingStyle: 'turns' }, 5, aiSeats(4));
    expect(s.round.stage).toBe('bid');
    expect(s.round.deck.length).toBe(0);
    expect(s.round.hands.every((h) => h.length === 25)).toBe(true);
    const turn = s.round.bidTurn!;
    expect(s.activePlayerId).toBe(s.seats[turn]!.id);
    for (let seat = 0; seat < 4; seat++) if (seat !== turn) expect(validBidsFor(s, seat)).toEqual([]);
  });
});

describe('full AI matches', () => {
  const cases: Array<{ name: string; n: number; config?: Partial<TractorConfig>; rounds?: number }> = [
    { name: '4p tractor, 2 decks, live bidding', n: 4 },
    { name: '4p tractor, turn bidding', n: 4, config: { biddingStyle: 'turns' } },
    { name: '5p finding friends', n: 5 },
    { name: '6p finding friends, 3 decks, protections', n: 6, config: { trickDrawPolicy: 'longerTuplesProtected' }, rounds: 4 },
    { name: '3p finding friends, 1 deck', n: 3 },
    { name: '2p tractor, 1 deck', n: 2, config: { gameMode: 'tractor' } },
    { name: '8p tractor, 4 decks', n: 8, rounds: 3 },
    { name: '6p tractor, 2 decks, explicit kitty, power penalty', n: 6, config: { numDecks: 2, kittySize: 6, kittyPenalty: 'power', throwPenalty: 'tenPointsPerAttempt' }, rounds: 4 },
    { name: '4p finding friends, NT max rank, highest-card throw evaluation', n: 4, config: { gameMode: 'findingFriends', maxRank: 'NT', throwEvaluationPolicy: 'highest' }, rounds: 6 },
    { name: '7p finding friends, easy AIs, higher-suit bidding', n: 7, config: { bidPolicy: 'jokerOrHigherSuit', jokerBidPolicy: 'bothNumDecks' }, rounds: 3 },
  ];
  for (const c of cases) {
    it(c.name, () => {
      const levels = c.name.includes('easy') ? (['easy'] as const) : (['normal', 'easy'] as const);
      const res = simulateMatch({ seats: aiSeats(c.n, [...levels]), config: c.config, seed: c.n * 31 + 7, maxRounds: c.rounds ?? 40 });
      expect(res.rounds).toBeGreaterThan(0);
      if (!c.rounds) expect(res.finished).toBe(true);
      for (const r of res.state.history) {
        expect(r.advancements.length).toBe(c.n);
        expect(r.landlordTeam).toContain(r.landlord);
      }
    }, 60_000);
  }
});
