// Ending a round early must never change anything: the levels, who's
// landlord next, the game ending. Unit cases, then a property check over real
// AI rounds: wherever ending early is offered, the early result must equal
// the result of playing the round out.

import { describe, expect, it } from 'vitest';
import { nextAutomatedStep } from '@/ai/driver';
import { aiSeats } from '@/ai/simulate';
import { defaultConfig, type TractorConfig } from './config';
import { applyAction, attackerPointRange, canEndRoundEarly, createInitialState, IllegalAction } from './engine';
import { act, bid, exchange, pickUp, play, scenario } from './testing';
import type { RoundResult, TractorState } from './types';

/** Seat 0 (landlord) trumps everything; attackers have no point cards; all the points sit in seat 0's hand. */
function settled(extra: Partial<TractorConfig> = {}) {
  let s = scenario({
    hands: [['2S', 'BJ', 'KS'], ['3H', '4H', '6H'], ['6C', '7C', '8C'], ['8D', '9D', 'JD']],
    kitty: ['3C', '4C'],
    landlord: 0,
    config: { numDecks: 1, ...extra },
  });
  return act(s, bid(0, '2S', 1), pickUp(0), exchange(0, ['3C', '4C']));
}

describe('ending a round early', () => {
  it('is never available before the first trick, or in the middle of one', () => {
    let s = settled();
    expect(canEndRoundEarly(s)).toBe(false);
    s = act(s, play(0, 'BJ'));
    expect(canEndRoundEarly(s)).toBe(false);
    expect(() => act(s, { type: 'endRoundEarly', seat: 1 })).toThrow(IllegalAction);
  });

  it('waits while unplayed point cards could still move the result', () => {
    let s = settled();
    s = act(s, play(0, 'BJ'), play(1, '3H'), play(2, '6C'), play(3, '8D'));
    // K♠ (10) is still out there as far as the table knows: 0 → +3 for the landlord, 10 → +2.
    expect(attackerPointRange(s)!.max).toBeGreaterThan(0);
    expect(canEndRoundEarly(s)).toBe(false);
  });

  it('is offered once every point card is accounted for, and scores exactly like playing on', () => {
    let s = settled();
    s = act(s, play(0, 'KS'), play(1, '3H'), play(2, '6C'), play(3, '8D')); // the only points go to the landlord
    expect(canEndRoundEarly(s)).toBe(true);
    const early = act(s, { type: 'endRoundEarly', seat: 2 });
    const full = act(s, play(0, 'BJ'), play(1, '4H'), play(2, '7C'), play(3, '9D'), play(0, '2S'), play(1, '6H'), play(2, '8C'), play(3, 'JD'));
    expect(early.round.result!.endedEarly).toBe(true);
    expect(outcome(early.round.result!)).toEqual(outcome(full.round.result!));
    expect(early.levels).toEqual(full.levels);
  });

  it('counts the defenders’ captures as unknown when their points are hidden', () => {
    let s = settled({ hideLandlordPoints: true });
    s = act(s, play(0, 'KS'), play(1, '3H'), play(2, '6C'), play(3, '8D'));
    expect(canEndRoundEarly(s)).toBe(false);
  });

  it('allows for future failed-throw penalties', () => {
    let s = settled({ throwPenalty: 'tenPointsPerAttempt' });
    s = act(s, play(0, 'KS'), play(1, '3H'), play(2, '6C'), play(3, '8D'));
    // Attackers on 0 could still end up +10 (a defender's failed throw): +3 would become +2.
    expect(attackerPointRange(s)!.max).toBe(20);
    expect(canEndRoundEarly(s)).toBe(false);
  });

  it('waits in Finding Friends until every friend is known', () => {
    let s = scenario({
      hands: [['2S', 'BJ'], ['3H', 'AH'], ['6C', '7C'], ['8D', '9D']],
      landlord: 0,
      config: { gameMode: 'findingFriends', numDecks: 1 },
    });
    s = act(s, bid(0, '2S', 1), pickUp(0), exchange(0, [], [{ card: 'AH', ordinal: 1 }]));
    s = act(s, play(0, 'BJ'), play(1, '3H'), play(2, '6C'), play(3, '8D'));
    expect(canEndRoundEarly(s)).toBe(false);
  });
});

function outcome(r: RoundResult) {
  return {
    landlordWon: r.landlordWon,
    landlordDelta: r.landlordDelta,
    attackerDelta: r.attackerDelta,
    bonusLevel: r.bonusLevel,
    nextLandlord: r.nextLandlord,
    winners: r.winners,
    advancements: r.advancements,
  };
}

/** Play the round out with the AI from `s`; return its result and levels. */
function playOut(s: TractorState): TractorState {
  const n = s.round.number;
  for (let i = 0; i < 5000 && s.round.number === n && s.round.stage !== 'roundOver'; i++) {
    const step = nextAutomatedStep(s, () => true);
    if (!step) break;
    s = applyAction(s, step.action);
  }
  return s;
}

describe('ending early never changes the result (AI rounds)', () => {
  const configs: Array<{ n: number; config: Partial<TractorConfig> }> = [
    { n: 4, config: {} },
    { n: 4, config: { throwPenalty: 'tenPointsPerAttempt', kittyPenalty: 'power' } },
    { n: 5, config: {} },
    { n: 6, config: { numDecks: 2 } },
    { n: 4, config: { gameMode: 'findingFriends', numDecks: 1 } },
  ];
  it('matches the played-out result everywhere it is offered', () => {
    let offered = 0;
    for (const { n, config } of configs) {
      for (let seed = 1; seed <= 12; seed++) {
        let s = createInitialState({ ...defaultConfig(n), ...config }, seed * 101 + n, aiSeats(n));
        // A few rounds per match, so levels and landlords vary.
        for (let round = 0; round < 3 && s.status === 'playing'; round++) {
          while (s.round.stage !== 'roundOver' && s.status === 'playing') {
            if (canEndRoundEarly(s)) {
              offered++;
              const early = applyAction(s, { type: 'endRoundEarly', seat: 0 });
              const full = playOut(s);
              expect(outcome(early.round.result!)).toEqual(outcome(full.round.result!));
              expect(early.levels).toEqual(full.levels);
              expect(early.status).toBe(full.status);
            }
            const step = nextAutomatedStep(s, () => true)!;
            s = applyAction(s, step.action);
          }
          if (s.status === 'playing') s = applyAction(s, { type: 'nextRound' });
        }
      }
    }
    expect(offered).toBeGreaterThan(20);
  }, 120_000);
});
