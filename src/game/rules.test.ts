// The nuanced rules, scenario by scenario: declaring / overcalling, point
// cards, thresholds per deck count, Finding Friends calls and reveals,
// throws and their penalty, levelling up and winning the game. Several are
// ports of shengji's core/src/game_state/mod.rs and mechanics bidding /
// scoring tests (same hands, same expectations); the rest are written to the
// same rules.

import { describe, expect, it } from 'vitest';
import { validBids, type Bid, type BidReinforcementPolicy } from './bidding';
import { RANKS, buildDecks, cardPoints, totalPoints, type Card, type Level } from './cards';
import { autoNumDecks, defaultConfig, resolveRules } from './config';
import {
  activeSeat,
  attackerPointsSoFar,
  canPickUp,
  friendCallProblems,
  IllegalAction,
  landlordToBe,
  playProblemFor,
  scoringParams,
  sortedHand,
  validBidsFor,
} from './engine';
import { advancePlayer, DEFAULT_SCORING, materialize, scoreRound, type AdvancementPolicy } from './scoring';
import { act, bid, exchange, pass, pickUp, plainSeats, play, scenario } from './testing';
import type { TractorState } from './types';

const sortCards = (cs: Card[]) => cs.slice().sort();

// ------------------------------------------------------------------ declaring

describe('declaring trump and overcalling', () => {
  // shengji bidding.rs test_valid_bids / test_valid_bids_joker_or_higher_suit
  const hand = ['2C', '2C', '2C', '2S', '2S', 'LJ', 'LJ', 'BJ', 'BJ'];
  const b = (seat: number, card: Card, count: number): Bid => ({ seat, card, count, epoch: 0 });
  const set = (bids: Bid[]) => new Set(bids.map((x) => `${x.seat}:${x.card}x${x.count}`));
  const valid = (bids: Bid[], policy: BidReinforcementPolicy, higherSuit = false) =>
    set(validBids(0, bids, hand, '2', false, 0, {
      bidPolicy: higherSuit ? 'jokerOrHigherSuit' : 'jokerOrGreaterLength',
      jokerBidPolicy: 'bothTwoOrMore',
      bidReinforcementPolicy: policy,
      numDecks: 3,
    }));

  it('lists the same legal bids as shengji', () => {
    expect(valid([], 'reinforceWhileWinning')).toEqual(set([b(0, '2S', 1), b(0, '2S', 2), b(0, '2C', 1), b(0, '2C', 2), b(0, '2C', 3), b(0, 'BJ', 2), b(0, 'LJ', 2)]));
    expect(valid([b(0, '2C', 1)], 'reinforceWhileWinning')).toEqual(set([b(0, '2C', 2), b(0, '2C', 3)]));
    expect(valid([b(0, '2C', 1)], 'reinforceWhileEquivalent')).toEqual(set([b(0, '2C', 2), b(0, '2C', 3)]));
    expect(valid([b(0, '2C', 1)], 'overturnOrReinforceWhileWinning')).toEqual(set([b(0, '2S', 2), b(0, '2C', 2), b(0, '2C', 3), b(0, 'BJ', 2), b(0, 'LJ', 2)]));
    const overcalled = [b(0, '2C', 1), b(1, '2S', 2)];
    expect(valid(overcalled, 'reinforceWhileWinning')).toEqual(set([b(0, '2C', 3), b(0, 'BJ', 2), b(0, 'LJ', 2)]));
    expect(valid(overcalled, 'overturnOrReinforceWhileWinning')).toEqual(set([b(0, '2C', 3), b(0, 'BJ', 2), b(0, 'LJ', 2)]));
    expect(valid(overcalled, 'reinforceWhileEquivalent')).toEqual(set([b(0, '2C', 2), b(0, '2C', 3), b(0, 'BJ', 2), b(0, 'LJ', 2)]));
    expect(valid([b(0, '2C', 1), b(1, 'LJ', 2)], 'reinforceWhileEquivalent')).toEqual(set([b(0, '2C', 3), b(0, 'BJ', 2)]));
  });

  it('ranks suits under the higher-suit policy (♠ > ♥ > ♣ > ♦) like shengji', () => {
    const r = 'reinforceWhileWinning';
    expect(valid([b(0, '2C', 1), b(1, '2S', 2)], r, true)).toEqual(set([b(0, '2C', 3), b(0, 'BJ', 2), b(0, 'LJ', 2)]));
    expect(valid([b(0, '2C', 1), b(1, '2H', 2)], r, true)).toEqual(set([b(0, '2S', 2), b(0, '2C', 3), b(0, 'BJ', 2), b(0, 'LJ', 2)]));
    expect(valid([b(0, '2C', 1), b(1, '2D', 2)], r, true)).toEqual(set([b(0, '2S', 2), b(0, '2C', 2), b(0, '2C', 3), b(0, 'BJ', 2), b(0, 'LJ', 2)]));
    expect(valid([b(0, '2C', 1), b(1, '2D', 3)], r, true)).toEqual(set([b(0, '2C', 3)]));
    expect(valid([b(0, '2C', 1), b(1, 'LJ', 2)], r, true)).toEqual(set([b(0, '2C', 3), b(0, 'BJ', 2)]));
    expect(valid([b(0, '2C', 1), b(1, 'BJ', 2)], r, true)).toEqual(set([b(0, '2C', 3)]));
    expect(valid([b(0, '2C', 1), b(1, 'LJ', 3)], r, true)).toEqual(set([]));
    expect(valid([b(0, '2C', 1), b(1, 'BJ', 3)], r, true)).toEqual(set([]));
  });

  it('runs shengji’s bid sequence: reinforce, then jokers overcall', () => {
    // test_bid_sequence: p1 2♥2♥, p2 big jokers, p3 small jokers, p4 2♠2♠.
    let s = scenario({ hands: [['2H', '2H'], ['BJ', 'BJ'], ['LJ', 'LJ'], ['2S', '2S']] });
    s = act(s, bid(0, '2H', 1), bid(0, '2H', 2), bid(2, 'LJ', 2), bid(1, 'BJ', 2));
    expect(() => act(s, bid(0, '2H', 2))).toThrow(IllegalAction);
    expect(landlordToBe(s)).toBe(1);
    expect(canPickUp(s, 0)).toBe(false);
    expect(canPickUp(s, 1)).toBe(true);
    s = act(s, pickUp(1));
    expect(s.round.landlord).toBe(1);
    expect(s.round.trump).toEqual({ kind: 'noTrump', number: '2' });
  });

  it('the overcaller becomes the first landlord, with their suit as trump', () => {
    const hands = [['2H', '2H', '3C'], ['4C', '5C', '6C'], ['2S', '2S', '7C'], ['8C', '9C', 'TC']];
    let s = act(scenario({ hands }), bid(0, '2H', 1));
    expect(landlordToBe(s)).toBe(0);
    s = act(s, bid(2, '2S', 2));
    expect(landlordToBe(s)).toBe(2);
    // Seat 0 can't take the kitty any more; it could only overcall with three 2s or jokers.
    expect(() => act(s, pickUp(0))).toThrow(IllegalAction);
    expect(validBidsFor(s, 0)).toEqual([]);
    s = act(s, pickUp(2));
    expect(s.landlord).toBe(2);
    expect(s.round.trump).toEqual({ kind: 'standard', suit: 'S', number: '2' });
    expect(s.round.stage).toBe('exchange');
    s = act(s, exchange(2, []));
    expect(s.round.landlordTeam).toEqual([0, 2]);
    expect(activeSeat(s)).toBe(2); // the landlord leads the first trick
  });

  it('with "first bidder" the first declarer stays landlord but the overcall sets trump', () => {
    const hands = [['2H', '3C'], ['4C', '5C'], ['2S', '2S'], ['8C', '9C']];
    let s = act(scenario({ hands, config: { firstLandlordSelection: 'firstBid' } }), bid(0, '2H', 1), bid(2, '2S', 2));
    expect(landlordToBe(s)).toBe(0);
    s = act(s, pickUp(0));
    expect(s.round.trump).toEqual({ kind: 'standard', suit: 'S', number: '2' });
  });

  it('once there is a landlord, everyone bids at the landlord’s level and the landlord keeps the seat', () => {
    const hands = [['5H', '3C'], ['2S', '2S'], ['5D', '4C'], ['5S', '5S']];
    let s = scenario({ hands, landlord: 1, levels: ['2', '5', '2', '5'] });
    // Seat 1's 2s are no use: the level is 5.
    expect(validBidsFor(s, 1)).toEqual([]);
    s = act(s, bid(0, '5H', 1), bid(3, '5S', 2));
    expect(landlordToBe(s)).toBe(1);
    expect(() => act(s, pickUp(3))).toThrow(IllegalAction);
    s = act(s, pickUp(1));
    expect(s.round.landlord).toBe(1);
    expect(s.round.trump).toEqual({ kind: 'standard', suit: 'S', number: '5' });
  });

  it('an overcall can still come after the deal ends, but not after the kitty is taken', () => {
    let s = act(scenario({ hands: [['2H', '3C'], ['2D', '2D'], ['4C', '5C'], ['6C', '7C']] }), bid(0, '2H', 1));
    s = act(s, bid(1, '2D', 2));
    s = act(s, pickUp(1));
    expect(validBidsFor(s, 0)).toEqual([]);
    expect(() => act(s, bid(0, '2H', 2))).toThrow(IllegalAction);
  });

  it('turn bidding: overcalls go round until everyone else passes', () => {
    const hands = [['2H', '3C'], ['2S', '2S'], ['4C', '5C'], ['6C', '7C']];
    let s = scenario({ hands, turns: true, dealer: 0 });
    expect(() => act(s, bid(1, '2S', 2))).toThrow(IllegalAction); // not seat 1's turn yet
    s = act(s, bid(0, '2H', 1), bid(1, '2S', 2), pass(2), pass(3));
    expect(s.round.stage).toBe('bid');
    s = act(s, pass(0));
    expect(s.round.stage).toBe('exchange');
    expect(s.round.landlord).toBe(1);
    expect(s.round.trump).toEqual({ kind: 'standard', suit: 'S', number: '2' });
  });

  it('nobody declaring reshuffles in the first round, and turns over the kitty later', () => {
    const hands = [['3C'], ['4C'], ['5C'], ['6C']];
    let s = act(scenario({ hands, turns: true }), pass(0), pass(1), pass(2), pass(3));
    expect(s.round.stage).toBe('bid');
    expect(s.round.hands.every((h) => h.length > 1)).toBe(true); // a fresh deal
    let t = scenario({ hands, kitty: ['9H', '8D'], turns: true, landlord: 2, levels: ['2', '2', '7', '2'] });
    t = act(t, pass(2), pass(3), pass(0), pass(1));
    expect(t.round.bidTurn).toBeNull();
    t = act(t, { type: 'revealKitty', seat: 2 });
    expect(t.round.trump).toEqual({ kind: 'standard', suit: 'H', number: '7' });
    expect(t.round.stage).toBe('exchange');
  });
});

// ------------------------------------------------------------------ points

describe('points: only 5s, 10s and kings', () => {
  it('values every card', () => {
    for (const r of RANKS) {
      for (const s of ['S', 'H', 'D', 'C']) expect(cardPoints(r + s)).toBe(r === '5' ? 5 : r === 'T' || r === 'K' ? 10 : 0);
    }
    expect(cardPoints('LJ')).toBe(0);
    expect(cardPoints('BJ')).toBe(0);
    expect(totalPoints(buildDecks(1))).toBe(100);
    expect(totalPoints(buildDecks(3))).toBe(300);
  });
});

/**
 * One deck, tractor, landlord seat 0 at level 2. Seat 1 declares hearts.
 * Trick 1: 3♣ ← K♣ (s1) · 6♠ · 10♣ → seat 1 takes 20.
 * Trick 2 (last): seat 1 leads 2♥ → takes K♦ (10), and the kitty.
 */
function attackersTakeOver(levels: Level[], extra: Partial<Parameters<typeof scenario>[0]> = {}) {
  const hands = [['3C', '4D'], ['2H', 'KC'], ['6S', '7S'], ['TC', 'KD']];
  let s = scenario({ hands, kitty: ['KS', 'TS', '5S', '5H'], landlord: 0, levels, config: { numDecks: 1, ...(extra.config ?? {}) } });
  s = act(s, bid(1, '2H', 1), pickUp(0), exchange(0, ['KS', 'TS', '5S', '5H']));
  s = act(s, play(0, '3C'), play(1, 'KC'), play(2, '6S'), play(3, 'TC'));
  expect(s.round.tricks[0]!.winner).toBe(1);
  s = act(s, play(1, '2H'), play(2, '7S'), play(3, 'KD'), play(0, '4D'));
  return s;
}

describe('the kitty and the attackers’ total', () => {
  it('attackers winning the last trick get the kitty ×2 (single) and count only their own captures', () => {
    const s = attackersTakeOver(['2', '2', '2', '2']);
    const r = s.round.result!;
    expect(r.kittyPoints).toBe(30);
    expect(r.kittyMultiplier).toBe(2);
    expect(r.kittyToAttackers).toBe(true);
    expect(r.attackerPoints).toBe(20 + 10 + 60);
  });

  it('a pair in the last trick doubles the multiplier to ×4', () => {
    // Seat 0 (landlord) trumps trick 1, then leads 3♣3♣; attacker seat 1's Q♣Q♣ wins the last trick.
    let s = scenario({
      hands: [['2S', '3C', '3C'], ['QC', 'QC', '4D'], ['4C', '4C', '5D'], ['6C', '6C', '6D']],
      kitty: ['KS', '5S'],
      landlord: 0,
      config: { numDecks: 2 },
    });
    s = act(s, bid(0, '2S', 1), pickUp(0), exchange(0, ['KS', '5S']));
    s = act(s, play(0, '2S'), play(1, '4D'), play(2, '5D'), play(3, '6D'));
    expect(s.round.tricks[0]!.winner).toBe(0); // the 5♦ goes to the landlord's side
    s = act(s, play(0, '3C', '3C'), play(1, 'QC', 'QC'), play(2, '4C', '4C'), play(3, '6C', '6C'));
    const r = s.round.result!;
    expect(r.kittyMultiplier).toBe(4);
    expect(r.kittyToAttackers).toBe(true);
    expect(r.attackerPoints).toBe(15 * 4);
  });

  it('the landlord side winning the last trick keeps the kitty away from the attackers', () => {
    let s = scenario({
      hands: [['2S', '3C'], ['4D', '4C'], ['5D', '6C'], ['6D', '7C']],
      kitty: ['KS', 'TS'],
      landlord: 0,
      config: { numDecks: 2 },
    });
    s = act(s, bid(0, '2S', 1), pickUp(0), exchange(0, ['KS', 'TS']));
    s = act(s, play(0, '3C'), play(1, '4C'), play(2, '6C'), play(3, '7C')); // seat 3 wins
    s = act(s, play(3, '6D'), play(0, '2S'), play(1, '4D'), play(2, '5D')); // seat 0 trumps the last trick
    const r = s.round.result!;
    expect(r.kittyToAttackers).toBe(false);
    expect(r.attackerPoints).toBe(0);
  });
});

// ------------------------------------------------------------------ thresholds

describe('thresholds scale with the number of decks (which follows the player count)', () => {
  it('needs 40 points per deck to take over', () => {
    const expected: Record<number, [number, number]> = { 2: [1, 40], 3: [1, 40], 4: [2, 80], 5: [2, 80], 6: [3, 120], 7: [3, 120], 8: [4, 160], 10: [5, 200] };
    for (const [n, [decks, threshold]] of Object.entries(expected)) {
      const rules = resolveRules(defaultConfig(Number(n)), plainSeats(Number(n)));
      expect(rules.numDecks).toBe(decks);
      expect(autoNumDecks(Number(n))).toBe(decks);
      const m = materialize(scoringParams(defaultConfig(Number(n))), decks, decks * 100);
      expect(m.landlordWins[m.landlordWins.length - 1]!.end).toBe(threshold);
      expect(scoreRound(m, threshold - 5).landlordWon).toBe(true);
      expect(scoreRound(m, threshold).landlordWon).toBe(false);
    }
  });

  it('matches shengji’s level table (2 decks), including penalties below zero and the 3-deck bonus', () => {
    const m = materialize({ ...DEFAULT_SCORING, bonusLevelForSmallerTeam: false }, 2, 200);
    const table: Array<[number, boolean, number, number]> = [
      [-80, true, 5, 0], [-40, true, 4, 0], [-35, true, 3, 0], [0, true, 3, 0], [5, true, 2, 0], [35, true, 2, 0],
      [40, true, 1, 0], [75, true, 1, 0], [80, false, 0, 0], [115, false, 0, 0], [120, false, 0, 1], [155, false, 0, 1],
      [160, false, 0, 2], [195, false, 0, 2], [200, false, 0, 3], [235, false, 0, 3], [240, false, 0, 4], [280, false, 0, 5],
    ];
    for (const [pts, won, ld, ad] of table) expect(scoreRound(m, pts)).toEqual({ landlordWon: won, landlordDelta: ld, attackerDelta: ad });
    const m3 = materialize(DEFAULT_SCORING, 3, 300);
    expect(scoreRound(m3, 0).landlordDelta).toBe(3);
    expect(scoreRound(m3, 50).landlordDelta).toBe(2);
  });

  it('an explicit deck count overrides the default', () => {
    expect(resolveRules({ ...defaultConfig(6), numDecks: 2 }, plainSeats(6)).numDecks).toBe(2);
  });
});

// ------------------------------------------------------------------ friends

describe('Finding Friends', () => {
  it('only allows the calls shengji allows (test_set_friends)', () => {
    const cases: Array<[string, Card, Array<[Card, boolean]>]> = [
      ['unrestricted', '7S', [['KC', true], ['3S', false], ['3C', true], ['AC', true]]],
      ['pointCardNotAllowed', '7S', [['KC', false], ['3S', false], ['3C', true], ['AC', true]]],
      ['pointCardNotAllowed', 'AS', [['KC', true], ['3S', false], ['3C', true], ['AC', false]]],
      ['highestCardNotAllowed', '7S', [['KC', true], ['3S', false], ['3C', true], ['AC', false]]],
      ['trumpsIncluded', '7S', [['KC', true], ['3S', true], ['3C', true], ['AC', true]]],
    ];
    for (const [policy, bidCard, calls] of cases) {
      const level = bidCard[0] as Level;
      let s = scenario({
        hands: [[bidCard], [bidCard], [bidCard], [bidCard]],
        landlord: 1,
        levels: ['2', level, '2', '2'],
        config: { gameMode: 'findingFriends', friendSelectionPolicy: policy as never },
      });
      s = act(s, bid(0, bidCard, 1), pickUp(1));
      for (const [card, ok] of calls) {
        expect(friendCallProblems(s, [{ card, ordinal: 1 }]).length === 0, `${policy} ${bidCard}: ${card}`).toBe(ok);
      }
    }
  });

  it('refuses the wrong number of calls, repeats, and copies that do not exist', () => {
    let s = scenario({ hands: [['2S'], ['3C'], ['4C'], ['5C'], ['6C'], ['7C']], landlord: 0, config: { gameMode: 'findingFriends' } });
    s = act(s, bid(0, '2S', 1), pickUp(0));
    expect(s.rules.numFriends).toBe(2);
    expect(friendCallProblems(s, [{ card: 'AH', ordinal: 1 }]).length).toBeGreaterThan(0);
    expect(friendCallProblems(s, [{ card: 'AH', ordinal: 1 }, { card: 'AH', ordinal: 1 }]).length).toBeGreaterThan(0);
    expect(friendCallProblems(s, [{ card: 'AH', ordinal: 1 }, { card: 'AH', ordinal: 4 }]).length).toBeGreaterThan(0); // 3 decks
    expect(friendCallProblems(s, [{ card: 'AH', ordinal: 1 }, { card: 'AH', ordinal: 3 }])).toEqual([]);
    expect(() => act(s, exchange(0, [], [{ card: 'AH', ordinal: 1 }]))).toThrow(IllegalAction);
  });

  it('“the second A♠”: the first copy (by anyone) is skipped, the second makes the friend', () => {
    let s = scenario({
      hands: [['2H', '3S'], ['AS', '4C'], ['7C', '8C'], ['AS', '6C']],
      landlord: 0,
      config: { gameMode: 'findingFriends', numDecks: 2 },
    });
    s = act(s, bid(0, '2H', 1), pickUp(0), exchange(0, [], [{ card: 'AS', ordinal: 2 }]));
    s = act(s, play(0, '3S'), play(1, 'AS'));
    expect(s.round.landlordTeam).toEqual([0]);
    expect(s.round.friends[0]!.skip).toBe(0);
    s = act(s, play(2, '7C'), play(3, 'AS'));
    expect(s.round.landlordTeam).toEqual([0, 3]);
    expect(s.round.friends[0]!.seat).toBe(3);
  });

  it('the landlord playing the called copy uses up the call (or not, with "no double join")', () => {
    const hands = [['2H', 'AS'], ['AS', '4C'], ['7C', '8C'], ['5C', '6C']];
    const run = (policy: 'unrestricted' | 'noDoubleJoin') => {
      let s = scenario({ hands, landlord: 0, config: { gameMode: 'findingFriends', numDecks: 2, multipleJoinPolicy: policy } });
      s = act(s, bid(0, '2H', 1), pickUp(0), exchange(0, [], [{ card: 'AS', ordinal: 1 }]));
      return act(s, play(0, 'AS'), play(1, 'AS'), play(2, '7C'), play(3, '5C'));
    };
    const a = run('unrestricted');
    expect(a.round.landlordTeam).toEqual([0]);
    expect(a.round.friends[0]!.seat).toBe(0);
    const b2 = run('noDoubleJoin');
    expect(b2.round.landlordTeam).toEqual([0, 1]);
    expect(b2.round.friends[0]!.seat).toBe(1);
  });

  it('a friend’s points captured before the reveal count for the landlord’s side', () => {
    let s = scenario({
      hands: [['2S', '3H', '4D'], ['KH', '6C', '5D'], ['AH', 'AD', '9C'], ['5H', 'JC', 'QC']],
      landlord: 0,
      config: { gameMode: 'findingFriends', numDecks: 1 },
    });
    s = act(s, bid(0, '2S', 1), pickUp(0), exchange(0, [], [{ card: 'AD', ordinal: 1 }]));
    s = act(s, play(0, '3H'), play(1, 'KH'), play(2, 'AH'), play(3, '5H'));
    expect(attackerPointsSoFar(s)).toBe(15); // seat 2 isn't known to be a friend yet
    s = act(s, play(2, 'AD'), play(3, 'JC'), play(0, '4D'), play(1, '5D'));
    expect(s.round.landlordTeam).toEqual([0, 2]);
    expect(attackerPointsSoFar(s)).toBe(0);
    s = act(s, play(2, '9C'), play(3, 'QC'), play(0, '2S'), play(1, '6C'));
    const r = s.round.result!;
    expect(r.attackerPoints).toBe(0);
    expect(r.landlordWon).toBe(true);
    expect(r.bonusLevel).toBe(false);
    expect(s.levels).toEqual(['5', '2', '5', '2']);
  });

  it('a short-handed landlord gets a bonus level (shengji test_landlord_small_team)', () => {
    const hands = [['7S', '3D'], ['4D', '5D'], ['6C', '8C'], ['9C', 'TC'], ['JC', 'QC'], ['KC', 'AC'], ['2H', '3H'], ['4H', '5H']];
    let s = scenario({
      hands,
      kitty: ['6H', '6H', '8S', '8S', '9H', '9H', 'JH', 'JH'],
      landlord: 0,
      levels: ['7', '2', '2', '2', '2', '2', '2', '2'],
      config: { gameMode: 'findingFriends', numFriends: 3 },
    });
    s = act(s, bid(0, '7S', 1), pickUp(0));
    s = act(s, exchange(0, ['6H', '6H', '8S', '8S', '9H', '9H', 'JH', 'JH'], [{ card: '3D', ordinal: 1 }, { card: '4D', ordinal: 1 }, { card: '5D', ordinal: 1 }]));
    expect(s.round.landlordTeam).toEqual([0]);
    s = act(s, ...hands.map((h, i) => play(i, h[0]!)));
    expect(s.round.landlordTeam).toEqual([0, 1]);
    s = act(s, ...[0, 1, 2, 3, 4, 5, 6, 7].map((i) => play(i, hands[i]![1]!)));
    expect(s.round.landlordTeam).toEqual([0, 1]);
    expect(s.round.friends.every((f) => f.seat !== null)).toBe(true);
    const r = s.round.result!;
    expect(r.attackerPoints).toBe(0);
    expect(r.bonusLevel).toBe(true);
    expect(r.landlordDelta).toBe(4);
    expect(s.levels).toEqual(['J', '6', '2', '2', '2', '2', '2', '2']);
    expect(r.nextLandlord).toBe(1); // the next defender after the landlord
  });

  it('plays shengji’s 6-player game (test_full_game_play): every follow legal, winners lead in turn', () => {
    const hands = [
      ['LJ', '7D', '7D', '7H', 'KH', '9H', '9H', '4H', '3H', 'AS', 'QS', 'QS', '9S', '9S', '8S', '5S', 'KD', '8D', '6D', '5D', '4D', 'KC', 'KC', 'JC', '9C', '8C'],
      ['BJ', 'BJ', '7C', '7C', '7S', 'KH', 'KH', '6H', '4H', '3H', 'KS', 'JS', '4S', '3S', '2S', 'KD', 'TD', '4D', '4D', '2D', '2D', 'KC', '9C', '5C', '4C', '3C'],
      ['LJ', '7S', 'AH', 'TH', 'TH', '8H', '8H', '5H', '5H', '2H', 'TS', '8S', '5S', '3S', 'AD', 'JD', '8D', '6D', '5D', 'AC', 'JC', 'TC', '6C', '5C', '5C', '2C'],
      ['7H', '7S', 'QH', 'QH', 'JH', 'JH', '8H', 'KS', 'JS', 'TS', 'TS', '6S', '2S', 'QD', '8D', '5D', '3D', '2D', 'AC', 'QC', 'JC', '9C', '8C', '6C', '2C', '2C'],
      ['LJ', '7D', 'AH', '9H', '6H', '3H', '2H', '2H', 'KS', '6S', '6S', '5S', '4S', '2S', 'QD', 'JD', 'JD', 'TD', '9D', '9D', '3D', '3D', 'QC', 'TC', '3C', '3C'],
      ['BJ', '7H', 'AH', 'QH', 'TH', '6H', '5H', '4H', 'AS', 'AS', 'QS', 'JS', '8S', '4S', '3S', 'AD', 'AD', 'KD', 'QD', 'TD', '9D', 'AC', '8C', '6C', '4C', '4C'],
    ];
    const kitty = ['7C', '9S', '6D', 'JD', 'QC', 'TC'];
    let s = scenario({ hands, kitty, landlord: 1, levels: ['2', '7', '2', '2', '2', '2'], config: { gameMode: 'findingFriends' } });
    expect(s.rules.numDecks).toBe(3);
    expect(s.rules.numFriends).toBe(2);
    s = act(s, bid(0, '7D', 2), pickUp(1));
    expect(s.round.trump).toEqual({ kind: 'standard', suit: 'D', number: '7' });
    s = act(s, exchange(1, kitty, [{ card: 'KC', ordinal: 1 }, { card: 'KH', ordinal: 1 }]));
    const tricks: Array<Array<[number, Card[]]>> = [
      [[1, ['KH', 'KH']], [2, ['8H', '8H']], [3, ['JH', 'JH']], [4, ['2H', '2H']], [5, ['4H', '5H']], [0, ['9H', '9H']]],
      [[1, ['3C']], [2, ['6C']], [3, ['6C']], [4, ['TC']], [5, ['6C']], [0, ['KC']]],
      [[0, ['AS']], [1, ['2S']], [2, ['3S']], [3, ['2S']], [4, ['2S']], [5, ['3S']]],
      [[0, ['QS', 'QS']], [1, ['3S', '4S']], [2, ['5S', '8S']], [3, ['TS', 'TS']], [4, ['6S', '6S']], [5, ['AS', 'AS']]],
      [[5, ['BJ']], [0, ['4D']], [1, ['7S']], [2, ['5D']], [3, ['5D']], [4, ['TD']]],
      [[5, ['AD', 'AD']], [0, ['7D', '7D']], [1, ['2D', '2D']], [2, ['6D', '8D']], [3, ['2D', '3D']], [4, ['3D', '3D']]],
      [[0, ['9S', '9S']], [1, ['JS', 'KS']], [2, ['TS', '2H']], [3, ['6S', 'JS']], [4, ['4S', '5S']], [5, ['4S', '8S']]],
      [[0, ['5S']], [1, ['TD']], [2, ['2C']], [3, ['KS']], [4, ['KS']], [5, ['JS']]],
      [[1, ['BJ', 'BJ']], [2, ['JD', 'AD']], [3, ['8D', 'QD']], [4, ['9D', '9D']], [5, ['9D', 'TD']], [0, ['5D', 'KD']]],
      [[1, ['7C', '7C']], [2, ['7S', 'LJ']], [3, ['7S', '7H']], [4, ['JD', 'JD']], [5, ['QD', 'KD']], [0, ['6D', '8D']]],
      [[1, ['4D', '4D']], [2, ['TC', 'JC']], [3, ['8C', '9C']], [4, ['QD', '7D']], [5, ['8C', '7H']], [0, ['7H', 'LJ']]],
      [[1, ['3H']], [2, ['AH']], [3, ['8H']], [4, ['3H']], [5, ['6H']], [0, ['3H']]],
      [[2, ['TH', 'TH']], [3, ['QH', 'QH']], [4, ['6H', '9H']], [5, ['TH', 'QH']], [0, ['4H', 'KH']], [1, ['4H', '6H']]],
      [[3, ['2C']], [4, ['3C']], [5, ['4C']], [0, ['KC']], [1, ['KC']], [2, ['5C']]],
      [[0, ['8S']], [1, ['4C']], [2, ['AC']], [3, ['AC']], [4, ['3C']], [5, ['QS']]],
      [[5, ['4C']], [0, ['8C']], [1, ['9C']], [2, ['5C']], [3, ['2C']], [4, ['QC']]],
      [[4, ['AH']], [5, ['AH']], [0, ['9C']], [1, ['5C']], [2, ['5H']], [3, ['JC']]],
      [[4, ['LJ']], [5, ['AC']], [0, ['JC']], [1, ['KD']], [2, ['5H']], [3, ['QC']]],
    ];
    tricks.forEach((trick, t) => {
      for (const [seat, cards] of trick) {
        expect(activeSeat(s), `trick ${t + 1}: seat ${seat + 1} to play`).toBe(seat);
        s = act(s, play(seat, ...cards));
      }
      if (t === 0) expect(s.round.landlordTeam).toEqual([1]); // the landlord played the called K♥ itself
      if (t === 1) expect(s.round.landlordTeam).toEqual([1, 0]); // p1's K♣ makes them a friend
    });
    expect(s.round.tricks.length).toBe(18);
  });
});

// ------------------------------------------------------------------ throws

describe('throws and the failed-throw penalty', () => {
  function throwing(throwPenalty: 'none' | 'tenPointsPerAttempt') {
    // Landlord seat 0 (♠ trump) throws A♥ Q♥; seat 1's K♥ beats the queen.
    let s = scenario({
      hands: [['2S', 'AH', 'QH'], ['KH', '3C', '4C'], ['3D', '4D', '6D'], ['6C', '7C', '8C']],
      landlord: 0,
      config: { throwPenalty },
    });
    s = act(s, bid(0, '2S', 1), pickUp(0), exchange(0, []));
    return s;
  }

  it('a throw nobody can beat stands', () => {
    let s = scenario({ hands: [['2S', 'AH', 'KH'], ['QH', '3C', '4C'], ['3D', '4D', '6D'], ['5C', '6C', '7C']], landlord: 0 });
    s = act(s, bid(0, '2S', 1), pickUp(0), exchange(0, []), play(0, 'AH', 'KH'));
    expect(s.round.trick!.plays[0]!.cards).toEqual(['KH', 'AH']);
    expect(s.round.trick!.format!.units.length).toBe(2);
  });

  it('a beatable throw plays only the beatable part; the rest goes back to the hand', () => {
    let s = throwing('none');
    s = act(s, play(0, 'AH', 'QH'));
    const lead = s.round.trick!.plays[0]!;
    expect(lead.cards).toEqual(['QH']);
    expect(lead.badThrowCards).toEqual(['AH']);
    expect(lead.betterPlayer).toBe(1);
    expect(sortCards(s.round.hands[0]!)).toEqual(['2S', 'AH']);
    // Followers now follow a single.
    expect(() => act(s, play(1, 'KH', '3C'))).toThrow(IllegalAction);
    s = act(s, play(1, 'KH'));
    expect(s.round.trick!.winner).toBe(1);
  });

  it('even a partner holding the higher card stops the throw', () => {
    let s = scenario({ hands: [['2S', 'AH', 'QH'], ['3C', '4C', '8C'], ['KH', '4D', '6D'], ['5C', '6C', '7C']], landlord: 0 });
    s = act(s, bid(0, '2S', 1), pickUp(0), exchange(0, []), play(0, 'AH', 'QH'));
    expect(s.round.trick!.plays[0]!.betterPlayer).toBe(2);
  });

  it('the penalty: 10 to the attackers when a defender misses, 10 off them when an attacker does', () => {
    const finish = (s: TractorState) => act(
      s,
      play(1, 'KH'), play(2, '3D'), play(3, '6C'), // seat 1 wins K♥ (10)
      play(1, '3C'), play(2, '4D'), play(3, '7C'), play(0, 'AH'), // seat 3 wins
      play(3, '8C'), play(0, '2S'), play(1, '4C'), play(2, '6D'), // seat 0 trumps
    );
    const none = finish(act(throwing('none'), play(0, 'AH', 'QH')));
    expect(none.round.result!.attackerPoints).toBe(10);
    expect(none.round.result!.penaltyAdjustment).toBe(0);
    const ten = finish(act(throwing('tenPointsPerAttempt'), play(0, 'AH', 'QH')));
    expect(ten.round.penalties[0]).toBe(10);
    expect(ten.round.result!.penaltyAdjustment).toBe(10);
    expect(ten.round.result!.attackerPoints).toBe(20);

    // An attacker's miss: seat 1 wins trick 1 and throws 3♣ 4♣ into seat 3's clubs.
    let s = scenario({
      hands: [['2S', '3H', '9D'], ['KH', '3C', '4C'], ['4H', '4D', '6D'], ['5H', '5C', '6C']],
      landlord: 0,
      config: { throwPenalty: 'tenPointsPerAttempt' },
    });
    s = act(s, bid(0, '2S', 1), pickUp(0), exchange(0, []), play(0, '3H'), play(1, 'KH'), play(2, '4H'), play(3, '5H'));
    s = act(s, play(1, '3C', '4C'));
    expect(s.round.trick!.plays[0]!.cards).toEqual(['3C']);
    s = act(s, play(2, '4D'), play(3, '5C'), play(0, '9D')); // seat 3 wins the 5♣
    s = act(s, play(3, '6C'), play(0, '2S'), play(1, '4C'), play(2, '6D'));
    expect(s.round.result!.penaltyAdjustment).toBe(-10);
    expect(s.round.result!.attackerPoints).toBe(15 + 5 - 10);
  });
});

// ------------------------------------------------------------------ levels and the game

describe('winning a round, levelling up, winning the game', () => {
  it('attackers over the threshold take over, level up, and lead next round', () => {
    const s = attackersTakeOver(['2', '2', '2', '2']);
    const r = s.round.result!;
    expect(r.landlordWon).toBe(false);
    expect(r.attackerDelta).toBe(2); // 90 with one deck: 40 takes over, +1 per 20 after 60
    expect(s.levels).toEqual(['2', '4', '2', '4']);
    expect(r.nextLandlord).toBe(1);
    const next = act(s, { type: 'nextRound' });
    expect(next.landlord).toBe(1);
    expect(next.round.number).toBe(2);
  });

  it("attackers can't pass the final rank without defending it", () => {
    const s = attackersTakeOver(['2', 'A', '2', 'A']);
    expect(s.levels).toEqual(['2', 'A', '2', 'A']);
    expect(s.round.result!.advancements[1]).toMatchObject({ blocked: true, finished: false });
    expect(s.status).toBe('playing');
  });

  /** Landlord seat 0 wins every trick, attackers score nothing. */
  function defendersSweep(levels: Level[], config = {}) {
    const level = levels[0]!;
    const card = `${level}S`; // the landlord declares spades with a card of their level
    let s = scenario({ hands: [[card, 'BJ'], ['3H', '4H'], ['6C', '7C'], ['8D', '9D']], landlord: 0, levels, config });
    s = act(s, bid(0, card, 1), pickUp(0), exchange(0, []));
    return act(s, play(0, 'BJ'), play(1, '3H'), play(2, '6C'), play(3, '8D'), play(0, card), play(1, '4H'), play(2, '7C'), play(3, '9D'));
  }

  it('defenders who hold go up and pass the landlord seat to the next defender (the partner)', () => {
    let s = scenario({ hands: [['5S', 'BJ'], ['3H', '4H'], ['6C', '7C'], ['8D', '9D']], landlord: 0, levels: ['5', '2', '5', '2'] });
    s = act(s, bid(0, '5S', 1), pickUp(0), exchange(0, []), play(0, 'BJ'), play(1, '3H'), play(2, '6C'), play(3, '8D'), play(0, '5S'), play(1, '4H'), play(2, '7C'), play(3, '9D'));
    expect(s.round.result!.landlordDelta).toBe(3);
    expect(s.levels).toEqual(['8', '2', '8', '2']);
    expect(s.round.result!.nextLandlord).toBe(2);
  });

  it('defending the final rank wins the game for the whole team', () => {
    const s = defendersSweep(['A', '2', 'A', '2']);
    expect(s.status).toBe('gameOver');
    expect(s.winners).toEqual([0, 2]);
    expect(() => act(s, { type: 'nextRound' })).toThrow();
  });

  it('a defender reaching the final rank stops there that round, even with levels to spare', () => {
    const s = defendersSweep(['Q', '2', 'Q', '2']);
    expect(s.status).toBe('playing');
    expect(s.levels).toEqual(['A', '2', 'A', '2']);
    expect(s.round.result!.advancements[0]).toMatchObject({ from: 'Q', to: 'A', steps: 2, blocked: true });
  });

  it('with "no defending needed" the game can be won by levels to spare', () => {
    const s = defendersSweep(['K', '2', 'K', '2'], { advancementPolicy: 'fullyUnrestricted' });
    expect(s.status).toBe('gameOver');
  });

  // shengji game_state tests: test_must_defend_sequence_* (stopping where
  // shengji wraps a finished player back to 2 — here that ends the game).
  const step = (from: Level, bump: number, defending: boolean, policy: AdvancementPolicy, max: Level, landlordLevel: Level = from) =>
    advancePlayer(from, bump, defending, landlordLevel, max, policy);

  it('landlord team, one level a round: 2 … A → NT → done (max NT)', () => {
    for (const policy of ['unrestricted', 'fullyUnrestricted', 'defendPoints'] as AdvancementPolicy[]) {
      let lvl: Level = '2';
      const seen: Level[] = [];
      for (let i = 0; i < 20; i++) {
        const a = step(lvl, 1, true, policy, 'NT');
        if (a.finished) break;
        lvl = a.to;
        seen.push(lvl);
      }
      expect(seen).toEqual(['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'NT']);
    }
  });

  it('landlord team, two levels a round (shengji test_must_defend_sequence_landlord_advancing_skip_by_2)', () => {
    const run = (policy: AdvancementPolicy) => {
      let lvl: Level = '2';
      const seen: Level[] = [];
      for (let i = 0; i < 20; i++) {
        const a = step(lvl, 2, true, policy, 'NT');
        if (a.finished) break;
        lvl = a.to;
        seen.push(lvl);
      }
      return seen;
    };
    expect(run('unrestricted')).toEqual(['4', '6', '8', 'T', 'Q', 'A', 'NT']);
    expect(run('fullyUnrestricted')).toEqual(['4', '6', '8', 'T', 'Q', 'A']);
    expect(run('defendPoints')).toEqual(['4', '5', '7', '9', 'T', 'Q', 'K', 'A', 'NT']);
  });

  it('attackers two levels a round get stuck at A (or at 5 when points must be defended)', () => {
    const run = (policy: AdvancementPolicy) => {
      let lvl: Level = '2';
      const seen: Level[] = [];
      for (let i = 0; i < 7; i++) {
        const a = step(lvl, 2, false, policy, 'NT', '2');
        if (a.finished) break;
        lvl = a.to;
        seen.push(lvl);
      }
      return seen;
    };
    expect(run('unrestricted')).toEqual(['4', '6', '8', 'T', 'Q', 'A', 'A']);
    expect(run('defendPoints').slice(0, 3)).toEqual(['4', '5', '5']);
  });

  it('passing NT needs the landlord at NT too (test_must_defend_sequence_landlord_must_defend_to_win)', () => {
    expect(step('NT', 2, true, 'unrestricted', 'NT', '2')).toMatchObject({ to: 'NT', blocked: true, finished: false });
    expect(step('NT', 2, true, 'unrestricted', 'NT', 'NT')).toMatchObject({ finished: true });
  });

  it('both sides levelling under defendPoints (test_player_level_deltas)', () => {
    expect(step('2', 2, true, 'unrestricted', 'NT', '5').to).toBe('4');
    expect(step('4', 2, false, 'defendPoints', 'NT', 'A').to).toBe('5');
    expect(step('5', 2, true, 'defendPoints', 'NT', 'A').to).toBe('7');
    expect(step('5', 2, false, 'defendPoints', 'NT', 'A').to).toBe('5');
    expect(step('7', 2, true, 'defendPoints', 'NT', 'A').to).toBe('9');
  });
});

// ------------------------------------------------------------------ trump is one suit

describe('trump: the trump suit, every trump-number card and both jokers are one suit', () => {
  // ♥ trump, 2s trump: seat 3 is landlord and leads, seat 0 follows.
  function after(lead: Card[], follower: Card[]): TractorState {
    const filler = ['4C', '6C', '9C', 'TC', '3D', '5D'];
    let s = scenario({ hands: [follower, filler, filler, [...filler, ...lead, '2H']], landlord: 3, levels: ['2', '2', '2', '2'] });
    s = act(s, bid(3, '2H'), pickUp(3), exchange(3, []));
    return act(s, play(3, ...lead));
  }
  const ok = (lead: Card[], hand: Card[], cards: Card[]) => expect(playProblemFor(after(lead, hand), 0, cards)).toBeNull();
  const refused = (lead: Card[], hand: Card[], cards: Card[]) => expect(playProblemFor(after(lead, hand), 0, cards)).not.toBeNull();

  it('follows a trump-suit lead with a trump number of any suit, or a joker', () => {
    ok(['5H'], ['2S', '3H', 'KC'], ['2S']);
    ok(['5H'], ['2C', '3H', 'KC'], ['2C']);
    ok(['5H'], ['2H', '3H', 'KC'], ['2H']);
    ok(['5H'], ['LJ', '3H', 'KC'], ['LJ']);
    ok(['5H'], ['BJ', '3H', 'KC'], ['BJ']);
    ok(['5H', '5H'], ['2S', '2S', '3H', 'KC'], ['2S', '2S']);
    ok(['5H', '5H'], ['2S', '2C', '3H', 'KC'], ['2S', '3H']);
    // A 2 is trump, so it must be played on a trump lead before an off-suit card.
    refused(['5H'], ['2S', 'KC'], ['KC']);
    // Holding a trump pair (2♠2♠), a trump pair lead must be answered with a pair.
    refused(['5H', '5H'], ['2S', '2S', '3H', 'KC'], ['2S', '3H']);
  });

  it('follows a trump-number or joker lead with the trump suit', () => {
    ok(['2C'], ['3H', '2S', 'KC'], ['3H']);
    ok(['LJ'], ['3H', 'KC'], ['3H']);
    ok(['BJ'], ['2D', 'KC'], ['2D']);
  });

  it("doesn't count a trump number as its printed suit", () => {
    // 2♠ is trump: on a ♠ lead, a spade has to be played while there is one.
    refused(['5S'], ['2S', 'KS'], ['2S']);
    ok(['5S'], ['2S', 'KS'], ['KS']);
    ok(['5S'], ['2S', 'KC'], ['2S']);
  });

  it('ranks trump suit < trump number < trump-suit number < small joker < big joker, ties to the first played', () => {
    const winner = (plays: Card[][]) => {
      let s = scenario({ hands: plays.map((p, i) => (i === 0 ? [...p, '2H', 'KD'] : [...p, 'KD'])), landlord: 0, levels: ['2', '2', '2', '2'] });
      s = act(s, bid(0, '2H'), pickUp(0), exchange(0, []));
      for (let i = 0; i < 4; i++) s = act(s, play(i, ...plays[i]!));
      return s.round.tricks[s.round.tricks.length - 1]!.winner;
    };
    expect(winner([['KH'], ['AH'], ['2C'], ['3H']])).toBe(2); // any 2 beats the trump-suit ace
    expect(winner([['AH'], ['2S'], ['2C'], ['2D']])).toBe(1); // off-suit 2s tie: the first one wins
    expect(winner([['2D'], ['2S'], ['2C'], ['KH']])).toBe(0); // ... including the lead
    expect(winner([['AH'], ['2S'], ['2H'], ['2C']])).toBe(2); // the trump-suit 2 beats the other 2s
    expect(winner([['2H'], ['LJ'], ['3H'], ['AH']])).toBe(1); // the small joker beats the trump-suit 2
    expect(winner([['2H'], ['LJ'], ['BJ'], ['AH']])).toBe(2); // the big joker beats everything
    expect(winner([['5S'], ['2C'], ['LJ'], ['3H']])).toBe(2); // trumping in: the best trump wins
  });

  it('sorts a hand by printed suit while bidding and with trump as one suit once the kitty is taken', () => {
    const hand: Card[] = ['BJ', '2H', '9S', '2S', 'KH', '3C', 'LJ', '5D', '2C', 'AS'];
    let s = scenario({ hands: [hand, ['4C'], ['4D'], ['4S']], landlord: 0, levels: ['2', '2', '2', '2'] });
    s = act(s, bid(0, '2H'));
    // Bidding (♥ leads): every card in its printed suit — ♣ ♦ ♠ ♥ — then the jokers.
    expect(sortedHand(s, 0)).toEqual(['2C', '3C', '5D', '2S', '9S', 'AS', '2H', 'KH', 'LJ', 'BJ']);
    s = act(s, pickUp(0));
    // Trump settled: ♥ suit, then the off-suit 2s, the ♥2 and the jokers, all together at the end.
    expect(sortedHand(s, 0)).toEqual(['3C', '5D', '9S', 'AS', 'KH', '2S', '2C', '2H', 'LJ', 'BJ']);
  });
});
