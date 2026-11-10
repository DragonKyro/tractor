// Ported from shengji's mechanics/src/trick.rs tests (the ones that don't
// involve bombs or rainbows). Same scenarios, same expected winners / legality.

import { describe, expect, it } from 'vitest';
import type { Card } from './cards';
import {
  DEFAULT_TRACTOR_REQUIREMENTS,
  findPlays,
  formatFromCards,
  fullDecompositionOrdering,
  isLegalFollow,
  matchFormat,
  unitCards,
  type ThrowEvaluationPolicy,
  type TrickDrawPolicy,
  type TrickFormat,
  type TractorRequirements,
} from './format';
import { completeTrick, newTrick, playCards, type TrickRules } from './trick';
import type { Trump } from './trump';

const TRUMP: Trump = { kind: 'standard', number: '4', suit: 'S' };
const ALL_TEP: ThrowEvaluationPolicy[] = ['all', 'highest', 'trickUnitLength'];

function rules(tep: ThrowEvaluationPolicy = 'all', tdp: TrickDrawPolicy = 'noProtections'): TrickRules {
  return { trickDrawPolicy: tdp, throwEvaluationPolicy: tep, tractorRequirements: DEFAULT_TRACTOR_REQUIREMENTS };
}

/** Play a 4-seat trick (seat 0 leads); `plays[i]` is what seat i plays. */
function runTrick(trump: Trump, hands: Card[][], plays: Card[][], r: TrickRules) {
  let trick = newTrick(0, hands.length);
  const hs = hands.map((h) => h.slice());
  for (let seat = 0; seat < plays.length; seat++) {
    const out = playCards(trick, trump, hs, seat, plays[seat]!, r);
    trick = out.trick;
    hs[seat] = out.hand;
  }
  return { result: completeTrick(trick), trick };
}

const sorted = (cs: Card[]) => cs.slice().sort();

describe('lead decomposition (test_play_formats)', () => {
  const check = (cards: Card[], expected: Card[][][], req: TractorRequirements = DEFAULT_TRACTOR_REQUIREMENTS) => {
    const got = new Set(
      findPlays(TRUMP, req, cards).map((units) => JSON.stringify(units.map(unitCards).map(sorted).sort())),
    );
    const want = new Set(expected.map((units) => JSON.stringify(units.map(sorted).sort())));
    expect(got).toEqual(want);
  };
  it('reads plays the way shengji does', () => {
    check(['2H', '3H', '7H'], [[['7H'], ['3H'], ['2H']]]);
    check(['2H', '2H', '2H'], [[['2H', '2H', '2H']]]);
    check(['2H', '2H', '3H', '3H'], [[['2H', '2H', '3H', '3H']]]);
    check(['2H', '2H', '3H', '3H'], [[['3H', '3H'], ['2H', '2H']]], { minLength: 3, minCount: 2 });
    check(['2H', '2H', '3H', '3H', '5H', '5H'], [[['2H', '2H', '3H', '3H', '5H', '5H']]], { minLength: 3, minCount: 2 });
    check(['2H', '2H', '3H', '3H'], [[['3H', '3H'], ['2H', '2H']]], { minLength: 3, minCount: 3 });
    check(['2H', '2H', '2H', '3H', '3H', '3H'], [[['2H', '2H', '2H', '3H', '3H', '3H']]], { minLength: 2, minCount: 3 });
    check(['2H', '2H', '2H', '3H', '3H'], [
      [['2H'], ['2H', '2H', '3H', '3H']],
      [['3H', '3H'], ['2H', '2H', '2H']],
    ]);
    check(['2H', '2H', '3H', '3H', '3H'], [
      [['3H'], ['2H', '2H', '3H', '3H']],
      [['3H', '3H', '3H'], ['2H', '2H']],
    ]);
    check(['4H', '4H', '4S', '4S'], [[['4H', '4H', '4S', '4S']]]);
    check(['4H', '4H', 'AS', 'AS'], [[['AS', 'AS', '4H', '4H']]]);
    check(['QS', 'QS', 'KS', 'KS', 'AS'], [[['AS'], ['QS', 'QS', 'KS', 'KS']]]);
    check(['3H', '3H', '3H', '5H', '5H', '5H'], [[['3H', '3H', '3H', '5H', '5H', '5H']]]);
    check(['2H', '2H', '3H', '3H', '3H', '5H', '5H', '5H'], [
      [['5H', '5H', '5H'], ['3H'], ['2H', '2H', '3H', '3H']],
      [['3H', '3H', '3H', '5H', '5H', '5H'], ['2H', '2H']],
      [['5H'], ['3H'], ['2H', '2H', '3H', '3H', '5H', '5H']],
    ]);
  });
});

describe('decomposition ordering', () => {
  it('orders 4 cards the shengji way', () => {
    expect(fullDecompositionOrdering(4)).toEqual([[[4]], [[3], [1]], [[2, 2]], [[2], [2]], [[2], [1], [1]], [[1], [1], [1], [1]]]);
  });
});

describe('simple tricks', () => {
  for (const tep of ALL_TEP) {
    it(`singles (${tep})`, () => {
      const hands = [['2S', '3S', '5S'], ['2S', '3S', '5S'], ['2S', '3S', '5S'], ['2S', '3S', '5S']];
      const { result } = runTrick(TRUMP, hands, [['2S'], ['5S'], ['3S'], ['5S']], rules(tep));
      expect(result.winner).toBe(1);
      expect(result.largestUnitSize).toBe(1);
      expect(result.pointCards).toEqual(['5S', '5S']);
    });
    it(`trump (${tep})`, () => {
      const hands = [['2S', '3S', '5S'], ['2H', '3H', '4S'], ['2S', '3S', '5S'], ['2S', '3S', '5S']];
      const { result } = runTrick(TRUMP, hands, [['2S'], ['4S'], ['3S'], ['5S']], rules(tep));
      expect(result.winner).toBe(1);
      expect(result.pointCards).toEqual(['5S']);
    });
    it(`pairs (${tep})`, () => {
      const hands = [['2S', '2S', '5S'], ['2H', '3S', '4S'], ['5S', '5S', '5S'], ['3S', '4S', '5S']];
      const { result } = runTrick(TRUMP, hands, [['2S', '2S'], ['3S', '4S'], ['5S', '5S'], ['3S', '5S']], rules(tep));
      expect(result.winner).toBe(2);
      expect(result.largestUnitSize).toBe(2);
      expect(result.points).toBe(15);
    });
    it(`tractor (${tep})`, () => {
      const hands = [
        ['2S', '2S', '3S', '3S', '4S'],
        ['6S', '6S', '7S', '7S', '4S'],
        ['2S', '5S', '5S', '5S', '4S'],
        ['6S', '6S', '6S', '6S', '4S'],
      ];
      const plays = [['2S', '2S', '3S', '3S'], ['6S', '6S', '7S', '7S'], ['2S', '5S', '5S', '5S'], ['6S', '6S', '6S', '6S']];
      const { result } = runTrick(TRUMP, hands, plays, rules(tep));
      expect(result.winner).toBe(1);
      expect(result.largestUnitSize).toBe(4);
      expect(result.points).toBe(15);
    });
  }
});

describe('throws', () => {
  it('a standing throw is beaten by trump that matches it', () => {
    const hands = [['8H', '8H', '7H', '2H'], ['2H', '2S', '2S', '2S'], ['2S', '2S', '3S', '4S'], ['4S', '4S', '4S', '4S']];
    const { result } = runTrick(TRUMP, hands, hands, rules());
    expect(result.largestUnitSize).toBe(2);
    expect(result.winner).toBe(2);
    expect(result.points).toBe(0);
  });

  for (const tep of ALL_TEP) {
    it(`double over-trump (${tep})`, () => {
      const trump: Trump = { kind: 'standard', suit: 'H', number: '8' };
      const hands = [
        ['AC', 'AC', 'QC', 'QC', 'TC', 'TC'],
        ['8S', '8S', '9H', '9H', '3H', '3H'],
        ['8H', '8H', 'KH', 'KH', 'TH', 'TH'],
        ['LJ', 'LJ', '8H', '8H', 'KH', 'KH'],
      ];
      const { result } = runTrick(trump, hands, hands, rules(tep));
      expect(result.winner).toBe(3);
    });
  }

  it('a failed throw is cut down to the beatable unit', () => {
    const hands = [['8H', '8H', '7H', '2H'], ['2H', '2S', '2S', '2S'], ['2S', '2S', '3S', '4S'], ['4S', '4S', '4S', '3H']];
    let trick = newTrick(0, 4);
    const hs = hands.map((h) => h.slice());
    const lead = playCards(trick, TRUMP, hs, 0, ['8H', '8H', '7H', '2H'], rules());
    expect(lead.failedThrow).not.toBeNull();
    expect(lead.failedThrow!.forced).toEqual(['2H']);
    expect(lead.failedThrow!.betterPlayer).toBe(3);
    expect(sorted(lead.hand)).toEqual(sorted(['8H', '8H', '7H']));
    trick = lead.trick;
    hs[0] = lead.hand;
    for (const [seat, cards] of [[1, ['2H']], [2, ['3S']], [3, ['3H']]] as const) {
      const out = playCards(trick, TRUMP, hs, seat, [...cards], rules());
      trick = out.trick;
      hs[seat] = out.hand;
    }
    const result = completeTrick(trick);
    expect(result.largestUnitSize).toBe(1);
    expect(result.winner).toBe(2);
    expect(result.failedThrowSize).toBe(3);
  });

  it('a tractor + single throw stands when nobody can beat a unit', () => {
    const hands = [
      ['QS', 'QS', 'KS', 'KS', 'AS'],
      ['2S', '3S', '3S', '5S', '3H'],
      ['AS', 'AS', '3H', '3H', '3H'],
      ['3H', '3H', '3H', '3H', '3H'],
    ];
    const { result } = runTrick(TRUMP, hands, hands, rules());
    expect(result.largestUnitSize).toBe(4);
    expect(result.winner).toBe(0);
    expect(result.points).toBe(25);
    expect(result.failedThrowSize).toBe(0);
  });

  it('a throw with a tractor is beaten by a trump tractor (other tractor in game)', () => {
    const trump: Trump = { kind: 'standard', number: '4', suit: 'H' };
    const hands = [
      ['QS', 'QS', 'KS', 'KS', 'AS'],
      ['2H', '2H', '3H', 'AH', '3H'],
      ['AS', 'AS', '3S', '3S', '3S'],
      ['3S', '3S', '3S', '3S', '3S'],
    ];
    const { result } = runTrick(trump, hands, hands, rules());
    expect(result.largestUnitSize).toBe(4);
    expect(result.winner).toBe(1);
    expect(sorted(result.pointCards)).toEqual(['KS', 'KS']);
  });

  it('throw evaluation: highest card', () => {
    const trump: Trump = { kind: 'standard', number: 'K', suit: 'S' };
    const hands = [['4C', '6C'], ['2S', '3S'], ['3S', '4S'], ['2S', 'BJ']];
    expect(runTrick(trump, hands, hands, rules('all')).result.winner).toBe(2);
    expect(runTrick(trump, hands, hands, rules('highest')).result.winner).toBe(3);
  });

  it('throw evaluation: trick unit length', () => {
    const trump: Trump = { kind: 'standard', number: '2', suit: 'S' };
    const hands = [
      ['AH', 'KH', 'KH', 'KH', '9H', '9H'],
      ['5S', '5S', '5S', 'QS', 'QS', 'AS'],
      ['5S', '5S', '5S', 'AS', 'AS', '4S'],
      ['4S', '4S', '4S', '3S', '3S', 'BJ'],
    ];
    expect(runTrick(trump, hands, hands, rules('all')).result.winner).toBe(1);
    expect(runTrick(trump, hands, hands, rules('highest')).result.winner).toBe(3);
    expect(runTrick(trump, hands, hands, rules('trickUnitLength')).result.winner).toBe(2);
  });

  it('throw of trump-number cards in trump', () => {
    const hands = [['4H', '4S'], ['4D', '2S'], ['3S', '3S'], ['3S', '3S']];
    const { result } = runTrick(TRUMP, hands, hands, rules());
    expect(result.largestUnitSize).toBe(1);
    expect(result.winner).toBe(0);
  });

  for (const tep of ALL_TEP) {
    it(`trump beats a thrown pair of singles (${tep})`, () => {
      const trump: Trump = { kind: 'standard', number: '5', suit: 'D' };
      const hands = [['AC', 'KC'], ['5C', 'AD'], ['LJ', '5S'], ['QS', 'AD']];
      expect(runTrick(trump, hands, hands, rules(tep)).result.winner).toBe(2);
    });
    it(`ambiguous follow parse picks the winning reading (${tep})`, () => {
      const trump: Trump = { kind: 'standard', number: '2', suit: 'C' };
      const hands = [['8D', '8D', 'JD', 'QD'], ['5C', '5C', 'AC', '2S'], ['3S', '3S', '7C', '8C'], ['6C', '6C', '2C', '2C']];
      expect(runTrick(trump, hands, hands, rules(tep)).result.winner).toBe(3);
    });
  }
});

describe('a follow that can be read two ways', () => {
  // ♠ trump. Seat 0 throws A♥A♥ K♥ Q♥ (stands: nobody has hearts).
  // Seat 1 trumps with 7♠7♠9♠9♠ = "pair 9s + 7, 7" or "pair 7s + 9, 9".
  const trump: Trump = { kind: 'standard', suit: 'S', number: '2' };
  const lead = ['AH', 'AH', 'KH', 'QH'];
  const seat1 = ['7S', '7S', '9S', '9S'];
  const clubs = ['3C', '4C', '5C', '6C'];

  it('a later play must beat every reading of the winner’s cards (ties and ambiguity go to the earlier player)', () => {
    // 8♠8♠ J♠ 10♠ beats "pair 7s + 9, 9" but not "pair 9s + 7, 7".
    const { result } = runTrick(trump, [lead, seat1, ['8S', '8S', 'JS', 'TS'], clubs], [lead, seat1, ['8S', '8S', 'JS', 'TS'], clubs], rules());
    expect(result.winner).toBe(1);
  });

  it('a play that beats every reading takes the trick', () => {
    const seat2 = ['TS', 'TS', 'QS', 'JS'];
    const { result } = runTrick(trump, [lead, seat1, seat2, clubs], [lead, seat1, seat2, clubs], rules());
    expect(result.winner).toBe(2);
  });
});

describe('trick formats', () => {
  it('basic triple', () => {
    const tf = formatFromCards(TRUMP, DEFAULT_TRACTOR_REQUIREMENTS, ['2S', '2S', '2S']);
    expect(tf).toEqual({ suit: 'T', units: [{ kind: 'repeated', count: 3, card: '2S' }] });
    expect(matchFormat(TRUMP, tf, ['2S', '2S', '2S']).length).toBeGreaterThan(0);
    expect(matchFormat(TRUMP, tf, ['2S', '2S']).length).toBe(0);
  });
  it('tractor of triples skips the trump number', () => {
    const tf = formatFromCards(TRUMP, DEFAULT_TRACTOR_REQUIREMENTS, ['2S', '2S', '2S', '3S', '3S', '3S', '5S', '5S', '5S']);
    expect(tf).toEqual({ suit: 'T', units: [{ kind: 'tractor', count: 3, members: ['2S', '3S', '5S'] }] });
    expect(matchFormat(TRUMP, tf, ['3S', '3S', '3S', '5S', '5S', '5S', '6S', '6S', '6S']).length).toBeGreaterThan(0);
    expect(matchFormat(TRUMP, tf, ['2S', '2S', '2S', '3S', '3S', '3S', '6S', '6S', '6S']).length).toBe(0);
  });
  it('tractor + long tuple throw', () => {
    const cards = ['2S', '2S', '2S', '2S', '2S', '2S', '2S', '3S', '3S', '5S', '5S'];
    const tf = formatFromCards(TRUMP, DEFAULT_TRACTOR_REQUIREMENTS, cards);
    expect(tf.units).toEqual([
      { kind: 'tractor', count: 2, members: ['3S', '5S'] },
      { kind: 'repeated', count: 7, card: '2S' },
    ]);
    expect(matchFormat(TRUMP, tf, ['8S', '8S', '8S', '8S', '8S', '8S', '8S', '3S', '3S', '5S', '5S']).length).toBeGreaterThan(0);
    const tf2 = formatFromCards(TRUMP, DEFAULT_TRACTOR_REQUIREMENTS, ['2S', '2S', '3S', '3S', '5S', '5S', '8S', '8S', '8S']);
    expect(matchFormat(TRUMP, tf2, ['2S', '2S', '2S', '2S', '2S', '3S', '3S', '5S', '5S']).length).toBeGreaterThan(0);
  });
  it('simple throw', () => {
    const tf = formatFromCards(TRUMP, DEFAULT_TRACTOR_REQUIREMENTS, ['2S', '2S', '2S', '3S', '5S', '5S', '5S']);
    expect(tf.units).toEqual([
      { kind: 'repeated', count: 1, card: '3S' },
      { kind: 'repeated', count: 3, card: '2S' },
      { kind: 'repeated', count: 3, card: '5S' },
    ]);
    expect(matchFormat(TRUMP, tf, ['5S', '5S', '5S', '3S', '3S', '3S', '2S']).length).toBeGreaterThan(0);
  });
});

describe('legal follows', () => {
  const pairOf3: TrickFormat = { suit: 'T', units: [{ kind: 'repeated', count: 2, card: '3S' }] };
  const legal = (tf: TrickFormat, hand: Card[], play: Card[], p: TrickDrawPolicy) => isLegalFollow(TRUMP, tf, hand, play, p);

  it('pairs must be followed by pairs', () => {
    const hand = ['2S', '2S', '3S', '3S', '5S', '5S'];
    expect(legal(pairOf3, hand, ['2S', '2S'], 'noProtections')).toBe(true);
    expect(legal(pairOf3, hand, ['2S', '3S'], 'noProtections')).toBe(false);
    expect(legal(pairOf3, hand, ['2S', '3S', '3S'], 'noProtections')).toBe(false);
    expect(legal(pairOf3, hand, ['2S', '2S'], 'noFormatBasedDraw')).toBe(true);
    expect(legal(pairOf3, hand, ['2S', '3S'], 'noFormatBasedDraw')).toBe(true);
    expect(legal(pairOf3, hand, ['2S', '3S', '3S'], 'noFormatBasedDraw')).toBe(false);
  });

  it("longer tuples aren't broken when protected", () => {
    const hand = ['2S', '2S', '2S', '3S', '5S'];
    expect(legal(pairOf3, hand, ['3S', '5S'], 'longerTuplesProtected')).toBe(true);
    expect(legal(pairOf3, hand, ['3S', '5S'], 'noFormatBasedDraw')).toBe(true);
    expect(legal(pairOf3, hand, ['3S', '5S'], 'noProtections')).toBe(false);
  });

  it('triples, quintuples and tractors fall back the shengji way', () => {
    const triple: TrickFormat = { suit: 'T', units: [{ kind: 'repeated', count: 3, card: '3S' }] };
    let hand = ['2S', '2S', '3S', '3S', '5S', '5S'];
    expect(legal(triple, hand, ['2S', '2S', '5S'], 'noProtections')).toBe(true);
    expect(legal(triple, hand, ['2S', '3S', '5S'], 'noProtections')).toBe(false);
    expect(legal(triple, hand, ['2S', '3S', '5S'], 'noFormatBasedDraw')).toBe(true);

    const five: TrickFormat = { suit: 'T', units: [{ kind: 'repeated', count: 5, card: '3S' }] };
    expect(legal(five, hand, ['2S', '2S', '3S', '3S', '5S'], 'noProtections')).toBe(true);
    hand = ['2S', '2S', '2S', '2S', '3S', '3S', '5S', '5S'];
    expect(legal(five, hand, ['2S', '2S', '2S', '2S', '5S'], 'noProtections')).toBe(true);

    const tractor: TrickFormat = { suit: 'T', units: [{ kind: 'tractor', count: 2, members: ['2S', '3S'] }] };
    expect(legal(tractor, hand, ['2S', '2S', '2S', '2S'], 'noProtections')).toBe(false);
    expect(legal(tractor, hand, ['2S', '2S', '3S', '3S'], 'noProtections')).toBe(true);
    expect(legal(tractor, hand, ['3S', '3S', '5S', '5S'], 'noProtections')).toBe(true);
    expect(legal(tractor, hand, ['2S', '2S', '2S', '2S'], 'longerTuplesProtected')).toBe(false);
    expect(legal(tractor, hand, ['2S', '2S', '3S', '3S'], 'longerTuplesProtected')).toBe(true);
    expect(legal(tractor, hand, ['2S', '2S', '2S', '2S'], 'longerTuplesProtectedAndOnlyDrawTractorOnTractor')).toBe(false);
    expect(legal(tractor, hand, ['2S', '2S', '2S', '2S'], 'noFormatBasedDraw')).toBe(true);

    hand = ['2S', '2S', '2S', '2S', '3S', '5S', '5S'];
    expect(legal(tractor, hand, ['2S', '2S', '2S', '2S'], 'noProtections')).toBe(true);
    expect(legal(tractor, hand, ['2S', '2S', '5S', '5S'], 'noProtections')).toBe(true);
    expect(legal(tractor, hand, ['2S', '2S', '5S', '3S'], 'noProtections')).toBe(false);
    expect(legal(tractor, hand, ['2S', '2S', '5S', '3S'], 'noFormatBasedDraw')).toBe(true);
    expect(legal(tractor, hand, ['2S', '2S', '2S', '2S'], 'longerTuplesProtected')).toBe(true);
    expect(legal(tractor, hand, ['2S', '2S', '5S', '5S'], 'longerTuplesProtected')).toBe(true);
    expect(legal(tractor, hand, ['2S', '2S', '5S', '3S'], 'longerTuplesProtected')).toBe(true);

    const pairSingle: TrickFormat = {
      suit: 'T',
      units: [
        { kind: 'repeated', count: 2, card: '2S' },
        { kind: 'repeated', count: 1, card: '3S' },
      ],
    };
    hand = ['2S', '2S', '2S', '5S'];
    for (const p of ['noProtections', 'noFormatBasedDraw', 'longerTuplesProtected'] as TrickDrawPolicy[]) {
      expect(legal(pairSingle, hand, ['2S', '2S', '2S'], p)).toBe(true);
      expect(legal(pairSingle, hand, ['2S', '2S', '5S'], p)).toBe(true);
    }
  });

  it('protected tuples (test_protected_tuple)', () => {
    const triple: TrickFormat = { suit: 'T', units: [{ kind: 'repeated', count: 3, card: '3S' }] };
    let hand = ['2S', '2S', '2S', '2S', '5S', '6S', '7S', '8S'];
    expect(legal(triple, hand, ['6S', '7S', '8S'], 'noProtections')).toBe(false);
    expect(legal(triple, hand, ['6S', '7S', '8S'], 'noFormatBasedDraw')).toBe(true);
    expect(legal(triple, hand, ['6S', '7S', '8S'], 'longerTuplesProtected')).toBe(true);
    hand = ['2S', '2S', '2S', '2S', '5S', '5S', '6S', '7S', '8S'];
    expect(legal(triple, hand, ['5S', '5S', '6S'], 'noProtections')).toBe(false);
    expect(legal(triple, hand, ['5S', '5S', '6S'], 'longerTuplesProtected')).toBe(true);
    expect(legal(triple, hand, ['5S', '5S', '6S'], 'longerTuplesProtectedAndOnlyDrawTractorOnTractor')).toBe(true);
    expect(legal(triple, hand, ['6S', '7S', '8S'], 'longerTuplesProtected')).toBe(false);
  });

  it('protected wider tractor', () => {
    const tf: TrickFormat = { suit: 'T', units: [{ kind: 'tractor', count: 2, members: ['6S', '7S'] }] };
    const hand = ['2S', '2S', '2S', '3S', '3S', '3S', '5S', '6S', '7S', '8S'];
    const play = ['5S', '6S', '7S', '8S'];
    expect(legal(tf, hand, play, 'noProtections')).toBe(false);
    expect(legal(tf, hand, play, 'noFormatBasedDraw')).toBe(true);
    expect(legal(tf, hand, play, 'longerTuplesProtected')).toBe(true);
    expect(legal(tf, hand, play, 'onlyDrawTractorOnTractor')).toBe(false);
    expect(legal(tf, hand, play, 'longerTuplesProtectedAndOnlyDrawTractorOnTractor')).toBe(true);
  });

  it('protected tractor vs triple (off-suit lead)', () => {
    const heartTrump: Trump = { kind: 'standard', number: '4', suit: 'H' };
    const tf: TrickFormat = {
      suit: 'S',
      units: [
        { kind: 'tractor', count: 2, members: ['9S', '9S'] },
        { kind: 'repeated', count: 1, card: 'KS' },
      ],
    };
    const hand = ['3S', '5S', 'TS', 'JS', 'QS', '6S', '8S', '8S', '8S'];
    const l = (play: Card[], p: TrickDrawPolicy) => isLegalFollow(heartTrump, tf, hand, play, p);
    expect(l(['3S', '5S', 'TS', 'JS', 'QS'], 'noProtections')).toBe(false);
    expect(l(['3S', '5S', 'TS', 'JS', 'QS'], 'noFormatBasedDraw')).toBe(true);
    expect(l(['3S', '5S', 'TS', 'JS', 'QS'], 'longerTuplesProtected')).toBe(true);
    expect(l(['3S', '6S', '8S', '8S', '8S'], 'noProtections')).toBe(true);
    expect(l(['3S', '6S', '8S', '8S', '8S'], 'longerTuplesProtected')).toBe(true);
  });

  it("a longer tuple in the play doesn't hide a genuine pair left in the hand", () => {
    const heartTrump: Trump = { kind: 'standard', number: '4', suit: 'H' };
    const tf: TrickFormat = {
      suit: 'S',
      units: [
        { kind: 'tractor', count: 2, members: ['6S', '7S'] },
        { kind: 'repeated', count: 1, card: '5S' },
      ],
    };
    const play = ['6S', '6S', '6S', '8S', '8S'];
    expect(isLegalFollow(heartTrump, tf, play, play, 'longerTuplesProtected')).toBe(true);
    expect(isLegalFollow(heartTrump, tf, play, play, 'longerTuplesProtectedAndOnlyDrawTractorOnTractor')).toBe(true);
    expect(isLegalFollow(heartTrump, tf, [...play, 'JS', 'JS'], play, 'longerTuplesProtected')).toBe(false);

    const tractor3 = formatFromCards(TRUMP, DEFAULT_TRACTOR_REQUIREMENTS, ['TH', 'TH', 'JH', 'JH', 'QH', 'QH']);
    const hand = ['KH', 'KH', '2H', '2H', '2H', '5H', '6H', '7H', '8H'];
    expect(isLegalFollow(TRUMP, tractor3, hand, ['2H', '2H', '2H', '5H', '6H', '7H'], 'longerTuplesProtected')).toBe(false);
    expect(isLegalFollow(TRUMP, tractor3, hand, ['2H', '2H', 'KH', 'KH', '5H', '6H'], 'longerTuplesProtected')).toBe(true);
    const hand2 = ['2H', '2H', '2H', '5H', '6H', '7H', '8H', '9H', 'JH'];
    expect(isLegalFollow(TRUMP, tractor3, hand2, ['2H', '2H', '2H', '5H', '6H', '7H'], 'longerTuplesProtected')).toBe(true);
  });

  it('the longer-tuple guard needs enough genuine tuples', () => {
    const tractor: TrickFormat = { suit: 'T', units: [{ kind: 'tractor', count: 2, members: ['2S', '3S'] }] };
    const pairSingle: TrickFormat = {
      suit: 'T',
      units: [
        { kind: 'repeated', count: 2, card: '3S' },
        { kind: 'repeated', count: 1, card: '5S' },
      ],
    };
    expect(legal(tractor, ['2S', '2S', '2S', '2S', '3S', '5S', '5S'], ['2S', '2S', '2S', '2S'], 'longerTuplesProtected')).toBe(true);
    const withPair = ['2S', '2S', '2S', '5S', '5S', '6S'];
    expect(legal(pairSingle, withPair, ['2S', '2S', '2S'], 'longerTuplesProtected')).toBe(false);
    expect(legal(pairSingle, withPair, ['5S', '5S', '2S'], 'longerTuplesProtected')).toBe(true);
    expect(legal(pairSingle, ['2S', '2S', '2S', '6S', '7S', '8S'], ['2S', '2S', '2S'], 'longerTuplesProtected')).toBe(true);
  });

  it('a long tractor lead draws out pairs', () => {
    const trump: Trump = { kind: 'standard', number: 'K', suit: 'S' };
    const hands = [
      ['7S', '7S', '8S', '8S', '9S', '9S', '4C', '4C'],
      ['4S', 'TS', 'AS', 'KH', 'KD', 'KC', 'KS', 'KS'],
      ['2S', '6S', 'JS', 'QS', 'KH', 'KC', 'TC', 'LJ'],
      ['4C', '6C', '7C', '3S', '3S', 'BJ', '5C', '8C'],
    ];
    for (const policy of [
      'noProtections',
      'longerTuplesProtected',
      'noFormatBasedDraw',
      'onlyDrawTractorOnTractor',
      'longerTuplesProtectedAndOnlyDrawTractorOnTractor',
    ] as TrickDrawPolicy[]) {
      const r = rules('all', policy);
      let trick = newTrick(0, 4);
      const hs = hands.map((h) => h.slice());
      const step = (seat: number, cards: Card[]) => {
        const out = playCards(trick, trump, hs, seat, cards, r);
        trick = out.trick;
        hs[seat] = out.hand;
      };
      step(0, ['7S', '7S', '8S', '8S', '9S', '9S']);
      if (policy === 'noFormatBasedDraw') {
        step(1, ['4S', 'TS', 'AS', 'KH', 'KD', 'KC']);
      } else {
        expect(() => step(1, ['4S', 'TS', 'AS', 'KH', 'KD', 'KC'])).toThrow();
        step(1, ['4S', 'TS', 'AS', 'KH', 'KS', 'KS']);
      }
      step(2, ['2S', '6S', 'JS', 'QS', 'KH', 'KC']);
      step(3, ['4C', '6C', '7C', '3S', '3S', 'BJ']);
    }
  });
});
