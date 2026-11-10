import { describe, expect, it } from 'vitest';
import { defaultConfig } from '@/game/config';
import { applyAction, createInitialState, drawingSeat, IllegalAction } from '@/game/engine';
import { act, bid, scenario } from '@/game/testing';
import { useGameStore } from '@/store/gameStore';
import { nextAutomatedStep } from './driver';
import { aiSeats } from './simulate';

const humanAt0 = () => aiSeats(4).map((s, i) => (i === 0 ? { ...s, isAI: false, id: 'me' } : s));

describe('the live deal', () => {
  it('deals one card at a time, to whoever is next', () => {
    let s = createInitialState(defaultConfig(4), 3, aiSeats(4));
    const first = drawingSeat(s)!;
    expect(() => applyAction(s, { type: 'draw', seat: (first + 1) % 4 })).toThrow(IllegalAction);
    s = applyAction(s, { type: 'draw', seat: first });
    expect(s.round.hands[first]!.length).toBe(1);
    expect(drawingSeat(s)).toBe((first + 1) % 4);
  });

  it('the driver draws for AI seats and waits on a human’s own draw', () => {
    let s = createInitialState(defaultConfig(4), 5, humanAt0());
    const isAI = (i: number) => s.seats[i]!.isAI;
    for (let i = 0; i < 12; i++) {
      const step = nextAutomatedStep(s, isAI);
      if (drawingSeat(s) === 0) {
        expect(step?.action.type === 'draw').toBe(false);
        s = applyAction(s, { type: 'draw', seat: 0 });
      } else {
        s = applyAction(s, step!.action);
      }
    }
    expect(s.round.hands[0]!.length).toBe(3);
  });

  it('holds the bidding window open until the humans say they’re done', () => {
    const hands = [['2H', '3C'], ['2S', '4C'], ['5C', '6C'], ['7C', '8C']];
    let s = scenario({ hands });
    s.seats[0] = { ...s.seats[0]!, isAI: false };
    s = act(s, bid(1, '2S', 1)); // AI seat 1 would take the kitty…
    const isAI = (i: number) => s.seats[i]!.isAI;
    expect(nextAutomatedStep(s, isAI)).toMatchObject({ action: { type: 'pickUpKitty', seat: 1 }, pace: 'bidWindow' });
    s = act(s, { type: 'doneBidding', seat: 0 });
    expect(nextAutomatedStep(s, isAI)).toMatchObject({ action: { type: 'pickUpKitty' }, pace: 'ai' });
  });

  it('an overcall reopens the window for everyone', () => {
    let s = act(scenario({ hands: [['2H', '2H'], ['2S', '4C'], ['5C', '6C'], ['7C', '8C']] }), bid(1, '2S', 1));
    s = act(s, { type: 'doneBidding', seat: 2 });
    expect(s.round.bidDone).toEqual([2]);
    s = act(s, bid(0, '2H', 2));
    expect(s.round.bidDone).toEqual([]);
    // The landlord-to-be has nothing to finish.
    expect(() => act(s, { type: 'doneBidding', seat: 0 })).toThrow(IllegalAction);
  });
});

describe('the match record', () => {
  it('keeps every action and trims back on a rollback', () => {
    const store = useGameStore.getState();
    const s0 = createInitialState(defaultConfig(4), 9, aiSeats(4));
    store.loadGame(s0, null, null);
    const step = () => {
      const cur = useGameStore.getState().state!;
      const a = nextAutomatedStep(cur, () => true)!.action;
      expect(useGameStore.getState().dispatch(a)).toBeNull();
      return cur;
    };
    step();
    const before = step();
    step();
    expect(useGameStore.getState().actions.length).toBe(3);
    useGameStore.getState().restoreState(before);
    expect(useGameStore.getState().actions.length).toBe(1);
    expect(useGameStore.getState().initial).toBe(s0);
    useGameStore.getState().clear();
  });
});
