// Holds the current match's state. dispatch() applies an action locally and,
// when online, hands it (with the state it was applied to) to the
// networkStore, which writes it to the room's log. applyLocal() is the
// receiver path: silent apply, no broadcast. Both run the same reducer so
// every device stays in step. restoreState() rolls back (online, when
// another device's move took the log slot this device's move was headed
// for). States are immutable, so a snapshot is just the state reference.
//
// The store also records the match — its first state plus every action
// applied — so the review screen can replay it. A rollback trims the record
// back to the restored state.

import { create } from 'zustand';
import type { PlayerId, Seat } from '@/core/types';
import type { TractorConfig } from '@/game/config';
import { applyAction } from '@/game/engine';
import type { Action, TractorState } from '@/game/types';

/** `fromAI`: the AI driver made this move (online, a dropped AI move needs no notice). */
type BroadcastHandler = (action: Action, before: TractorState, fromAI: boolean) => void;
/** Online: true when this device may not move at all (a spectator). */
type DispatchGuard = () => boolean;

export interface MatchSetup {
  config: TractorConfig;
  seats: Seat[];
}

/** How many recorded actions led to each state (for trimming the record on rollback). */
const stepOf = new WeakMap<TractorState, number>();

interface GameStore {
  state: TractorState | null;
  /** The match's opening state and every action since, in order. */
  initial: TractorState | null;
  actions: Action[];
  /** The human seat this device plays (online), or the hot-seat table's first human. */
  localPlayerId: PlayerId | null;
  /** The options + lineup the match started from, for "Play again". */
  setup: MatchSetup | null;
  /** False while this device is loading the AI network for a match it didn't start (an online stand-in). */
  aiReady: boolean;

  loadGame(state: TractorState, localPlayerId: PlayerId | null, setup: MatchSetup | null): void;
  /** Apply a move (and log it, online). Returns why it was refused, or null. `{ ai: true }` marks the AI driver's moves. */
  dispatch(action: Action, opts?: { ai?: boolean }): string | null;
  /** Apply a move without broadcasting it. False (and nothing changes) when the reducer rejects it. */
  applyLocal(action: Action): boolean;
  restoreState(state: TractorState): void;
  setAiReady(ready: boolean): void;
  clear(): void;

  registerBroadcastHandler(fn: BroadcastHandler | null): void;
  registerDispatchGuard(fn: DispatchGuard | null): void;
  _broadcast: BroadcastHandler | null;
  _blocked: DispatchGuard | null;
}

function record(actions: Action[], before: TractorState, action: Action, next: TractorState): Action[] {
  const at = stepOf.get(before) ?? actions.length;
  const out = [...actions.slice(0, at), action];
  stepOf.set(next, out.length);
  return out;
}

export const useGameStore = create<GameStore>((set, get) => ({
  state: null,
  initial: null,
  actions: [],
  localPlayerId: null,
  setup: null,
  aiReady: true,
  _broadcast: null,
  _blocked: null,

  loadGame: (state, localPlayerId, setup) => {
    stepOf.set(state, 0);
    set({ state, localPlayerId, setup, aiReady: true, initial: state, actions: [] });
  },

  dispatch: (action, opts) => {
    const { state, _broadcast, _blocked } = get();
    if (!state) return 'No match in progress.';
    if (_blocked?.()) return 'Spectators cannot move.';
    let next: TractorState;
    try {
      next = applyAction(state, action);
    } catch (e) {
      return (e as Error).message;
    }
    set({ state: next, actions: record(get().actions, state, action, next) });
    _broadcast?.(action, state, opts?.ai === true);
    return null;
  },

  applyLocal: (action) => {
    const { state } = get();
    if (!state) return false;
    try {
      const next = applyAction(state, action);
      set({ state: next, actions: record(get().actions, state, action, next) });
      return true;
    } catch (e) {
      console.warn('[game] move rejected:', (e as Error).message);
      return false;
    }
  },

  restoreState: (state) => {
    const at = stepOf.get(state);
    set({ state, actions: at === undefined ? get().actions : get().actions.slice(0, at) });
  },
  setAiReady: (ready) => set({ aiReady: ready }),
  clear: () => set({ state: null, localPlayerId: null, setup: null, aiReady: true, initial: null, actions: [] }),

  registerBroadcastHandler: (fn) => set({ _broadcast: fn }),
  registerDispatchGuard: (fn) => set({ _blocked: fn }),
}));
