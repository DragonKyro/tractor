// Holds the current match's state. dispatch() applies an action locally and,
// when online, hands it (with the state it was applied to) to the
// networkStore, which writes it to the room's log. applyLocal() is the
// receiver path: silent apply, no broadcast. Both run the same reducer so
// every device stays in step. restoreState() rolls back (online, when
// another device's move took the log slot this device's move was headed
// for). States are immutable, so a snapshot is just the state reference.

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

interface GameStore {
  state: TractorState | null;
  /** The human seat this device plays (online), or the hot-seat table's first human. */
  localPlayerId: PlayerId | null;
  /** The options + lineup the match started from, for "Play again". */
  setup: MatchSetup | null;
  /** False while this device is loading AI policies for a match it didn't start (an online stand-in). */
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

export const useGameStore = create<GameStore>((set, get) => ({
  state: null,
  localPlayerId: null,
  setup: null,
  aiReady: true,
  _broadcast: null,
  _blocked: null,

  loadGame: (state, localPlayerId, setup) => set({ state, localPlayerId, setup, aiReady: true }),

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
    set({ state: next });
    _broadcast?.(action, state, opts?.ai === true);
    return null;
  },

  applyLocal: (action) => {
    const { state } = get();
    if (!state) return false;
    try {
      set({ state: applyAction(state, action) });
      return true;
    } catch (e) {
      console.warn('[game] move rejected:', (e as Error).message);
      return false;
    }
  },

  restoreState: (state) => set({ state }),
  setAiReady: (ready) => set({ aiReady: ready }),
  clear: () => set({ state: null, localPlayerId: null, setup: null, aiReady: true }),

  registerBroadcastHandler: (fn) => set({ _broadcast: fn }),
  registerDispatchGuard: (fn) => set({ _blocked: fn }),
}));
