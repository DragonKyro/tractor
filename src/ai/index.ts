// The AI players. Every decision goes through an Observation (what the seat
// may see), then the trained network (src/ai/nn, loaded by nn/network.ts).

import type { Seat } from '@/core/types';
import type { Action, TractorState } from '@/game/types';
import { actWithPolicy, netPolicy, type BidMemo } from './nn/agent';
import { loadNetwork, loadedNetwork } from './nn/network';

/** Live-deal declines of the AI seats on this device (see BidMemo). */
const bidMemo: BidMemo = new Map();

/**
 * The AI's next action for `seat`, or null when it has nothing to do (or
 * chooses to wait). The network must be loaded (prepareAi, or adviseAction);
 * until then the seat just waits.
 */
export function chooseAIAction(state: TractorState, seat: number): Action | null {
  const net = loadedNetwork();
  return net ? actWithPolicy(state, seat, netPolicy(net), bidMemo) : null;
}

/** `seat`'s best action, loading the network first if need be (hints, taking over for a player who dropped). */
export async function adviseAction(state: TractorState, seat: number): Promise<Action | null> {
  await loadNetwork();
  return chooseAIAction(state, seat);
}

/** Load the network before a match with AI seats. The online host and the lobby await this. */
export async function prepareAi(seats: readonly Seat[]): Promise<void> {
  if (seats.some((s) => s.isAI)) await loadNetwork();
}

export { observe } from './observation';
export type { Observation } from './observation';
