// FROZEN COPY of src/ai/nn (encoder v2, action space v1, tractor/cand-mlp@1) for benchmarking
// v2 policies (training/benchmarks/) against newer networks. Do not edit: a v2
// policy only plays as trained if this code stays exactly as it was. (The one
// change from the original: friend calls the rules reject are no longer swapped
// for the deleted heuristic's; the benchmark is only used at 4-player Tractor,
// where there are no friend calls, and in tests.)

// Tractor learned agent — turns "score these candidates" into one reducer
// action. The exchange is several decisions (each buried card, each friend
// call); the draft between them stays local to this loop, so the game state
// only changes once the whole exchange is chosen.

import type { TractorState, Action } from '@/game/types';
import { validBidsFor } from '@/game/engine';
import { observe } from '@/ai/observation';
import { applyDraft, emptyDraft, nextStep, type Decision } from './decision';
import { encodeObservation } from './encode';
import type { TractorPolicyNet } from './runtime';

/** Pick a candidate index for a decision, given the encoded observation. */
export type CandidatePolicy = (obs: Float32Array, decision: Decision) => number;

/**
 * Live deal: training asks a seat about bidding only when its legal bids
 * change (a card arrives, someone bids; training/tractor/vec-env.ts), while
 * the app's driver polls every AI seat at every card. With a memo, a seat
 * that declined isn't asked again until its options change. Only declines
 * are remembered: a bid that gets lost (an online race) is simply made again.
 */
export type BidMemo = Map<string, string>;

function bidAsk(state: TractorState, seat: number): string {
  const bids = validBidsFor(state, seat).map((b) => `${b.card}${b.count}`).join(',');
  return `${state.round.number}|${state.rngState.seed}|${bids}`;
}

/** `seat`'s next action under `policy`, or null when it has nothing to do (or chooses not to bid). */
export function actWithPolicy(state: TractorState, seat: number, policy: CandidatePolicy, memo?: BidMemo): Action | null {
  const draft = emptyDraft();
  const obs0 = observe(state, seat);
  let ask: string | null = null;
  for (let part = 0; part < 256; part++) {
    const step = nextStep(obs0, draft);
    if (!step) return null;
    if (memo && part === 0 && obs0.stage === 'draw' && 'decision' in step && step.decision.kind === 'bid') {
      ask = bidAsk(state, seat);
      if (memo.get(state.seats[seat]!.id) === ask) return null;
    }
    if ('forced' in step) return step.forced;
    const d = step.decision;
    const i = policy(encodeObservation(applyDraft(obs0, draft)), d);
    const pick = d.candidates[Math.max(0, Math.min(d.candidates.length - 1, i))]!.pick;
    if ('none' in pick) {
      if (memo && ask !== null) memo.set(state.seats[seat]!.id, ask);
      return null;
    }
    if ('action' in pick) return pick.action;
    if ('bury' in pick) draft.buried.push(pick.bury);
    else draft.calls.push(pick.call);
  }
  throw new Error('actWithPolicy: the exchange did not finish');
}

export function argmax(xs: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i]! > xs[best]!) best = i;
  return best;
}

export function netPolicy(net: TractorPolicyNet): CandidatePolicy {
  return (obs, d) => argmax(net.logits(obs, d.candidates.map((c) => c.features)));
}
