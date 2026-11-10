// Tractor learned agent — turns "score these candidates" into one reducer
// action. The exchange is several decisions (each buried card, each friend
// call); the draft between them stays local to this loop, so the game state
// only changes once the whole exchange is chosen.

import { RANKS, SUITS } from '@/game/cards';
import type { TractorState, Action, FriendCall } from '@/game/types';
import { friendCallProblems, validBidsFor } from '@/game/engine';
import { observe } from '../observation';
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
    if ('forced' in step) return repairExchange(state, step.forced);
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

/**
 * Friend calls the rules allow, for when the chosen ones don't (the call
 * candidates already follow the friend-selection policy, so this is a safety
 * net): the first `want` single calls, highest rank first, that pass
 * friendCallProblems. Not a strategy, just legal.
 */
export function legalFriendCalls(state: TractorState, want: number): FriendCall[] {
  const out: FriendCall[] = [];
  const legal = (c: FriendCall) => friendCallProblems(state, [c]).every((p) => p.startsWith('Call exactly'));
  for (let ordinal = 1; ordinal <= state.rules.numDecks && out.length < want; ordinal++) {
    for (const rank of [...RANKS].reverse()) {
      for (const suit of SUITS) {
        const call = { card: rank + suit, ordinal } as FriendCall;
        if (out.length < want && legal(call)) out.push(call);
      }
    }
  }
  return out;
}

/** A finished exchange with its friend calls made legal, if they weren't. */
export function repairExchange(state: TractorState, a: Action): Action {
  if (a.type !== 'exchange' || friendCallProblems(state, a.friends).length === 0) return a;
  return { ...a, friends: legalFriendCalls(state, state.rules.numFriends) };
}

export function argmax(xs: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i]! > xs[best]!) best = i;
  return best;
}

export function netPolicy(net: TractorPolicyNet): CandidatePolicy {
  return (obs, d) => argmax(net.logits(obs, d.candidates.map((c) => c.features)));
}
