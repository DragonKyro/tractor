// What a round is worth to each seat, at its end. Zero-sum between the sides:
//
//   levels  (my side's levels gained − the other side's) / 3
//   seat    ± holding / taking the landlord seat (the defenders keep it by
//           winning, the attackers take it by reaching the threshold)
//   points  ± (attackers' points − threshold) / threshold, clipped to ±1 —
//           a dense-ish signal of how close the round was
//   win     ± passing the final rank (the match is won)
//
// Weights are an env option; the defaults keep levels the main objective.

import { scoringParams } from '@/game/engine';
import { materialize } from '@/game/scoring';
import type { TractorState } from '@/game/types';

export interface RewardWeights {
  levels: number;
  seat: number;
  points: number;
  win: number;
}

export const DEFAULT_REWARD: RewardWeights = { levels: 1, seat: 0.25, points: 0.25, win: 1 };

export function roundRewards(state: TractorState, w: RewardWeights = DEFAULT_REWARD): number[] {
  const r = state.round.result;
  if (!r) return state.seats.map(() => 0);
  const m = materialize(scoringParams(state.config), state.rules.numDecks, state.rules.numDecks * 100);
  const thr = m.landlordWins[m.landlordWins.length - 1]!.end;
  const pts = Math.max(-1, Math.min(1, (r.attackerPoints - thr) / thr));
  return state.seats.map((_, seat) => {
    const defending = r.landlordTeam.includes(seat);
    const sign = defending ? 1 : -1;
    const levels = r.landlordWon ? r.landlordDelta : -r.attackerDelta;
    const seatTerm = r.landlordWon ? 1 : -1;
    const winners = r.winners.length === 0 ? 0 : r.winners.some((x) => r.landlordTeam.includes(x)) ? 1 : -1;
    return sign * (w.levels * (levels / 3) + w.seat * seatTerm + w.points * -pts + w.win * winners);
  });
}

/** Levels my side gained minus the other side's (the evaluation metric). */
export function sideGain(state: TractorState, seat: number): number {
  const r = state.round.result;
  if (!r) return 0;
  const levels = r.landlordWon ? r.landlordDelta : -r.attackerDelta;
  return r.landlordTeam.includes(seat) ? levels : -levels;
}
