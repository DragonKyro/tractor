// Per-seat measures of the skills that separate strong Tractor players, read
// from a finished round (evaluate.py compares the network's seats with its
// opponents' on the same duplicate deals):
//
//   throws / throwFails   multi-unit leads tried, and how many a hidden hand
//                         beat (card counting before a throw)
//   trumpLeads            leads of trump (drawing trumps: tempo)
//   ruffWins              side-suit tricks won by trumping in (using voids,
//                         often made by the kitty)
//   pointsFed / pointsGiven   point cards played into tricks the seat's own side
//                         won (not by itself) / that the other side won
//   lastTrick             the seat's side won the final trick
//   kittyBuried           point value of the buried kitty (the landlord)
//   kittyConceded         kitty points × multiplier the attackers took on the
//                         final trick (charged to every defender)

import { totalPoints } from '@/game/cards';
import { tractorRequirements } from '@/game/engine';
import { formatFromCards } from '@/game/format';
import { effectiveSuit } from '@/game/trump';
import type { TractorState } from '@/game/types';

export const SKILLS = ['throws', 'throwFails', 'trumpLeads', 'ruffWins', 'pointsFed', 'pointsGiven', 'lastTrick', 'kittyBuried', 'kittyConceded'] as const;
export type Skill = (typeof SKILLS)[number];

export function skillStats(s: TractorState): Record<Skill, number[]> {
  const r = s.round;
  const n = s.seats.length;
  const out = Object.fromEntries(SKILLS.map((k) => [k, new Array<number>(n).fill(0)])) as Record<Skill, number[]>;
  const trump = r.trump;
  if (!trump || r.tricks.length === 0) return out;
  const defending = (seat: number) => r.landlordTeam.includes(seat);
  const sameSide = (a: number, b: number) => defending(a) === defending(b);
  const req = tractorRequirements(s.config);
  for (const t of r.tricks) {
    const lead = t.plays[0]!;
    if (lead.badThrowCards.length > 0) {
      out.throws[lead.seat]!++;
      out.throwFails[lead.seat]!++;
    } else {
      try {
        if (formatFromCards(trump, req, lead.cards).units.length > 1) out.throws[lead.seat]!++;
      } catch {
        /* not readable as a lead: not a throw */
      }
    }
    if (lead.cards.every((c) => effectiveSuit(trump, c) === 'T')) out.trumpLeads[lead.seat]!++;
    const win = t.plays.find((p) => p.seat === t.winner);
    if (t.suit !== 'T' && win && win.seat !== lead.seat && win.cards.every((c) => effectiveSuit(trump, c) === 'T')) out.ruffWins[win.seat]!++;
    for (const p of t.plays) {
      if (p.seat === t.winner) continue;
      const pts = totalPoints(p.cards);
      if (!pts) continue;
      if (sameSide(p.seat, t.winner)) out.pointsFed[p.seat]! += pts;
      else out.pointsGiven[p.seat]! += pts;
    }
  }
  const last = r.tricks[r.tricks.length - 1]!;
  for (let seat = 0; seat < n; seat++) out.lastTrick[seat] = sameSide(seat, last.winner) ? 1 : 0;
  if (r.landlord !== null) out.kittyBuried[r.landlord] = totalPoints(r.kitty);
  const kb = r.kittyBonus;
  if (kb && !defending(kb.seat)) for (let seat = 0; seat < n; seat++) if (defending(seat)) out.kittyConceded[seat] = kb.points * kb.multiplier;
  return out;
}
