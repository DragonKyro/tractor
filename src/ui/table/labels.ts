// Short UI strings for game things.

import { SUIT_SYMBOL, levelLabel, type Level } from '@/game/cards';
import type { Trump } from '@/game/trump';
import type { TractorState } from '@/game/types';

export function trumpParts(trump: Trump | null): { symbol: string; number: string; red: boolean; label: string } {
  if (!trump) return { symbol: '?', number: '', red: false, label: 'Trump not set yet' };
  const number = trump.number === null ? '' : levelLabel(trump.number);
  if (trump.kind === 'noTrump') return { symbol: 'NT', number: number ? ` · ${number}s` : '', red: false, label: trump.number === null ? 'No trump (jokers only)' : `No trump suit · ${number}s are trump` };
  return {
    symbol: SUIT_SYMBOL[trump.suit],
    number,
    red: trump.suit === 'H' || trump.suit === 'D',
    label: `${SUIT_SYMBOL[trump.suit]} is trump · ${number}s are trump`,
  };
}

export function levelText(level: Level): string {
  return levelLabel(level);
}

/** "Landlord", "Friend", "Partner" etc. for a seat, as far as the table knows. */
export function roleOf(state: TractorState, seat: number): 'landlord' | 'defender' | 'attacker' | 'unknown' {
  const r = state.round;
  if (r.landlord === null) return 'unknown';
  if (r.landlord === seat) return 'landlord';
  if (r.landlordTeam.includes(seat)) return 'defender';
  if (r.stage === 'draw' || r.stage === 'bid' || r.stage === 'exchange') {
    return state.rules.gameMode === 'tractor' ? (seat % 2 === r.landlord % 2 ? 'defender' : 'attacker') : 'unknown';
  }
  if (state.rules.gameMode === 'findingFriends' && r.friends.some((f) => f.seat === null)) return 'unknown';
  return 'attacker';
}
