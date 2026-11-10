// Short UI strings for game things.

import { SUIT_SYMBOL, levelLabel, type Level, type Suit } from '@/game/cards';
import type { Trump } from '@/game/trump';
import type { LogEntry, TractorState } from '@/game/types';

/** The trump as display parts; `suitClass` colours the symbol (empty for no-trump). */
export function trumpParts(trump: Trump | null): { symbol: string; number: string; suit: Suit | null; suitClass: string; label: string } {
  if (!trump) return { symbol: '?', number: '', suit: null, suitClass: '', label: 'Trump not set yet' };
  const number = trump.number === null ? '' : levelLabel(trump.number);
  if (trump.kind === 'noTrump') {
    return { symbol: 'NT', number: number ? ` · ${number}s` : '', suit: null, suitClass: '', label: trump.number === null ? 'No trump (jokers only)' : `No trump suit · ${number}s are trump` };
  }
  return {
    symbol: SUIT_SYMBOL[trump.suit],
    number,
    suit: trump.suit,
    suitClass: `suit suit-${trump.suit}`,
    label: `${SUIT_SYMBOL[trump.suit]} is trump · ${number}s are trump`,
  };
}

export function levelText(level: Level): string {
  return levelLabel(level);
}

/** "Hide landlord points": this seat's captured points stay hidden until the round ends. */
export function pointsHidden(state: TractorState, seat: number): boolean {
  const r = state.round;
  return state.config.hideLandlordPoints && r.stage !== 'roundOver' && r.landlordTeam.includes(seat);
}

/** A log line as this table may see it (cards / points left out while the hide options apply). */
export function logLine(state: TractorState, e: LogEntry): string {
  const r = state.round;
  if (e.bare === undefined || e.round !== r.number || r.stage === 'roundOver') return e.text;
  if (state.config.hidePlayedCards && (e.kind === 'play' || e.kind === 'throw')) return e.bare;
  if (e.kind === 'trick' && e.seat !== undefined && pointsHidden(state, e.seat)) return e.bare;
  return e.text;
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
