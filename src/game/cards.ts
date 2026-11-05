// Cards, suits, ranks and levels. Pure data + helpers, no trump awareness
// (that lives in trump.ts).
//
// A card is a short string so states and the online action log stay small
// and readable: rank + suit for suited cards ('2S', 'TH' for the ten of
// hearts, 'AD', 'KC'), 'LJ' for the small (little) joker and 'BJ' for the big
// joker. With several decks the same string simply appears several times.

export type Suit = 'S' | 'H' | 'D' | 'C';
export type Rank = '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | 'T' | 'J' | 'Q' | 'K' | 'A';
/** A player's level: a rank, or no-trump (only reachable when the max rank is NT). */
export type Level = Rank | 'NT';
export type Card = string;

export const SMALL_JOKER: Card = 'LJ';
export const BIG_JOKER: Card = 'BJ';

/** Suit order used for tie-breaks and successor lists (shengji's ALL_SUITS). */
export const SUITS: readonly Suit[] = ['S', 'H', 'D', 'C'];
export const RANKS: readonly Rank[] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
export const LEVELS: readonly Level[] = [...RANKS, 'NT'];

/** Bid strength of suits under the "higher suit" bid policy: ♦ < ♣ < ♥ < ♠. */
export const SUIT_BID_ORDER: Record<Suit, number> = { D: 0, C: 1, H: 2, S: 3 };

export const SUIT_SYMBOL: Record<Suit, string> = { S: '♠', H: '♥', D: '♦', C: '♣' };
export const SUIT_NAME: Record<Suit, string> = { S: 'spades', H: 'hearts', D: 'diamonds', C: 'clubs' };
export const RED_SUITS: ReadonlySet<Suit> = new Set<Suit>(['H', 'D']);

export function isJoker(card: Card): boolean {
  return card === SMALL_JOKER || card === BIG_JOKER;
}

export function cardRank(card: Card): Rank | null {
  return isJoker(card) ? null : (card[0] as Rank);
}

export function cardSuit(card: Card): Suit | null {
  return isJoker(card) ? null : (card[1] as Suit);
}

export function makeCard(rank: Rank, suit: Suit): Card {
  return rank + suit;
}

/** 2..14 (ace high). */
export function rankValue(rank: Rank): number {
  return RANKS.indexOf(rank) + 2;
}

export function rankSuccessor(rank: Rank): Rank | null {
  const i = RANKS.indexOf(rank);
  return i >= 0 && i < RANKS.length - 1 ? RANKS[i + 1]! : null;
}

export function levelSuccessor(level: Level): Level | null {
  const i = LEVELS.indexOf(level);
  return i >= 0 && i < LEVELS.length - 1 ? LEVELS[i + 1]! : null;
}

export function levelIndex(level: Level): number {
  return LEVELS.indexOf(level);
}

/** Points a card is worth: 5s are 5, tens and kings 10. */
export function cardPoints(card: Card): number {
  const r = cardRank(card);
  if (r === '5') return 5;
  if (r === 'T' || r === 'K') return 10;
  return 0;
}

export function totalPoints(cards: readonly Card[]): number {
  let n = 0;
  for (const c of cards) n += cardPoints(c);
  return n;
}

export function isValidCard(card: unknown): card is Card {
  if (typeof card !== 'string') return false;
  if (isJoker(card)) return true;
  return card.length === 2 && RANKS.includes(card[0] as Rank) && SUITS.includes(card[1] as Suit);
}

/** One standard 54-card deck, in a fixed order (♦, ♣, ♥, ♠ from ace down, then the jokers). */
export function standardDeck(): Card[] {
  const out: Card[] = [];
  for (const suit of ['D', 'C', 'H', 'S'] as Suit[]) {
    for (let i = RANKS.length - 1; i >= 0; i--) out.push(RANKS[i]! + suit);
  }
  out.push(SMALL_JOKER, BIG_JOKER);
  return out;
}

export const DECK_SIZE = 54;
/** Point total of one deck: 4 × (5 + 10 + 10). */
export const POINTS_PER_DECK = 100;

export function buildDecks(numDecks: number): Card[] {
  const out: Card[] = [];
  for (let i = 0; i < numDecks; i++) out.push(...standardDeck());
  return out;
}

/** Multiset of cards → counts. */
export function countCards(cards: Iterable<Card>): Map<Card, number> {
  const m = new Map<Card, number>();
  for (const c of cards) m.set(c, (m.get(c) ?? 0) + 1);
  return m;
}

/** True when every card of `sub` (with multiplicity) is in `pool`. */
export function containsAll(pool: readonly Card[], sub: readonly Card[]): boolean {
  const counts = countCards(pool);
  for (const c of sub) {
    const n = counts.get(c) ?? 0;
    if (n <= 0) return false;
    counts.set(c, n - 1);
  }
  return true;
}

/** `pool` with one copy of each card of `sub` taken out. Throws when one is missing. */
export function removeCards(pool: readonly Card[], sub: readonly Card[]): Card[] {
  const out = pool.slice();
  for (const c of sub) {
    const i = out.indexOf(c);
    if (i < 0) throw new Error(`card ${c} is not there`);
    out.splice(i, 1);
  }
  return out;
}

export function rankLabel(rank: Rank): string {
  return rank === 'T' ? '10' : rank;
}

export function levelLabel(level: Level): string {
  return level === 'NT' ? 'NT' : rankLabel(level);
}

/** Short human label: "10♥", "A♠", "small joker", "big joker". */
export function cardLabel(card: Card): string {
  if (card === SMALL_JOKER) return 'small joker';
  if (card === BIG_JOKER) return 'big joker';
  return rankLabel(cardRank(card)!) + SUIT_SYMBOL[cardSuit(card)!];
}

/** Compact label for inline lists: jokers shown as "LJ"/"BJ". */
export function cardShort(card: Card): string {
  if (card === SMALL_JOKER) return 'LJ';
  if (card === BIG_JOKER) return 'BJ';
  return rankLabel(cardRank(card)!) + SUIT_SYMBOL[cardSuit(card)!];
}
