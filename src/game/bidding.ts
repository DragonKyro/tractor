// Declaring trump — a port of shengji's mechanics/src/bidding.rs.
//
// A bid shows `count` copies of one card: a card of the bidding level (its
// suit becomes trump) or a joker (no trump suit). A later bid must show more
// cards, or — at equal length — outrank it (big joker > small joker > a suit,
// and under the "higher suit" policy ♠ > ♥ > ♣ > ♦ for 2+ cards).

import { SUIT_BID_ORDER, BIG_JOKER, SMALL_JOKER, cardRank, cardSuit, countCards, isJoker, type Card, type Level } from './cards';

export type BidPolicy = 'jokerOrHigherSuit' | 'jokerOrGreaterLength' | 'greaterLength';
export type JokerBidPolicy = 'bothTwoOrMore' | 'bothNumDecks' | 'ljNumDecksHjNumDecksLessOne' | 'disabled';
export type BidReinforcementPolicy =
  /** You may only reinforce (add copies to) your bid while it's winning. */
  | 'reinforceWhileWinning'
  /** …or overturn it with a greater bid of your own. */
  | 'overturnOrReinforceWhileWinning'
  /** …or reinforce it to tie the current winning bid. */
  | 'reinforceWhileEquivalent';

export interface Bid {
  seat: number;
  card: Card;
  count: number;
  /** 0 during the deal; bumps each time someone over-bids during the exchange (kitty theft). */
  epoch: number;
}

export interface BidRules {
  bidPolicy: BidPolicy;
  jokerBidPolicy: JokerBidPolicy;
  bidReinforcementPolicy: BidReinforcementPolicy;
  numDecks: number;
}

function sameBid(a: Bid, b: Bid): boolean {
  return a.seat === b.seat && a.card === b.card && a.count === b.count && a.epoch === b.epoch;
}

/**
 * Every bid `seat` may make now. `bidLevel` is the level bids are made at:
 * the landlord's when there is one, else the bidder's own.
 */
export function validBids(
  seat: number,
  bids: readonly Bid[],
  hand: readonly Card[],
  bidLevel: Level,
  landlordKnown: boolean,
  epoch: number,
  rules: BidRules,
): Bid[] {
  // A landlord playing no-trump has nothing to bid for.
  if (landlordKnown && bidLevel === 'NT') return [];
  const counts = countCards(hand);
  const mostRecent = [...bids].reverse().find((b) => b.seat === seat);
  const last = bids[bids.length - 1];

  const candidates: Array<[Card, number]> = [];
  for (const [card, count] of counts) {
    if (isJoker(card) || (bidLevel !== 'NT' && cardRank(card) === bidLevel)) candidates.push([card, count]);
  }

  let valid: Bid[] | null = null;
  if (candidates.length > 0) {
    valid = [];
    const isNT = bidLevel === 'NT';
    const n = rules.numDecks;
    for (const [card, count] of candidates) {
      for (let k = 1; k <= count; k++) {
        if (isJoker(card)) {
          const p = rules.jokerBidPolicy;
          if (p === 'disabled') continue;
          if (p === 'bothTwoOrMore' && k <= 1 && !isNT) continue;
          if (card === SMALL_JOKER && (p === 'ljNumDecksHjNumDecksLessOne' || p === 'bothNumDecks') && k < n) continue;
          if (card === BIG_JOKER && p === 'ljNumDecksHjNumDecksLessOne' && k < n - 1) continue;
          if (card === BIG_JOKER && p === 'bothNumDecks' && k < n) continue;
        }
        const bid: Bid = { seat, card, count: k, epoch };
        if (!last) {
          valid.push(bid);
        } else if (bid.count > last.count) {
          valid.push(bid);
        } else if (bid.count === last.count && rules.bidPolicy !== 'greaterLength') {
          if (outranksAtEqualLength(bid, last, rules.bidPolicy)) valid.push(bid);
        }
      }
    }
  }

  const policy = rules.bidReinforcementPolicy;
  const winning = mostRecent && last && sameBid(mostRecent, last);
  if (mostRecent && winning && (policy === 'reinforceWhileWinning' || policy === 'reinforceWhileEquivalent')) {
    // The current winner may only add copies to their own bid.
    const available = counts.get(mostRecent.card) ?? 0;
    const out: Bid[] = [];
    for (let k = mostRecent.count + 1; k <= available; k++) out.push({ seat, card: mostRecent.card, count: k, epoch });
    return out;
  }
  if (mostRecent && policy === 'reinforceWhileEquivalent' && valid) {
    const available = counts.get(mostRecent.card) ?? 0;
    if (last && last.count <= available) {
      const bid: Bid = { seat, card: mostRecent.card, count: last.count, epoch };
      if (sameBid(bid, last) || !isJoker(last.card)) valid.push(bid);
    }
    return valid;
  }
  return valid ?? [];
}

function outranksAtEqualLength(bid: Bid, last: Bid, policy: BidPolicy): boolean {
  const a = bid.card;
  const b = last.card;
  if (a === BIG_JOKER) return b !== BIG_JOKER;
  if (a === SMALL_JOKER) return b !== BIG_JOKER && b !== SMALL_JOKER;
  if (isJoker(b)) return false;
  return policy === 'jokerOrHigherSuit' && bid.count > 1 && SUIT_BID_ORDER[cardSuit(a)!] > SUIT_BID_ORDER[cardSuit(b)!];
}

export function isValidBid(bid: Bid, valid: readonly Bid[]): boolean {
  return valid.some((v) => sameBid(v, bid));
}

/** The first bid and the winning bid (an automatic kitty bid wins outright). */
export function firstAndWinner(bids: readonly Bid[], autobid: Bid | null): { first: Bid; winner: Bid } | null {
  const winner = autobid ?? bids[bids.length - 1];
  const first = bids[0] ?? autobid;
  if (!winner || !first) return null;
  return { first, winner };
}
