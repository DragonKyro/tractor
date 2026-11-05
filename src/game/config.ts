// Match options chosen in the lobby, their defaults and validation, and the
// numbers derived from them at the start of a match (decks, kitty, friends).
// Option names follow shengji's settings so its rules page reads across.

import type { BidPolicy, BidReinforcementPolicy, JokerBidPolicy } from './bidding';
import { DECK_SIZE, LEVELS, RANKS, type Level, type Rank } from './cards';
import type { ThrowEvaluationPolicy, TrickDrawPolicy } from './format';
import { DEFAULT_SCORING, type AdvancementPolicy } from './scoring';
import type { Seat } from '@/core/types';

export type GameMode = 'tractor' | 'findingFriends';
/**
 * live:  cards are dealt a round at a time and anyone may declare trump as
 *        their cards arrive (the real-table feel; online / vs AI).
 * turns: everything is dealt at once, then players bid in turn — the only
 *        workable way when several humans share one screen (hot-seat).
 * auto:  turns when two or more humans share this device, else live.
 */
export type BiddingStyle = 'auto' | 'live' | 'turns';
export type FirstLandlordSelection = 'winningBid' | 'firstBid';
export type KittyBidPolicy = 'firstCard' | 'firstCardOfLevelOrHighest';
export type ThrowPenalty = 'none' | 'tenPointsPerAttempt';
export type KittyPenalty = 'times' | 'power';
export type FriendSelectionPolicy = 'unrestricted' | 'trumpsIncluded' | 'highestCardNotAllowed' | 'pointCardNotAllowed';
export type MultipleJoinPolicy = 'unrestricted' | 'noDoubleJoin';

export interface TractorConfig {
  gameMode: GameMode;
  /** Finding friends: how many friends the landlord calls (null = players/2 − 1). */
  numFriends: number | null;
  /** null = players / 2 (at least 1). */
  numDecks: number | null;
  /** null = the classic size (cards left over, topped up to at least 5). Other sizes remove low cards. */
  kittySize: number | null;
  biddingStyle: BiddingStyle;
  startingRank: Rank;
  /** The last rank to play; passing it wins the game. */
  maxRank: Level;
  bidPolicy: BidPolicy;
  jokerBidPolicy: JokerBidPolicy;
  bidReinforcementPolicy: BidReinforcementPolicy;
  firstLandlordSelection: FirstLandlordSelection;
  kittyBidPolicy: KittyBidPolicy;
  trickDrawPolicy: TrickDrawPolicy;
  throwEvaluationPolicy: ThrowEvaluationPolicy;
  throwPenalty: ThrowPenalty;
  kittyPenalty: KittyPenalty;
  advancementPolicy: AdvancementPolicy;
  friendSelectionPolicy: FriendSelectionPolicy;
  multipleJoinPolicy: MultipleJoinPolicy;
  tractorMinCount: number;
  tractorMinLength: number;
  /** Points per scoring step, per deck (20 = the classic 80-of-200 with two decks). */
  stepSizePerDeck: number;
  bonusLevelForSmallerTeam: boolean;
  revealKittyAtEnd: boolean;
}

export const MIN_PLAYERS = 2;
export const MAX_PLAYERS = 12;
export const MAX_DECKS = 12;

export function defaultConfig(numPlayers = 4): TractorConfig {
  return {
    gameMode: numPlayers % 2 === 0 ? 'tractor' : 'findingFriends',
    numFriends: null,
    numDecks: null,
    kittySize: null,
    biddingStyle: 'auto',
    startingRank: '2',
    maxRank: 'A',
    bidPolicy: 'jokerOrGreaterLength',
    jokerBidPolicy: 'bothTwoOrMore',
    bidReinforcementPolicy: 'reinforceWhileWinning',
    firstLandlordSelection: 'winningBid',
    kittyBidPolicy: 'firstCard',
    trickDrawPolicy: 'noProtections',
    throwEvaluationPolicy: 'all',
    throwPenalty: 'none',
    kittyPenalty: 'times',
    advancementPolicy: 'unrestricted',
    friendSelectionPolicy: 'unrestricted',
    multipleJoinPolicy: 'unrestricted',
    tractorMinCount: 2,
    tractorMinLength: 2,
    stepSizePerDeck: DEFAULT_SCORING.stepSizePerDeck,
    bonusLevelForSmallerTeam: true,
    revealKittyAtEnd: true,
  };
}

/** Fill in any option a stored / remote config lacks (older builds). */
export function normalizeConfig(raw: Partial<TractorConfig> | null | undefined, numPlayers: number): TractorConfig {
  return { ...defaultConfig(numPlayers), ...(raw ?? {}) };
}

export function autoNumDecks(numPlayers: number): number {
  return Math.max(1, Math.floor(numPlayers / 2));
}

export function autoNumFriends(numPlayers: number): number {
  return Math.max(0, Math.floor(numPlayers / 2) - 1);
}

/** The classic kitty: whatever doesn't deal out evenly, topped up by one round of cards when under 5 (shengji's rule). */
export function autoKittySize(deckSize: number, numPlayers: number): number {
  let k = deckSize % numPlayers;
  if (k === 0) k = numPlayers;
  if (k < 5) k += numPlayers;
  return k;
}

/** The numbers a match actually plays with, fixed at its start. */
export interface ResolvedRules {
  numPlayers: number;
  numDecks: number;
  kittySize: number;
  /** Cards removed per round so the deal comes out even (explicit kitty sizes only). */
  cardsToRemove: number;
  gameMode: GameMode;
  numFriends: number;
  biddingStyle: 'live' | 'turns';
}

export function resolveRules(config: TractorConfig, seats: readonly Seat[]): ResolvedRules {
  const n = seats.length;
  const numDecks = config.numDecks ?? autoNumDecks(n);
  const deck = numDecks * DECK_SIZE;
  let kittySize: number;
  let cardsToRemove = 0;
  if (config.kittySize === null) {
    kittySize = autoKittySize(deck, n);
  } else {
    kittySize = config.kittySize;
    // Remove cards until (deck − removed − kitty) deals out evenly.
    cardsToRemove = (((deck - kittySize) % n) + n) % n;
  }
  const localHumans = seats.filter((s) => !s.isAI && s.isLocal).length;
  const biddingStyle = config.biddingStyle === 'auto' ? (localHumans >= 2 ? 'turns' : 'live') : config.biddingStyle;
  return {
    numPlayers: n,
    numDecks,
    kittySize,
    cardsToRemove,
    gameMode: config.gameMode,
    numFriends: config.gameMode === 'findingFriends' ? config.numFriends ?? autoNumFriends(n) : 0,
    biddingStyle,
  };
}

/** Human-readable problems with this config for these seats ([] = fine). */
export function validateConfig(config: TractorConfig, seats: readonly Seat[]): string[] {
  const errors: string[] = [];
  const n = seats.length;
  if (n < MIN_PLAYERS || n > MAX_PLAYERS) errors.push(`Tractor takes ${MIN_PLAYERS}–${MAX_PLAYERS} players (${n} seated).`);
  if (config.gameMode === 'tractor' && n % 2 !== 0) {
    errors.push('Classic tractor (fixed partners) needs an even number of players — use Finding Friends for odd tables.');
  }
  const decks = config.numDecks ?? autoNumDecks(n);
  if (!Number.isInteger(decks) || decks < 1 || decks > MAX_DECKS) errors.push(`Use 1–${MAX_DECKS} decks.`);
  if (decks > Math.max(1, n * 2)) errors.push('At most two decks per player.');
  if (config.gameMode === 'findingFriends' && config.numFriends !== null) {
    if (!Number.isInteger(config.numFriends) || config.numFriends < 0) errors.push('Friends must be 0 or more.');
    else if (config.numFriends + 1 >= n) errors.push(`With ${n} players the landlord can call at most ${Math.max(0, n - 2)} friend${n - 2 === 1 ? '' : 's'}.`);
    else if (config.numFriends > 0 && decks < 1) errors.push('Need at least one deck.');
  }
  if (config.kittySize !== null) {
    const deck = decks * DECK_SIZE;
    if (!Number.isInteger(config.kittySize) || config.kittySize < 0) errors.push('Kitty size must be 0 or more.');
    else if (config.kittySize >= deck - n) errors.push('The kitty is too large for this deck.');
    else {
      const remove = (((deck - config.kittySize) % n) + n) % n;
      // Only one card per suit per deck of the lowest removable rank can go.
      if (remove > decks * 4) errors.push('That kitty size would mean removing too many cards.');
    }
  }
  if (!RANKS.includes(config.startingRank)) errors.push('Pick a starting rank.');
  if (!LEVELS.includes(config.maxRank)) errors.push('Pick a final rank.');
  else if (LEVELS.indexOf(config.maxRank) < LEVELS.indexOf(config.startingRank)) errors.push('The final rank must not be below the starting rank.');
  if (config.tractorMinCount < 2 || config.tractorMinLength < 2) errors.push('Tractors need at least two tuples of at least pairs.');
  const step = config.stepSizePerDeck * decks;
  if (step <= 0 || step % 5 !== 0 || step > decks * 100) errors.push('Scoring step must be a multiple of 5, at most 100 per deck.');
  const live = config.biddingStyle === 'live';
  if (live && seats.filter((s) => !s.isAI && s.isLocal).length >= 2) {
    errors.push('Live bidding needs every human on their own device — choose "Take turns" for hot-seat.');
  }
  return errors;
}
