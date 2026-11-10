// The Tractor match state and its actions. Everything here is plain JSON:
// online, every device rebuilds the state from the start record and replays
// the room's action log through `applyAction`.

import type { PlayerId, Seat } from '@/core/types';
import type { RngState } from '@/core/rng';
import type { Bid } from './bidding';
import type { Card, Level } from './cards';
import type { ResolvedRules, TractorConfig } from './config';
import type { EffSuit, Trump } from './trump';
import type { PlayedCards, TrickState } from './trick';
import type { Advancement } from './scoring';

/**
 * A round's stages:
 *   draw      — live bidding: cards are dealt a round at a time, anyone may bid.
 *               Once the deck is out the landlord-to-be picks up the kitty.
 *   bid       — turn bidding: everything's dealt; players bid or pass in turn.
 *   exchange  — the landlord has the kitty, buries as many cards (and in
 *               Finding Friends calls friends).
 *   play      — tricks.
 *   roundOver — scored; waiting for "next round".
 */
export type Stage = 'draw' | 'bid' | 'exchange' | 'play' | 'roundOver';

/** A landlord's friend call: "whoever plays the `ordinal`-th copy of `card` joins me". */
export interface FriendCall {
  card: Card;
  /** 1 = the first copy played, 2 = the second, … (at most the number of decks). */
  ordinal: number;
}

export interface Friend extends FriendCall {
  /** Copies still to be played before the one that counts. */
  skip: number;
  /** Who turned out to be the friend (null until revealed). */
  seat: number | null;
}

export interface CompletedTrick {
  leader: number;
  suit: EffSuit;
  plays: PlayedCards[];
  winner: number;
  points: number;
}

export interface RoundResult {
  round: number;
  landlord: number;
  landlordTeam: number[];
  trump: Trump;
  /** Points the attackers finished with (kitty and penalties included). */
  attackerPoints: number;
  /** Points in the buried kitty, the multiplier, and whether the attackers got them. */
  kittyPoints: number;
  kittyMultiplier: number;
  kittyToAttackers: boolean;
  penaltyAdjustment: number;
  landlordWon: boolean;
  landlordDelta: number;
  attackerDelta: number;
  bonusLevel: boolean;
  /** Per seat. */
  advancements: Advancement[];
  nextLandlord: number;
  kitty: Card[];
  /** Seats that passed the final rank this round (the game is over when non-empty). */
  winners: number[];
  /** Ended before the last trick because nothing left could change the result. */
  endedEarly?: boolean;
}

export interface RoundState {
  /** 1-based round number (a redeal keeps the number). */
  number: number;
  stage: Stage;
  /** Seat that receives the first card and bids first in turn bidding. */
  dealer: number;
  /** Cards still to deal (dealt from the end). */
  deck: Card[];
  dealTo: number;
  hands: Card[][];
  /** The kitty: face down during the deal, the landlord's buried cards afterwards. */
  kitty: Card[];
  /** The kitty as dealt (the landlord sees which cards came from it). */
  dealtKitty: Card[];
  kittySize: number;
  /** Cards taken out so the deal comes out even (explicit kitty sizes). */
  removed: Card[];
  bids: Bid[];
  /** No one bid: the trump came from turning over kitty cards. */
  autobid: Bid | null;
  /** Kitty cards turned over so far (no-bid flow). */
  revealed: number;
  /** Turn bidding: whose turn it is (null once bidding is over). */
  bidTurn: number | null;
  /** Turn bidding: consecutive passes. */
  passes: number;
  /** Live bidding, after the deal: seats that said they have no more bids. */
  bidDone: number[];
  landlord: number | null;
  trump: Trump | null;
  friends: Friend[];
  /** The landlord and the partners known so far (all of them in classic tractor). */
  landlordTeam: number[];
  trick: TrickState | null;
  tricks: CompletedTrick[];
  /** Point cards each seat has won in tricks. */
  captured: Card[][];
  /** Points from the kitty, added to the last trick's winner. */
  kittyBonus: { seat: number; points: number; multiplier: number } | null;
  /** Throw penalties per seat. */
  penalties: number[];
  result: RoundResult | null;
}

export type LogKind = 'deal' | 'bid' | 'play' | 'throw' | 'trick' | 'friend' | 'round' | 'info';

export interface LogEntry {
  /** Running index, so lists can key on it. */
  n: number;
  kind: LogKind;
  text: string;
  seat?: number;
  /** The line without card / point details (shown instead while the round's hide options apply). */
  bare?: string;
  /** Round the entry belongs to (set with `bare`). */
  round?: number;
}

export interface TractorState {
  /** Shell-level status: the match runs until someone passes the final rank. */
  status: 'playing' | 'gameOver';
  seats: Seat[];
  config: TractorConfig;
  rules: ResolvedRules;
  rngState: RngState;
  /** Each seat's level. */
  levels: Level[];
  /** The landlord for this round (null in the first round until the bidding decides). */
  landlord: number | null;
  round: RoundState;
  history: RoundResult[];
  log: LogEntry[];
  logCount: number;
  /** Id of the seat whose decision is awaited, or null (live deal: anyone may bid; between rounds). */
  activePlayerId: PlayerId | null;
  /** Seats that won the match. */
  winners: number[] | null;
  /** Final level index per player id once the match is over (shell convention). */
  finalScores: Record<PlayerId, number> | null;
}

export type Action =
  /** Live bidding: `seat` draws the next card (it must be their turn: round.dealTo). */
  | { type: 'draw'; seat: number }
  /** After a live deal: no more bids from `seat`. */
  | { type: 'doneBidding'; seat: number }
  | { type: 'bid'; seat: number; card: Card; count: number }
  /** Turn bidding: decline to bid this turn. */
  | { type: 'pass'; seat: number }
  /** No bids and a landlord: turn over the next kitty card to set trump. */
  | { type: 'revealKitty'; seat: number }
  /** No bids in the first round: shuffle and deal again. */
  | { type: 'redeal' }
  /** Live bidding over: the landlord-to-be takes the kitty. */
  | { type: 'pickUpKitty'; seat: number }
  | { type: 'exchange'; seat: number; bury: Card[]; friends: FriendCall[] }
  /** `format` picks a reading of an ambiguous lead (an index into possibleFormats). */
  | { type: 'play'; seat: number; cards: Card[]; format?: number }
  /** Stop the round once nothing left can change its result (canEndRoundEarly); `seat` is who asked. */
  | { type: 'endRoundEarly'; seat: number }
  | { type: 'nextRound' };

export type ActionType = Action['type'];
