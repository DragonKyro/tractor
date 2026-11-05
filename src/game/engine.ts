// The Tractor reducer: createInitialState + applyAction, pure and
// deterministic (shuffles draw from state.rngState). Round flow follows
// shengji's game_state phases: initialize → draw (bidding) → exchange →
// play → next round. Illegal actions throw IllegalAction and leave the
// state untouched.

import { createRng, rngInt, shuffle } from '@/core/rng';
import { firstAndWinner, isValidBid, validBids, type Bid } from './bidding';
import {
  SUITS,
  buildDecks,
  cardLabel,
  cardPoints,
  cardRank,
  cardShort,
  cardSuit,
  containsAll,
  isJoker,
  levelIndex,
  levelLabel,
  removeCards,
  totalPoints,
  type Card,
  type Level,
  type Rank,
} from './cards';
import { normalizeConfig, resolveRules, validateConfig, type TractorConfig } from './config';
import { describeShapes, requiredShapes, TrickError, type TractorRequirements } from './format';
import { advancePlayer, materialize, scoreRound, type ScoringParams, DEFAULT_SCORING } from './scoring';
import { completeTrick, newTrick, nextToPlay, playCards, playProblem, trickComplete, type TrickRules } from './trick';
import { compareCards, trumpLabel, type Trump } from './trump';
import type { Seat } from '@/core/types';
import type { Action, Friend, FriendCall, LogEntry, LogKind, RoundResult, RoundState, TractorState } from './types';

export class IllegalAction extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'IllegalAction';
  }
}

function fail(message: string): never {
  throw new IllegalAction(message);
}

const LOG_LIMIT = 250;

function clone<T>(v: T): T {
  return typeof structuredClone === 'function' ? structuredClone(v) : (JSON.parse(JSON.stringify(v)) as T);
}

export function seatName(state: TractorState, seat: number): string {
  return state.seats[seat]?.name ?? `Seat ${seat + 1}`;
}

/** `bare`: the same line without its card / point details, for the hide-played-cards and hide-landlord-points options. */
function log(s: TractorState, kind: LogKind, text: string, seat?: number, bare?: string): void {
  const entry: LogEntry = { n: s.logCount, kind, text };
  if (seat !== undefined) entry.seat = seat;
  if (bare !== undefined) {
    entry.bare = bare;
    entry.round = s.round.number;
  }
  s.log.push(entry);
  s.logCount++;
  if (s.log.length > LOG_LIMIT) s.log.splice(0, s.log.length - LOG_LIMIT);
}

export function trickRules(state: TractorState): TrickRules {
  return {
    trickDrawPolicy: state.config.trickDrawPolicy,
    throwEvaluationPolicy: state.config.throwEvaluationPolicy,
    tractorRequirements: tractorRequirements(state.config),
  };
}

export function tractorRequirements(config: TractorConfig): TractorRequirements {
  return { minCount: config.tractorMinCount, minLength: config.tractorMinLength };
}

export function scoringParams(config: TractorConfig): ScoringParams {
  return { ...DEFAULT_SCORING, stepSizePerDeck: config.stepSizePerDeck, bonusLevelForSmallerTeam: config.bonusLevelForSmallerTeam };
}

// ------------------------------------------------------------------ setup

export function createInitialState(config: TractorConfig, seed: number, seats: Seat[]): TractorState {
  const cfg = normalizeConfig(config, seats.length);
  const errors = validateConfig(cfg, seats);
  if (errors.length) throw new Error(errors.join(' '));
  const rules = resolveRules(cfg, seats);
  const rngState = createRng(seed);
  const dealer = rngInt(rngState, seats.length);
  const s: TractorState = {
    status: 'playing',
    seats: seats.map((x) => ({ ...x })),
    config: cfg,
    rules,
    rngState,
    levels: seats.map(() => cfg.startingRank as Level),
    landlord: null,
    round: emptyRound(1, dealer, seats.length),
    history: [],
    log: [],
    logCount: 0,
    activePlayerId: null,
    winners: null,
    finalScores: null,
  };
  const mode = rules.gameMode === 'tractor' ? 'Tractor' : `Finding Friends (${rules.numFriends} friend${rules.numFriends === 1 ? '' : 's'})`;
  log(s, 'info', `${mode} · ${seats.length} players · ${rules.numDecks} deck${rules.numDecks === 1 ? '' : 's'} · kitty of ${rules.kittySize}`);
  beginRound(s, 1, dealer);
  return finalize(s);
}

function emptyRound(number: number, dealer: number, n: number): RoundState {
  return {
    number,
    stage: 'draw',
    dealer,
    deck: [],
    dealTo: dealer,
    hands: Array.from({ length: n }, () => []),
    kitty: [],
    dealtKitty: [],
    kittySize: 0,
    removed: [],
    bids: [],
    autobid: null,
    revealed: 0,
    bidTurn: null,
    passes: 0,
    landlord: null,
    trump: null,
    friends: [],
    landlordTeam: [],
    trick: null,
    tricks: [],
    captured: Array.from({ length: n }, () => []),
    kittyBonus: null,
    penalties: Array.from({ length: n }, () => 0),
    result: null,
  };
}

/** Which rank to take out when an explicit kitty size needs cards removed (shengji's choice). */
function rankToRemove(s: TractorState): Rank {
  const min: Rank = '2';
  if (s.landlord !== null) {
    const level = s.levels[s.landlord]!;
    return level === min ? '3' : min;
  }
  const bad = new Set<string>(s.levels.filter((l) => l !== 'NT'));
  bad.add('5');
  bad.add('T');
  bad.add('K');
  for (const r of ['2', '3', '4', '6', '7', '8', '9', 'J', 'Q', 'A'] as Rank[]) if (!bad.has(r)) return r;
  return min;
}

function beginRound(s: TractorState, number: number, dealer: number): void {
  const n = s.seats.length;
  const round = emptyRound(number, dealer, n);
  round.landlord = s.landlord;
  round.kittySize = s.rules.kittySize;
  let deck = buildDecks(s.rules.numDecks);
  if (s.rules.cardsToRemove > 0) {
    const rank = rankToRemove(s);
    // One per suit per deck, clubs first, as shengji does.
    const order = [...SUITS].reverse();
    for (let i = 0; i < s.rules.cardsToRemove; i++) {
      const card = rank + order[i % order.length]!;
      const at = deck.indexOf(card);
      if (at < 0) break;
      deck.splice(at, 1);
      round.removed.push(card);
    }
  }
  deck = shuffle(s.rngState, deck);
  round.kitty = deck.slice(deck.length - round.kittySize);
  round.dealtKitty = round.kitty.slice();
  round.deck = deck.slice(0, deck.length - round.kittySize);
  s.round = round;

  const landlordText =
    s.landlord === null
      ? 'first to declare becomes landlord'
      : `${seatName(s, s.landlord)} is landlord at ${levelLabel(s.levels[s.landlord]!)}`;
  log(s, 'round', `Round ${number} — ${landlordText}.`);
  if (s.rules.biddingStyle === 'turns') {
    while (round.deck.length > 0) dealOne(round, n);
    round.stage = 'bid';
    round.bidTurn = dealer;
    if (s.landlord !== null && s.levels[s.landlord] === 'NT') {
      log(s, 'bid', 'The landlord is playing no-trump: no bidding this round.');
      startExchange(s, s.landlord);
    }
  }
}

function dealOne(round: RoundState, n: number): void {
  const card = round.deck.pop();
  if (card === undefined) return;
  round.hands[round.dealTo]!.push(card);
  round.dealTo = (round.dealTo + 1) % n;
}

// ------------------------------------------------------------------ selectors

/** The level bids are made at by `seat`: the landlord's if there is one, else their own. */
export function bidLevelFor(state: TractorState, seat: number): Level {
  return state.levels[state.landlord ?? seat]!;
}

/** Who becomes (or is) the landlord, from the known landlord or the bids so far. */
export function landlordToBe(state: TractorState): number | null {
  if (state.landlord !== null) return state.landlord;
  const fw = firstAndWinner(state.round.bids, state.round.autobid);
  if (!fw) return null;
  return state.config.firstLandlordSelection === 'firstBid' ? fw.first.seat : fw.winner.seat;
}

function bidsAllowed(state: TractorState, seat: number): boolean {
  const r = state.round;
  if (state.status !== 'playing' || r.revealed > 0 || r.autobid) return false;
  if (r.stage === 'draw') return true;
  if (r.stage === 'bid') return r.bidTurn === seat;
  return false;
}

/** Every bid `seat` could make right now. */
export function validBidsFor(state: TractorState, seat: number): Bid[] {
  if (!bidsAllowed(state, seat)) return [];
  return validBids(seat, state.round.bids, state.round.hands[seat]!, bidLevelFor(state, seat), state.landlord !== null, 0, {
    bidPolicy: state.config.bidPolicy,
    jokerBidPolicy: state.config.jokerBidPolicy,
    bidReinforcementPolicy: state.config.bidReinforcementPolicy,
    numDecks: state.rules.numDecks,
  });
}

function noBidNeeded(state: TractorState): boolean {
  return state.landlord !== null && state.levels[state.landlord] === 'NT';
}

/** Live bidding is done dealing and the trump is settled (or settleable): `seat` may take the kitty. */
export function canPickUp(state: TractorState, seat: number): boolean {
  const r = state.round;
  if (state.status !== 'playing' || r.stage !== 'draw' || r.deck.length > 0) return false;
  if (r.bids.length === 0 && !r.autobid && !noBidNeeded(state)) return false;
  return landlordToBe(state) === seat;
}

/** Nobody bid and there's a landlord: they turn over kitty cards to find trump. */
export function canRevealKitty(state: TractorState, seat: number): boolean {
  const r = state.round;
  if (state.status !== 'playing' || state.landlord !== seat || noBidNeeded(state)) return false;
  if (r.bids.length > 0 || r.autobid || r.revealed >= r.kitty.length) return false;
  if (r.stage === 'draw') return r.deck.length === 0;
  if (r.stage === 'bid') return r.bidTurn === null;
  return false;
}

/** Nobody bid in the first round (live bidding): deal again. */
export function canRedeal(state: TractorState): boolean {
  const r = state.round;
  return (
    state.status === 'playing'
    && r.stage === 'draw'
    && r.deck.length === 0
    && state.landlord === null
    && r.bids.length === 0
    && !r.autobid
  );
}

/** The seat whose decision the round waits on, or null (live deal, between rounds). */
export function activeSeat(state: TractorState): number | null {
  if (state.status !== 'playing') return null;
  const r = state.round;
  switch (r.stage) {
    case 'draw': {
      if (r.deck.length > 0) return null;
      const ll = landlordToBe(state);
      if (ll !== null && (r.bids.length > 0 || r.autobid || noBidNeeded(state))) return ll;
      if (state.landlord !== null) return state.landlord; // reveal the kitty
      return r.dealer; // redeal
    }
    case 'bid':
      return r.bidTurn ?? state.landlord;
    case 'exchange':
      return r.landlord;
    case 'play':
      return r.trick ? nextToPlay(r.trick) : null;
    case 'roundOver':
      return null;
  }
}

export function isLandlordTeam(state: TractorState, seat: number): boolean {
  return state.round.landlordTeam.includes(seat);
}

/** Points the attackers have won so far (as far as the table knows who's on which side). */
export function attackerPointsSoFar(state: TractorState): number {
  const r = state.round;
  let pts = 0;
  r.captured.forEach((cards, seat) => {
    if (!r.landlordTeam.includes(seat)) pts += totalPoints(cards);
  });
  if (r.kittyBonus && !r.landlordTeam.includes(r.kittyBonus.seat)) pts += r.kittyBonus.points * r.kittyBonus.multiplier;
  return pts;
}

/** Why `cards` isn't a legal play for `seat` (a readable sentence), or null when it is. */
export function playProblemFor(state: TractorState, seat: number, cards: readonly Card[]): string | null {
  const r = state.round;
  if (r.stage !== 'play' || !r.trick || !r.trump) return "It isn't time to play cards.";
  const p = playProblem(r.trick, r.trump, r.hands[seat]!, seat, cards, trickRules(state));
  if (p !== 'illegal') return p;
  const shapes = r.trick.format ? requiredShapes(r.trump, r.trick.format, r.hands[seat]!, state.config.trickDrawPolicy) : null;
  return shapes ? `You have to follow with ${describeShapes(shapes)} in the led suit.` : 'You have to follow the led suit as far as you can.';
}

// ------------------------------------------------------------------ reducer

export function applyAction(state: TractorState, action: Action): TractorState {
  if (state.status !== 'playing') fail('The game is over.');
  const s = clone(state);
  try {
    reduce(s, action);
  } catch (e) {
    if (e instanceof TrickError) throw new IllegalAction(e.message);
    throw e;
  }
  return finalize(s);
}

function reduce(s: TractorState, a: Action): void {
  const r = s.round;
  const n = s.seats.length;
  switch (a.type) {
    case 'draw': {
      if (r.stage !== 'draw' || r.deck.length === 0) fail('Nothing left to deal.');
      for (let i = 0; i < n && r.deck.length > 0; i++) dealOne(r, n);
      return;
    }
    case 'bid': {
      if (!bidsAllowed(s, a.seat)) fail("You can't bid now.");
      const bid: Bid = { seat: a.seat, card: a.card, count: a.count, epoch: 0 };
      if (!isValidBid(bid, validBidsFor(s, a.seat))) fail('That bid is not allowed.');
      const reinforcing = r.bids.length > 0 && r.bids[r.bids.length - 1]!.seat === a.seat;
      r.bids.push(bid);
      log(s, 'bid', `${seatName(s, a.seat)} ${reinforcing ? 'reinforces with' : 'declares'} ${bidText(bid)}.`, a.seat);
      if (r.stage === 'bid') {
        r.passes = 0;
        r.bidTurn = (a.seat + 1) % n;
      }
      return;
    }
    case 'pass': {
      if (r.stage !== 'bid' || r.bidTurn !== a.seat) fail("It isn't your turn to bid.");
      r.passes++;
      log(s, 'bid', `${seatName(s, a.seat)} passes.`, a.seat);
      if (r.bids.length > 0 && r.passes >= n - 1) {
        startExchange(s, landlordToBe(s)!);
      } else if (r.bids.length === 0 && r.passes >= n) {
        if (s.landlord === null) {
          log(s, 'bid', 'Nobody declared — reshuffling.');
          beginRound(s, r.number, (r.dealer + 1) % n);
        } else {
          r.bidTurn = null;
          log(s, 'bid', `Nobody declared — ${seatName(s, s.landlord)} turns over the kitty to find trump.`);
        }
      } else {
        r.bidTurn = (a.seat + 1) % n;
      }
      return;
    }
    case 'revealKitty': {
      if (!canRevealKitty(s, a.seat)) fail("You can't turn over the kitty now.");
      revealNext(s, a.seat);
      if (r.autobid) startExchange(s, a.seat);
      return;
    }
    case 'redeal': {
      if (!canRedeal(s)) fail('There is nothing to redeal.');
      log(s, 'bid', 'Nobody declared — reshuffling.');
      beginRound(s, r.number, (r.dealer + 1) % n);
      return;
    }
    case 'pickUpKitty': {
      if (!canPickUp(s, a.seat)) fail("You can't take the kitty now.");
      startExchange(s, a.seat);
      return;
    }
    case 'exchange':
      return doExchange(s, a.seat, a.bury, a.friends);
    case 'play':
      return doPlay(s, a.seat, a.cards, a.format);
    case 'endRoundEarly': {
      if (!Number.isInteger(a.seat) || a.seat < 0 || a.seat >= n) fail('Unknown seat.');
      if (!canEndRoundEarly(s)) fail('The rest of the round can still change the result — play on.');
      log(s, 'round', `${seatName(s, a.seat)} ends the round early: nothing left can change the result.`, a.seat);
      r.kittyBonus = null;
      finishRound(s, true);
      return;
    }
    case 'nextRound': {
      if (r.stage !== 'roundOver' || !r.result) fail('The round is not over.');
      s.landlord = r.result.nextLandlord;
      beginRound(s, r.number + 1, r.result.nextLandlord);
      return;
    }
    default:
      fail('Unknown action.');
  }
}

export function bidText(bid: Pick<Bid, 'card' | 'count'>): string {
  const what = isJoker(bid.card) ? `${bid.card === 'BJ' ? 'big' : 'small'} joker${bid.count > 1 ? 's' : ''}` : cardShort(bid.card);
  return `${bid.count > 1 ? `${bid.count}× ` : ''}${what}`;
}

function revealNext(s: TractorState, seat: number): void {
  const r = s.round;
  const level = s.levels[seat]!;
  const card = r.kitty[r.revealed]!;
  const policy = s.config.kittyBidPolicy;
  log(s, 'bid', `${seatName(s, seat)} turns over ${cardLabel(card)} from the kitty.`, seat);
  if (policy === 'firstCard' || isJoker(card) || cardRank(card) === level) {
    r.autobid = { seat, card, count: 1, epoch: 0 };
  } else if (r.revealed >= r.kitty.length - 1) {
    const order: Trump = { kind: 'noTrump', number: level === 'NT' ? null : level };
    const highest = r.kitty.slice().sort((x, y) => compareCards(order, x, y)).pop()!;
    r.autobid = { seat, card: highest, count: 1, epoch: 0 };
  }
  r.revealed++;
}

function startExchange(s: TractorState, landlord: number): void {
  const r = s.round;
  const level = s.levels[landlord]!;
  let trump: Trump;
  if (level === 'NT') {
    trump = { kind: 'noTrump', number: null };
  } else {
    const win = firstAndWinner(r.bids, r.autobid)!.winner;
    trump = isJoker(win.card) ? { kind: 'noTrump', number: level } : { kind: 'standard', suit: cardSuit(win.card)!, number: level };
  }
  s.landlord = landlord;
  r.landlord = landlord;
  r.trump = trump;
  r.bidTurn = null;
  r.hands[landlord]!.push(...r.kitty);
  r.kitty = [];
  r.stage = 'exchange';
  log(s, 'bid', `${seatName(s, landlord)} is the landlord. Trump: ${trumpLabel(trump)}.`, landlord);
}

/** Problems with a friend call under the match's rules ([] = fine). */
export function friendCallProblems(state: TractorState, calls: readonly FriendCall[]): string[] {
  const r = state.round;
  const errors: string[] = [];
  const want = state.rules.gameMode === 'findingFriends' ? state.rules.numFriends : 0;
  if (calls.length !== want) errors.push(want === 0 ? 'No friends are called in classic tractor.' : `Call exactly ${want} friend${want === 1 ? '' : 's'}.`);
  const trump = r.trump;
  if (!trump) return errors;
  const keys = new Set<string>();
  const policy = state.config.friendSelectionPolicy;
  for (const call of calls) {
    const key = `${call.card}#${call.ordinal}`;
    if (keys.has(key)) errors.push('Each friend call must be different.');
    keys.add(key);
    if (!Number.isInteger(call.ordinal) || call.ordinal < 1 || call.ordinal > state.rules.numDecks) {
      errors.push(`Pick which copy (1–${state.rules.numDecks}).`);
    }
    if (policy !== 'trumpsIncluded') {
      if (isJoker(call.card) || cardRank(call.card) === trump.number) errors.push("You can't call a joker or a trump-number card.");
      else if (trump.kind === 'standard' && cardSuit(call.card) === trump.suit) errors.push("You can't call a trump-suit card.");
    }
    if (policy === 'highestCardNotAllowed') {
      const rank = cardRank(call.card);
      if (rank === 'A' || (trump.number === 'A' && rank === 'K')) errors.push("You can't call the highest card.");
    }
    if (policy === 'pointCardNotAllowed') {
      const landlordLevel = state.levels[r.landlord ?? 0];
      if (cardPoints(call.card) > 0 && !(landlordLevel === 'A' && cardRank(call.card) === 'K')) errors.push("You can't call a point card.");
    }
  }
  return [...new Set(errors)];
}

function doExchange(s: TractorState, seat: number, bury: readonly Card[], calls: readonly FriendCall[]): void {
  const r = s.round;
  if (r.stage !== 'exchange' || r.landlord !== seat) fail("You aren't exchanging the kitty.");
  if (bury.length !== r.kittySize) fail(`Bury exactly ${r.kittySize} card${r.kittySize === 1 ? '' : 's'}.`);
  if (!containsAll(r.hands[seat]!, bury)) fail("Those cards aren't all in your hand.");
  const problems = friendCallProblems(s, calls);
  if (problems.length) fail(problems[0]!);
  r.hands[seat] = removeCards(r.hands[seat]!, bury);
  r.kitty = bury.slice();
  r.friends = calls.map((c): Friend => ({ card: c.card, ordinal: c.ordinal, skip: c.ordinal - 1, seat: null }));
  if (s.rules.gameMode === 'tractor') {
    r.landlordTeam = s.seats.map((_, i) => i).filter((i) => i % 2 === seat % 2);
  } else {
    r.landlordTeam = [seat];
    if (calls.length) {
      log(s, 'friend', `${seatName(s, seat)} calls ${calls.map((c) => `the ${ordinalText(c.ordinal)} ${cardLabel(c.card)}`).join(' and ')}.`, seat);
    }
  }
  r.trick = newTrick(seat, s.seats.length);
  r.stage = 'play';
}

export function ordinalText(n: number): string {
  return ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth'][n] ?? `${n}th`;
}

function doPlay(s: TractorState, seat: number, cards: readonly Card[], format: number | undefined): void {
  const r = s.round;
  if (r.stage !== 'play' || !r.trick || !r.trump) fail("It isn't time to play cards.");
  const leading = r.trick.plays.length === 0;
  const out = playCards(r.trick, r.trump, r.hands, seat, cards, trickRules(s), format);
  r.trick = out.trick;
  r.hands[seat] = out.hand;
  const played = r.trick.plays[r.trick.plays.length - 1]!.cards;
  if (out.failedThrow) {
    const by = seatName(s, out.failedThrow.betterPlayer);
    log(
      s,
      'throw',
      `${seatName(s, seat)} tried to throw ${out.failedThrow.attempted.map(cardShort).join(' ')}, but ${by} can beat part of it — only ${out.failedThrow.forced.map(cardShort).join(' ')} goes down.`,
      seat,
      `${seatName(s, seat)}'s throw failed — ${by} could beat part of it.`,
    );
    if (s.config.throwPenalty === 'tenPointsPerAttempt') r.penalties[seat]! += 10;
  } else {
    log(
      s,
      'play',
      `${seatName(s, seat)} ${leading ? 'leads' : 'plays'} ${played.map(cardShort).join(' ')}.`,
      seat,
      `${seatName(s, seat)} ${leading ? 'leads' : 'plays'} ${played.length} card${played.length === 1 ? '' : 's'}.`,
    );
  }
  revealFriends(s, seat, played);

  if (!trickComplete(r.trick)) return;
  const result = completeTrick(r.trick);
  r.captured[result.winner]!.push(...result.pointCards);
  r.tricks.push({ leader: r.trick.order[0]!, suit: r.trick.format!.suit, plays: r.trick.plays, winner: result.winner, points: result.points });
  const last = r.hands.every((h) => h.length === 0);
  log(
    s,
    'trick',
    `${seatName(s, result.winner)} wins the trick${result.points ? ` (+${result.points})` : ''}.`,
    result.winner,
    `${seatName(s, result.winner)} wins the trick.`,
  );
  if (last) {
    const kittyPts = totalPoints(r.kitty);
    const mult = s.config.kittyPenalty === 'power' ? 2 ** result.largestUnitSize : 2 * result.largestUnitSize;
    r.kittyBonus = { seat: result.winner, points: kittyPts, multiplier: mult };
    if (kittyPts > 0) log(s, 'trick', `The kitty holds ${kittyPts} points, ×${mult} to the last trick's winner.`);
    finishRound(s);
  } else {
    r.trick = newTrick(result.winner, s.seats.length);
  }
}

function revealFriends(s: TractorState, seat: number, played: readonly Card[]): void {
  const r = s.round;
  if (s.rules.gameMode !== 'findingFriends') return;
  for (const card of played) {
    for (const f of r.friends) {
      if (f.card !== card) continue;
      if (f.skip > 0) {
        f.skip--;
        continue;
      }
      if (f.seat !== null) continue;
      const already = r.landlordTeam.includes(seat);
      if (already && s.config.multipleJoinPolicy === 'noDoubleJoin') continue;
      f.seat = seat;
      if (already) {
        log(s, 'friend', `${seatName(s, seat)} played the called ${cardLabel(card)} but is already on the landlord's team.`, seat);
      } else {
        r.landlordTeam.push(seat);
        log(s, 'friend', `${seatName(s, seat)} played the ${ordinalText(f.ordinal)} ${cardLabel(card)} and joins the landlord!`, seat);
      }
    }
  }
}

/** Points the attackers have from tricks, plus throw penalties so far (no kitty). */
function attackerTrickPoints(r: RoundState): { points: number; penaltyAdjustment: number } {
  const team = r.landlordTeam;
  let points = 0;
  r.captured.forEach((cards, seat) => {
    if (!team.includes(seat)) points += totalPoints(cards);
  });
  let penaltyAdjustment = 0;
  r.penalties.forEach((p, seat) => {
    if (p > 0) penaltyAdjustment += team.includes(seat) ? p : -p;
  });
  return { points: points + penaltyAdjustment, penaltyAdjustment };
}

export interface RoundOutcome {
  landlordWon: boolean;
  /** Bonus level included. */
  landlordDelta: number;
  attackerDelta: number;
  bonusLevel: boolean;
}

/** How the round scores if the attackers finish on `attackerPoints`, with the landlord's team as it stands. */
export function outcomeFor(state: TractorState, attackerPoints: number): RoundOutcome {
  const params = scoringParams(state.config);
  const m = materialize(params, state.rules.numDecks, state.rules.numDecks * 100);
  const score = scoreRound(m, attackerPoints);
  const team = state.round.landlordTeam;
  const smallerTeam = state.rules.gameMode === 'findingFriends' && team.length < state.rules.numFriends + 1;
  const bonusLevel = score.landlordWon && params.bonusLevelForSmallerTeam && smallerTeam;
  return { landlordWon: score.landlordWon, landlordDelta: score.landlordDelta + (bonusLevel ? 1 : 0), attackerDelta: score.attackerDelta, bonusLevel };
}

/**
 * The lowest and highest totals the attackers could still finish the round
 * on, worked out from public information only (so the "end early" button
 * gives nothing away): the point cards nobody has seen go anywhere (with
 * "hide landlord points", the defenders' captures count as unseen too),
 * any kitty points could be multiplied by the most a last trick allows, and
 * every remaining trick could add a failed-throw penalty either way. Null
 * between deals, mid-trick, or while a friend in Finding Friends is unknown
 * (who's on which side can still change).
 */
export function attackerPointRange(state: TractorState): { now: number; min: number; max: number } | null {
  const r = state.round;
  if (state.status !== 'playing' || r.stage !== 'play' || !r.trick || r.trick.plays.length > 0 || r.tricks.length === 0) return null;
  if (state.rules.gameMode === 'findingFriends' && r.friends.some((f) => f.seat === null)) return null;
  const tricksLeft = r.hands[0]!.length;
  if (tricksLeft === 0) return null;
  const now = attackerTrickPoints(r).points;
  const total = state.rules.numDecks * 100 - totalPoints(r.removed);
  let seen = 0;
  r.captured.forEach((cards, seat) => {
    if (!state.config.hideLandlordPoints || !r.landlordTeam.includes(seat)) seen += totalPoints(cards);
  });
  const unseen = Math.max(0, total - seen);
  const kittyMax = Math.min(unseen, 10 * r.kittySize);
  const multiplier = state.config.kittyPenalty === 'power' ? 2 ** tricksLeft : 2 * tricksLeft;
  const penalties = state.config.throwPenalty === 'tenPointsPerAttempt' ? 10 * tricksLeft : 0;
  return { now, min: now - penalties, max: now + (unseen - kittyMax) + kittyMax * multiplier + penalties };
}

/** True when no way of playing out the rest of the round could change its result (levels, who leads next). */
export function canEndRoundEarly(state: TractorState): boolean {
  const range = attackerPointRange(state);
  if (!range) return false;
  if ((range.max - range.min) / 5 > 20000) return false;
  const key = (p: number) => JSON.stringify(outcomeFor(state, p));
  const now = key(range.now);
  for (let p = range.min; p <= range.max; p += 5) if (key(p) !== now) return false;
  return true;
}

function finishRound(s: TractorState, endedEarly = false): void {
  const r = s.round;
  const n = s.seats.length;
  const landlord = r.landlord!;
  const team = r.landlordTeam;
  const trickPart = attackerTrickPoints(r);
  let attackerPoints = trickPart.points;
  const penaltyAdjustment = trickPart.penaltyAdjustment;
  const kb = r.kittyBonus;
  const kittyToAttackers = !!kb && !team.includes(kb.seat);
  if (kb && kittyToAttackers) attackerPoints += kb.points * kb.multiplier;

  const score = outcomeFor(s, attackerPoints);
  const { bonusLevel, landlordDelta } = score;
  const landlordLevel = s.levels[landlord]!;

  const advancements = s.levels.map((level, seat) => {
    const defending = team.includes(seat);
    return advancePlayer(level, defending ? landlordDelta : score.attackerDelta, defending, landlordLevel, s.config.maxRank, s.config.advancementPolicy);
  });
  advancements.forEach((adv, seat) => {
    s.levels[seat] = adv.to;
  });

  let next = (landlord + 1) % n;
  for (let i = 0; i < n; i++) {
    if (score.landlordWon === team.includes(next)) break;
    next = (next + 1) % n;
  }
  const winners = advancements.map((a, seat) => (a.finished ? seat : -1)).filter((x) => x >= 0);
  const result: RoundResult = {
    round: r.number,
    landlord,
    landlordTeam: team.slice(),
    trump: r.trump!,
    attackerPoints,
    kittyPoints: kb?.points ?? 0,
    kittyMultiplier: kb?.multiplier ?? 0,
    kittyToAttackers,
    penaltyAdjustment,
    landlordWon: score.landlordWon,
    landlordDelta,
    attackerDelta: score.attackerDelta,
    bonusLevel,
    advancements,
    nextLandlord: next,
    kitty: r.kitty.slice(),
    winners,
    ...(endedEarly ? { endedEarly: true } : {}),
  };
  r.result = result;
  r.stage = 'roundOver';
  r.trick = null;
  s.history.push(result);

  const outcome = score.landlordWon
    ? landlordDelta > 0
      ? `The landlord's team goes up ${landlordDelta} level${landlordDelta === 1 ? '' : 's'}${bonusLevel ? ' (bonus level for winning short-handed)' : ''}`
      : "The landlord's team holds"
    : score.attackerDelta > 0
      ? `The attackers take over and go up ${score.attackerDelta} level${score.attackerDelta === 1 ? '' : 's'}`
      : 'The attackers take over';
  log(s, 'round', `Attackers scored ${attackerPoints}. ${outcome}.`);
  if (winners.length > 0) {
    s.status = 'gameOver';
    s.winners = winners;
    log(s, 'round', `${winners.map((w) => seatName(s, w)).join(' & ')} passed ${levelLabel(s.config.maxRank)} and win the game!`);
  }
}

function finalize(s: TractorState): TractorState {
  const seat = activeSeat(s);
  s.activePlayerId = seat === null ? null : s.seats[seat]!.id;
  if (s.status === 'gameOver') {
    s.finalScores = Object.fromEntries(s.seats.map((x, i) => [x.id, levelIndex(s.levels[i]!)]));
  }
  return s;
}

/** Hand sorted for display: by suit (trump last), then strength. */
export function sortedHand(state: TractorState, seat: number): Card[] {
  const r = state.round;
  const trump: Trump = r.trump ?? displayTrump(state, seat);
  return r.hands[seat]!.slice().sort((a, b) => compareCards(trump, a, b));
}

/** Before trump is set: sort as if the leading bid stood (or no-trump at the bid level). */
export function displayTrump(state: TractorState, seat: number): Trump {
  const level = bidLevelFor(state, seat);
  const number = level === 'NT' ? null : level;
  const r = state.round;
  const lead = r.autobid ?? r.bids[r.bids.length - 1];
  if (lead && !isJoker(lead.card) && number !== null) return { kind: 'standard', suit: cardSuit(lead.card)!, number };
  return { kind: 'noTrump', number };
}
