// Tractor learned agent — the observation encoder (v3).
//
// One flat float vector per decision, from a seat's Observation only (no
// hidden cards): a global block, then MAX_SEATS (12) seat blocks of the same
// layout, so one shared seat network reads every seat (runtime.ts). Cards are
// counted in trump-relative slots (slots.ts), so every trump suit and number
// looks alike.
//
// Seats are relative to the decider and placed from both ends: the decider is
// slot 0, the next players count up from slot 1, the previous ones down from
// slot 11. The player after me is always slot 1 and the one before me always
// slot 11, at any table size (in 4-player Tractor: 0 me, 1 left opponent,
// 10 partner, 11 right opponent). Absent seats are all zeros (`present` = 0).
//
//   global   card planes (8 × 72): my hand · unseen · the trick's winning
//            play · kitty as dealt / as buried (landlord only) · cards turned
//            over from the kitty · unrevealed friend calls · copies still to
//            come before a call's friend copy
//            + table, trump, points vs threshold, the trick's format and
//            who's winning it, bidding, my role, rules (GLOBAL fields)
//   seat ×12 card planes (8 × 72): led · followed (in the led suit) · ruffed
//            (trump on a side-suit lead) · discarded (neither) · first
//            discard (the cards off-suit in the seat's first discard) · this
//            trick · the last completed trick · shown (still held, publicly
//            known: bid cards and failed throws, less what it has played)
//            + level, side, captured points, tricks won, cards left, known
//            voids, this trick's state, bids, position (SEAT fields)
//
// ENCODER_VERSION goes in every policy file; a policy trained on another
// version is refused (bump it with any change to what a slot means).

import { LEVELS, levelIndex, totalPoints, type Card, type Level } from '@/game/cards';
import { MAX_PLAYERS } from '@/game/config';
import { scoringParams } from '@/game/engine';
import { materialize } from '@/game/scoring';
import { unitSize, type TrickUnit } from '@/game/format';
import { effectiveSuit, type EffSuit, type Trump } from '@/game/trump';
import type { PlayedCards } from '@/game/trick';
import { knownVoids, unseenCounts, type Observation } from '../observation';
import { CARD_SLOTS, GROUPS, TRUMP_GROUP, addCards, sideBlock, slotOf } from './slots';

export const ENCODER_VERSION = 3;
/** Every table the game allows. */
export const MAX_SEATS = MAX_PLAYERS;

export interface Segment {
  name: string;
  offset: number;
  size: number;
}

export class Layout {
  readonly segments: Segment[] = [];
  dim = 0;
  add(name: string, size: number): number {
    const offset = this.dim;
    this.segments.push({ name, offset, size });
    this.dim += size;
    return offset;
  }
}

/** A seat's slot: from both ends, so the next / previous players keep their slots at any table size. */
export function seatSlot(rel: number, n: number): number {
  const forward = Math.ceil(n / 2);
  return rel < forward ? rel : MAX_SEATS - (n - rel);
}

// ---------------------------------------------------------------- seat block

const SEAT = new Layout();
const S = {
  present: SEAT.add('present', 1), // must stay first: the model masks absent seats with it
  isMe: SEAT.add('isMe', 1),
  slot: SEAT.add('slot', MAX_SEATS),
  relFrac: SEAT.add('relFrac', 1),
  level: SEAT.add('level', LEVELS.length),
  levelVsMine: SEAT.add('levelVsMine', 1),
  levelsToMax: SEAT.add('levelsToMax', 1),
  isLandlord: SEAT.add('isLandlord', 1),
  knownDefender: SEAT.add('knownDefender', 1),
  knownAttacker: SEAT.add('knownAttacker', 1),
  sideUnknown: SEAT.add('sideUnknown', 1),
  knownPartner: SEAT.add('knownPartner', 1),
  captured: SEAT.add('captured', 1),
  tricksWon: SEAT.add('tricksWon', 1),
  handLeft: SEAT.add('handLeft', 1),
  voids: SEAT.add('voids', GROUPS),
  playedThisTrick: SEAT.add('playedThisTrick', 1),
  winningTrick: SEAT.add('winningTrick', 1),
  leaderThisTrick: SEAT.add('leaderThisTrick', 1),
  nextToPlay: SEAT.add('nextToPlay', 1),
  ledLastTrick: SEAT.add('ledLastTrick', 1),
  wonLastTrick: SEAT.add('wonLastTrick', 1),
  penalty: SEAT.add('penalty', 1),
  bidGroup: SEAT.add('bidGroup', 6),
  bidCount: SEAT.add('bidCount', 1),
  holdsWinningBid: SEAT.add('holdsWinningBid', 1),
  bidDone: SEAT.add('bidDone', 1),
  isDealer: SEAT.add('isDealer', 1),
  led: SEAT.add('led', CARD_SLOTS),
  followed: SEAT.add('followed', CARD_SLOTS),
  ruffed: SEAT.add('ruffed', CARD_SLOTS),
  discarded: SEAT.add('discarded', CARD_SLOTS),
  firstDiscard: SEAT.add('firstDiscard', CARD_SLOTS),
  thisTrick: SEAT.add('thisTrick', CARD_SLOTS),
  lastTrick: SEAT.add('lastTrick', CARD_SLOTS),
  shown: SEAT.add('shown', CARD_SLOTS),
};
export const SEAT_DIM = SEAT.dim;
export const SEAT_LAYOUT = SEAT;

// ---------------------------------------------------------------- global block

const GLOBAL = new Layout();
const G = {
  hand: GLOBAL.add('hand', CARD_SLOTS),
  unseen: GLOBAL.add('unseen', CARD_SLOTS),
  trickWinner: GLOBAL.add('trickWinner', CARD_SLOTS),
  kittyDealt: GLOBAL.add('kittyDealt', CARD_SLOTS),
  kittyBuried: GLOBAL.add('kittyBuried', CARD_SLOTS),
  revealedKitty: GLOBAL.add('revealedKitty', CARD_SLOTS),
  calls: GLOBAL.add('calls', CARD_SLOTS),
  callsBefore: GLOBAL.add('callsBefore', CARD_SLOTS),
  players: GLOBAL.add('players', MAX_SEATS - 1), // 2..12
  decks: GLOBAL.add('decks', 1),
  mode: GLOBAL.add('mode', 2),
  numFriends: GLOBAL.add('numFriends', 1),
  friendsHidden: GLOBAL.add('friendsHidden', 1),
  stage: GLOBAL.add('stage', 4),
  trumpKind: GLOBAL.add('trumpKind', 4),
  trumpNumber: GLOBAL.add('trumpNumber', 14),
  numberIsPoints: GLOBAL.add('numberIsPoints', 1),
  maxRank: GLOBAL.add('maxRank', 2),
  landlordLevelsToMax: GLOBAL.add('landlordLevelsToMax', 1),
  defendersAtMax: GLOBAL.add('defendersAtMax', 1),
  attackersAtMax: GLOBAL.add('attackersAtMax', 1),
  attackerPoints: GLOBAL.add('attackerPoints', 1),
  threshold: GLOBAL.add('threshold', 1),
  unseenPoints: GLOBAL.add('unseenPoints', 1),
  kittySize: GLOBAL.add('kittySize', 1),
  kittyPoints: GLOBAL.add('kittyPoints', 1),
  trickProgress: GLOBAL.add('trickProgress', 1),
  myHandLeft: GLOBAL.add('myHandLeft', 1),
  deckLeft: GLOBAL.add('deckLeft', 1),
  leading: GLOBAL.add('leading', 1),
  trickPosition: GLOBAL.add('trickPosition', 1),
  ledGroup: GLOBAL.add('ledGroup', GROUPS),
  formatCards: GLOBAL.add('formatCards', 1),
  formatUnits: GLOBAL.add('formatUnits', 4), // singles, pairs, triples+, tractors
  formatLongest: GLOBAL.add('formatLongest', 1),
  trickPoints: GLOBAL.add('trickPoints', 1),
  winnerIsPartner: GLOBAL.add('winnerIsPartner', 1),
  winnerIsMe: GLOBAL.add('winnerIsMe', 1),
  winnerIsOpponent: GLOBAL.add('winnerIsOpponent', 1),
  toPlayAfterMe: GLOBAL.add('toPlayAfterMe', 1),
  opponentsAfterMe: GLOBAL.add('opponentsAfterMe', 1),
  partnersAfterMe: GLOBAL.add('partnersAfterMe', 1),
  lastTrickPoints: GLOBAL.add('lastTrickPoints', 1),
  lastTrickWinner: GLOBAL.add('lastTrickWinner', 4), // me, partner, opponent, unknown side
  hasBid: GLOBAL.add('hasBid', 1),
  winningBidCount: GLOBAL.add('winningBidCount', 1),
  winningBidGroup: GLOBAL.add('winningBidGroup', 6),
  winningBidMine: GLOBAL.add('winningBidMine', 1),
  landlordToBeMe: GLOBAL.add('landlordToBeMe', 1),
  passes: GLOBAL.add('passes', 1),
  revealed: GLOBAL.add('revealed', 1),
  role: GLOBAL.add('role', 4), // landlord, known defender, known attacker, holds a called card
  drawPolicy: GLOBAL.add('drawPolicy', 5),
  throwEval: GLOBAL.add('throwEval', 3),
  kittyPower: GLOBAL.add('kittyPower', 1),
  throwPenalty: GLOBAL.add('throwPenalty', 1),
};
export const GLOBAL_DIM = GLOBAL.dim;
export const GLOBAL_LAYOUT = GLOBAL;

// ---------------------------------------------------------------- whole vector

export const OBS_DIM = GLOBAL_DIM + MAX_SEATS * SEAT_DIM;
const seatOffset = (slot: number) => GLOBAL_DIM + slot * SEAT_DIM;

const DRAW_POLICIES = ['noProtections', 'longerTuplesProtected', 'onlyDrawTractorOnTractor', 'longerTuplesProtectedAndOnlyDrawTractorOnTractor', 'noFormatBasedDraw'];
const THROW_EVALS = ['all', 'highest', 'trickUnitLength'];
const STAGES = ['draw', 'bid', 'exchange', 'play'];

/** Trump to slot cards by: the real one, or before it's declared "no trump suit at the bidding level". */
export function slotTrump(obs: Observation): Trump {
  if (obs.trump) return obs.trump;
  return { kind: 'noTrump', number: obs.bidLevel === 'NT' ? null : obs.bidLevel };
}

/** Bid card → 6-way group: side block 0..3 of its suit, small joker, big joker. */
export function bidGroup(trump: Trump, card: Card): number {
  if (card === 'LJ') return 4;
  if (card === 'BJ') return 5;
  return Math.max(0, sideBlock(trump, card[1] as never));
}

export function unitKinds(units: readonly TrickUnit[]): { singles: number; pairs: number; triples: number; tractors: number; longest: number } {
  const k = { singles: 0, pairs: 0, triples: 0, tractors: 0, longest: 0 };
  for (const u of units) {
    if (u.kind === 'tractor') k.tractors++;
    else if (u.count === 1) k.singles++;
    else if (u.count === 2) k.pairs++;
    else k.triples++;
    k.longest = Math.max(k.longest, unitSize(u));
  }
  return k;
}

/** Which side each seat is known to be on: 'D' defender, 'A' attacker, '?' unknown. */
export function knownSides(obs: Observation): Array<'D' | 'A' | '?'> {
  const n = obs.numPlayers;
  const out: Array<'D' | 'A' | '?'> = new Array(n).fill('?');
  if (obs.landlord === null || obs.stage === 'draw' || obs.stage === 'bid') return out;
  if (obs.rules.gameMode === 'tractor') {
    for (let i = 0; i < n; i++) out[i] = i % 2 === obs.landlord % 2 ? 'D' : 'A';
    return out;
  }
  const allKnown = obs.friends.every((f) => f.seat !== null);
  for (let i = 0; i < n; i++) out[i] = obs.landlordTeam.includes(i) || i === obs.landlord ? 'D' : allKnown ? 'A' : '?';
  return out;
}

export function threshold(obs: Observation): number {
  const m = materialize(scoringParams(obs.config), obs.rules.numDecks, obs.rules.numDecks * 100);
  return m.landlordWins[m.landlordWins.length - 1]!.end;
}

/** Group (side block or trump) of a card under the slot trump. */
export function groupOf(trump: Trump, card: Card): number {
  if (effectiveSuit(trump, card) === 'T') return TRUMP_GROUP;
  return Math.max(0, sideBlock(trump, card[1] as never));
}

export function encodeObservation(obs: Observation, out: Float32Array = new Float32Array(OBS_DIM)): Float32Array {
  out.fill(0);
  const n = obs.numPlayers;
  const me = obs.seat;
  const slotOfSeat = (seat: number) => seatSlot((seat - me + n) % n, n);
  const decks = obs.rules.numDecks;
  const w = 1 / decks;
  const trump = slotTrump(obs);
  const realTrump = obs.trump;
  const pointsScale = 1 / (100 * decks);
  const so = (seat: number) => seatOffset(slotOfSeat(seat));

  // ---- global card planes
  addCards(out, G.hand, trump, obs.hand, w);
  const unseen = unseenCounts(obs);
  let unseenPoints = 0;
  for (const [c, k] of unseen) {
    if (k <= 0) continue;
    out[G.unseen + slotOf(trump, c)]! += k * w;
    unseenPoints += totalPoints([c]) * k;
  }
  const trick = obs.trick;
  if (trick) {
    const win = trick.plays.find((p) => p.seat === trick.winner);
    if (win) addCards(out, G.trickWinner, trump, win.cards, w);
  }
  if (obs.dealtKitty) addCards(out, G.kittyDealt, trump, obs.dealtKitty, w);
  if (obs.kitty) addCards(out, G.kittyBuried, trump, obs.kitty, w);
  addCards(out, G.revealedKitty, trump, obs.revealedKitty, w);
  for (const f of obs.friends) {
    if (f.seat !== null) continue;
    const s = slotOf(trump, f.card);
    out[G.calls + s]! += 1 / Math.max(1, obs.rules.numFriends);
    out[G.callsBefore + s] = Math.max(out[G.callsBefore + s]!, f.skip * w);
  }

  // ---- seat card planes: how each card was played
  const tricksWon = new Array<number>(n).fill(0);
  const discardedBefore = new Array<boolean>(n).fill(false);
  const played = Array.from({ length: n }, () => new Map<Card, number>());
  const classify = (leader: number, suit: EffSuit | null, plays: readonly PlayedCards[]) => {
    const firstHere: Card[][] = Array.from({ length: n }, () => []);
    for (const p of plays) {
      const o = so(p.seat);
      for (const c of p.cards) played[p.seat]!.set(c, (played[p.seat]!.get(c) ?? 0) + 1);
      if (p.seat === leader || !suit || !realTrump) {
        addCards(out, o + S.led, trump, p.cards, w);
        continue;
      }
      for (const c of p.cards) {
        const es = effectiveSuit(realTrump, c);
        const plane = es === suit ? S.followed : es === 'T' ? S.ruffed : S.discarded;
        out[o + plane + slotOf(trump, c)]! += w;
        if (plane === S.discarded && !discardedBefore[p.seat]) firstHere[p.seat]!.push(c);
      }
    }
    for (let seat = 0; seat < n; seat++) {
      if (firstHere[seat]!.length === 0) continue;
      addCards(out, so(seat) + S.firstDiscard, trump, firstHere[seat]!, w);
      discardedBefore[seat] = true;
    }
  };
  for (const t of obs.tricks) {
    classify(t.leader, t.suit, t.plays);
    tricksWon[t.winner]!++;
  }
  if (trick) {
    classify(trick.order[0]!, trick.format?.suit ?? null, trick.plays);
    for (const p of trick.plays) addCards(out, so(p.seat) + S.thisTrick, trump, p.cards, w);
  }
  const last = obs.tricks[obs.tricks.length - 1];
  if (last) for (const p of last.plays) addCards(out, so(p.seat) + S.lastTrick, trump, p.cards, w);

  // Shown: bid cards and failed-throw cards still held (less what the seat has played since).
  const shown = Array.from({ length: n }, () => new Map<Card, number>());
  for (const b of obs.bids) shown[b.seat]!.set(b.card, Math.max(shown[b.seat]!.get(b.card) ?? 0, b.count));
  const allPlays = [...obs.tricks.flatMap((t) => t.plays), ...(trick ? trick.plays : [])];
  for (const p of allPlays) {
    if (p.badThrowCards.length === 0) continue;
    const m = shown[p.seat]!;
    const k = new Map<Card, number>();
    for (const c of p.badThrowCards) k.set(c, (k.get(c) ?? 0) + 1);
    for (const [c, x] of k) m.set(c, Math.max(m.get(c) ?? 0, x + (played[p.seat]!.get(c) ?? 0)));
  }
  for (let seat = 0; seat < n; seat++) {
    if (seat === me) continue;
    const o = so(seat);
    for (const [c, k] of shown[seat]!) {
      const left = k - (played[seat]!.get(c) ?? 0);
      if (left > 0) out[o + S.shown + slotOf(trump, c)]! += left * w;
    }
  }

  // ---- seat features
  const sides = knownSides(obs);
  const mySide = sides[me]!;
  const sideOf = (seat: number) => (seat === me ? 0 : mySide !== '?' && sides[seat] === mySide ? 1 : mySide !== '?' && sides[seat] !== '?' ? 2 : 3);
  const voids = knownVoids(obs);
  const maxIdx = levelIndex(obs.config.maxRank);
  const winning = obs.autobid ?? obs.bids[obs.bids.length - 1] ?? null;
  const lastBid = new Map<number, { card: Card; count: number }>();
  for (const b of obs.bids) lastBid.set(b.seat, b);
  const initialHand = Math.max(1, obs.hand.length + obs.tricks.reduce((k, t) => k + (t.plays.find((p) => p.seat === me)?.cards.length ?? 0), 0));
  const totalTricks = obs.tricks.length + Math.max(1, obs.hand.length);
  for (let seat = 0; seat < n; seat++) {
    const slot = slotOfSeat(seat);
    const o = seatOffset(slot);
    const lvl = obs.levels[seat]!;
    out[o + S.present] = 1;
    out[o + S.isMe] = seat === me ? 1 : 0;
    out[o + S.slot + slot] = 1;
    out[o + S.relFrac] = ((seat - me + n) % n) / n;
    out[o + S.level + levelIndex(lvl)] = 1;
    out[o + S.levelVsMine] = (levelIndex(lvl) - levelIndex(obs.levels[me]!)) / 12;
    out[o + S.levelsToMax] = (maxIdx - levelIndex(lvl)) / 13;
    out[o + S.isLandlord] = obs.landlord === seat ? 1 : 0;
    out[o + S.knownDefender] = sides[seat] === 'D' ? 1 : 0;
    out[o + S.knownAttacker] = sides[seat] === 'A' ? 1 : 0;
    out[o + S.sideUnknown] = sides[seat] === '?' ? 1 : 0;
    out[o + S.knownPartner] = seat !== me && mySide !== '?' && sides[seat] === mySide ? 1 : 0;
    out[o + S.captured] = obs.capturedPoints[seat]! * pointsScale;
    out[o + S.tricksWon] = tricksWon[seat]! / totalTricks;
    out[o + S.handLeft] = obs.handSizes[seat]! / initialHand;
    if (realTrump) {
      for (const v of voids[seat]!) out[o + S.voids + (v === 'T' ? TRUMP_GROUP : Math.max(0, sideBlock(realTrump, v)))] = 1;
    }
    if (trick) {
      out[o + S.playedThisTrick] = trick.plays.some((p) => p.seat === seat) ? 1 : 0;
      out[o + S.winningTrick] = trick.winner === seat ? 1 : 0;
      out[o + S.leaderThisTrick] = trick.order[0] === seat ? 1 : 0;
      out[o + S.nextToPlay] = trick.order[trick.plays.length] === seat ? 1 : 0;
    }
    if (last) {
      out[o + S.ledLastTrick] = last.leader === seat ? 1 : 0;
      out[o + S.wonLastTrick] = last.winner === seat ? 1 : 0;
    }
    out[o + S.penalty] = obs.penalties[seat]! / 50;
    const b = lastBid.get(seat);
    if (b) {
      out[o + S.bidGroup + bidGroup(trump, b.card)] = 1;
      out[o + S.bidCount] = b.count * w;
    }
    out[o + S.holdsWinningBid] = winning && winning.seat === seat ? 1 : 0;
    out[o + S.bidDone] = obs.bidDone.includes(seat) ? 1 : 0;
    out[o + S.isDealer] = obs.dealer === seat ? 1 : 0;
  }

  // ---- global features
  if (n >= 2 && n <= MAX_SEATS) out[G.players + n - 2] = 1;
  out[G.decks] = decks / 4;
  out[G.mode + (obs.rules.gameMode === 'tractor' ? 0 : 1)] = 1;
  out[G.numFriends] = obs.rules.numFriends / 3;
  out[G.friendsHidden] = obs.friends.filter((f) => f.seat === null).length / 3;
  const st = STAGES.indexOf(obs.stage);
  if (st >= 0) out[G.stage + st] = 1;
  const kind = !obs.trump ? 3 : obs.trump.kind === 'standard' ? 0 : obs.trump.number !== null ? 1 : 2;
  out[G.trumpKind + kind] = 1;
  const number = trump.number;
  out[G.trumpNumber + (number === null ? 13 : levelIndex(number))] = 1;
  out[G.numberIsPoints] = number === '5' || number === 'T' || number === 'K' ? 1 : 0;
  out[G.maxRank + (obs.config.maxRank === 'NT' ? 1 : 0)] = 1;
  const landlordLevel: Level | null = obs.landlord !== null ? obs.levels[obs.landlord]! : null;
  out[G.landlordLevelsToMax] = landlordLevel ? (maxIdx - levelIndex(landlordLevel)) / 13 : 1;
  for (let seat = 0; seat < n; seat++) {
    if (obs.levels[seat] !== obs.config.maxRank) continue;
    if (sides[seat] === 'D') out[G.defendersAtMax] = 1;
    if (sides[seat] === 'A') out[G.attackersAtMax] = 1;
  }
  const thr = threshold(obs);
  let attackers = 0;
  for (let seat = 0; seat < n; seat++) if (sides[seat] !== 'D') attackers += obs.capturedPoints[seat]!;
  out[G.attackerPoints] = attackers / thr;
  out[G.threshold] = thr * pointsScale;
  out[G.unseenPoints] = unseenPoints * pointsScale;
  out[G.kittySize] = obs.kittySize / 12;
  if (obs.kitty) out[G.kittyPoints] = totalPoints(obs.kitty) / 50;
  out[G.trickProgress] = obs.tricks.length / totalTricks;
  out[G.myHandLeft] = obs.hand.length / initialHand;
  out[G.deckLeft] = obs.deckRemaining / Math.max(1, obs.deckRemaining + obs.handSizes.reduce((a, b) => a + b, 0));
  if (trick && realTrump) {
    const pos = trick.plays.length;
    out[G.leading] = pos === 0 ? 1 : 0;
    out[G.trickPosition] = pos / n;
    if (trick.format) {
      const led = trick.format.suit === 'T' ? TRUMP_GROUP : Math.max(0, sideBlock(realTrump, trick.format.suit));
      out[G.ledGroup + led] = 1;
      const k = unitKinds(trick.format.units);
      out[G.formatCards] = trick.format.units.reduce((a, u) => a + unitSize(u), 0) / 12;
      out[G.formatUnits + 0] = k.singles / 6;
      out[G.formatUnits + 1] = k.pairs / 6;
      out[G.formatUnits + 2] = k.triples / 6;
      out[G.formatUnits + 3] = k.tractors / 3;
      out[G.formatLongest] = k.longest / 12;
    }
    out[G.trickPoints] = totalPoints(trick.plays.flatMap((p) => p.cards)) / 50;
    if (trick.winner !== null) {
      const rel = sideOf(trick.winner);
      out[G.winnerIsMe] = rel === 0 ? 1 : 0;
      out[G.winnerIsPartner] = rel === 1 ? 1 : 0;
      out[G.winnerIsOpponent] = rel === 2 ? 1 : 0;
    }
    const after = trick.order.slice(pos + 1);
    out[G.toPlayAfterMe] = after.length / n;
    out[G.opponentsAfterMe] = after.filter((s) => sideOf(s) === 2).length / n;
    out[G.partnersAfterMe] = after.filter((s) => sideOf(s) === 1).length / n;
  }
  if (last) {
    out[G.lastTrickPoints] = last.points / 50;
    out[G.lastTrickWinner + sideOf(last.winner)] = 1;
  }
  if (winning) {
    out[G.hasBid] = 1;
    out[G.winningBidCount] = winning.count * w;
    out[G.winningBidGroup + bidGroup(trump, winning.card)] = 1;
    out[G.winningBidMine] = winning.seat === me ? 1 : 0;
  }
  out[G.landlordToBeMe] = obs.landlordToBe === me ? 1 : 0;
  out[G.passes] = obs.passes / n;
  out[G.revealed] = obs.revealedKitty.length / Math.max(1, obs.kittySize);
  out[G.role + 0] = obs.landlord === me && obs.stage !== 'draw' && obs.stage !== 'bid' ? 1 : 0;
  out[G.role + 1] = mySide === 'D' ? 1 : 0;
  out[G.role + 2] = mySide === 'A' ? 1 : 0;
  out[G.role + 3] = obs.friends.some((f) => f.seat === null && obs.hand.includes(f.card)) ? 1 : 0;
  out[G.drawPolicy + Math.max(0, DRAW_POLICIES.indexOf(obs.config.trickDrawPolicy))] = 1;
  out[G.throwEval + Math.max(0, THROW_EVALS.indexOf(obs.config.throwEvaluationPolicy))] = 1;
  out[G.kittyPower] = obs.config.kittyPenalty === 'power' ? 1 : 0;
  out[G.throwPenalty] = obs.config.throwPenalty === 'tenPointsPerAttempt' ? 1 : 0;
  return out;
}
