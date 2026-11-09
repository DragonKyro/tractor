// FROZEN COPY of src/ai/nn (encoder v2, action space v1, tractor/cand-mlp@1) for benchmarking
// v2 policies (training/benchmarks/) against newer networks. Do not edit: a v2
// policy only plays as trained if this code stays exactly as it was.

// Tractor learned agent — the observation encoder.
//
// One flat float vector per decision, from a seat's Observation only (no
// hidden cards). Seats are relative: seat 0 is the decider, 1 the next to
// play after them, and so on, padded to MAX_SEATS (12, the game's largest
// table), so the same network plays every table size. Cards are trump-relative slots (slots.ts),
// so the same network plays every trump suit and number.
//
//   card planes   20 × 72  counts / numDecks:
//                   hand · unseen (not in hand, not played, not buried by me,
//                   not removed) · played by each seat (12) · this trick so
//                   far · this trick's winning play · kitty as dealt / as
//                   buried (landlord only) · unresolved friend calls ·
//                   copies still to come before a call's friend copy
//   seats         12 × SEAT_DIM: level, side (landlord / known defender /
//                   known attacker / unknown / my partner), captured points,
//                   cards left, voids per suit group, this trick, bids, …
//   global        table size, decks, mode, stage, trump kind and number,
//                   final rank and who must defend it, points vs threshold,
//                   unseen points, the trick's format and who's winning it,
//                   the bidding, my role, the rules that change play
//
// ENCODER_VERSION goes in every policy file; a policy trained on another
// version is refused (bump it with any change to what a slot means).

import { LEVELS, levelIndex, totalPoints, type Card, type Level } from '@/game/cards';
import { MAX_PLAYERS } from '@/game/config';
import { scoringParams } from '@/game/engine';
import { materialize } from '@/game/scoring';
import { unitSize, type TrickUnit } from '@/game/format';
import { effectiveSuit, type Trump } from '@/game/trump';
import { knownVoids, unseenCounts, type Observation } from '@/ai/observation';
import { CARD_SLOTS, GROUPS, TRUMP_GROUP, addCards, sideBlock, slotOf } from './slots';

export const ENCODER_VERSION = 2;
/** Every table the game allows (v1 had 8). */
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

// ---------------------------------------------------------------- seats

const SEAT = new Layout();
const S = {
  present: SEAT.add('present', 1),
  isMe: SEAT.add('isMe', 1),
  level: SEAT.add('level', LEVELS.length),
  levelVsMine: SEAT.add('levelVsMine', 1),
  levelsToMax: SEAT.add('levelsToMax', 1),
  isLandlord: SEAT.add('isLandlord', 1),
  knownDefender: SEAT.add('knownDefender', 1),
  knownAttacker: SEAT.add('knownAttacker', 1),
  sideUnknown: SEAT.add('sideUnknown', 1),
  knownPartner: SEAT.add('knownPartner', 1),
  captured: SEAT.add('captured', 1),
  handLeft: SEAT.add('handLeft', 1),
  voids: SEAT.add('voids', GROUPS),
  playedThisTrick: SEAT.add('playedThisTrick', 1),
  winningTrick: SEAT.add('winningTrick', 1),
  nextToPlay: SEAT.add('nextToPlay', 1),
  penalty: SEAT.add('penalty', 1),
  bidGroup: SEAT.add('bidGroup', 6),
  bidCount: SEAT.add('bidCount', 1),
  holdsWinningBid: SEAT.add('holdsWinningBid', 1),
  bidDone: SEAT.add('bidDone', 1),
  isDealer: SEAT.add('isDealer', 1),
};
export const SEAT_DIM = SEAT.dim;
export const SEAT_LAYOUT = SEAT;

// ---------------------------------------------------------------- global

const GLOBAL = new Layout();
const G = {
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
  toPlayAfterMe: GLOBAL.add('toPlayAfterMe', 1),
  opponentsAfterMe: GLOBAL.add('opponentsAfterMe', 1),
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

export const PLANES = [
  'hand',
  'unseen',
  ...Array.from({ length: MAX_SEATS }, (_, i) => `played${i}`),
  'trickAll',
  'trickWinner',
  'kittyDealt',
  'kittyBuried',
  'calls',
  'callsBefore',
] as const;

export const OBS_LAYOUT = new Layout();
const PLANE_OFFSET = Object.fromEntries(PLANES.map((p) => [p, OBS_LAYOUT.add(p, CARD_SLOTS)])) as Record<(typeof PLANES)[number], number>;
const SEATS_OFFSET = OBS_LAYOUT.add('seats', MAX_SEATS * SEAT_DIM);
const GLOBAL_OFFSET = OBS_LAYOUT.add('global', GLOBAL_DIM);
export const OBS_DIM = OBS_LAYOUT.dim;

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

export function encodeObservation(obs: Observation, out: Float32Array = new Float32Array(OBS_DIM)): Float32Array {
  out.fill(0);
  const n = obs.numPlayers;
  const me = obs.seat;
  const rel = (seat: number) => (seat - me + n) % n;
  const decks = obs.rules.numDecks;
  const w = 1 / decks;
  const trump = slotTrump(obs);
  const pointsScale = 1 / (100 * decks);

  // ---- card planes
  addCards(out, PLANE_OFFSET.hand, trump, obs.hand, w);
  const unseen = unseenCounts(obs);
  let unseenPoints = 0;
  for (const [c, k] of unseen) {
    if (k <= 0) continue;
    out[PLANE_OFFSET.unseen + slotOf(trump, c)]! += k * w;
    unseenPoints += totalPoints([c]) * k;
  }
  const playedBy = (seat: number, cards: readonly Card[]) => {
    const r = rel(seat);
    if (r < MAX_SEATS) addCards(out, PLANE_OFFSET[`played${r}` as 'played0'], trump, cards, w);
  };
  for (const t of obs.tricks) for (const p of t.plays) playedBy(p.seat, p.cards);
  const trick = obs.trick;
  if (trick) {
    for (const p of trick.plays) {
      playedBy(p.seat, p.cards);
      addCards(out, PLANE_OFFSET.trickAll, trump, p.cards, w);
    }
    const win = trick.plays.find((p) => p.seat === trick.winner);
    if (win) addCards(out, PLANE_OFFSET.trickWinner, trump, win.cards, w);
  }
  if (obs.dealtKitty) addCards(out, PLANE_OFFSET.kittyDealt, trump, obs.dealtKitty, w);
  if (obs.kitty) addCards(out, PLANE_OFFSET.kittyBuried, trump, obs.kitty, w);
  for (const f of obs.friends) {
    if (f.seat !== null) continue;
    const s = slotOf(trump, f.card);
    out[PLANE_OFFSET.calls + s]! += 1 / Math.max(1, obs.rules.numFriends);
    out[PLANE_OFFSET.callsBefore + s] = Math.max(out[PLANE_OFFSET.callsBefore + s]!, f.skip * w);
  }

  // ---- seats
  const sides = knownSides(obs);
  const mySide = sides[me]!;
  const voids = knownVoids(obs);
  const maxIdx = levelIndex(obs.config.maxRank);
  const winning = obs.autobid ?? obs.bids[obs.bids.length - 1] ?? null;
  const lastBid = new Map<number, { card: Card; count: number }>();
  for (const b of obs.bids) lastBid.set(b.seat, b);
  const initialHand = Math.max(1, obs.hand.length + obs.tricks.reduce((k, t) => k + (t.plays.find((p) => p.seat === me)?.cards.length ?? 0), 0));
  for (let seat = 0; seat < n; seat++) {
    const r = rel(seat);
    if (r >= MAX_SEATS) continue;
    const o = SEATS_OFFSET + r * SEAT_DIM;
    const lvl = obs.levels[seat]!;
    out[o + S.present] = 1;
    out[o + S.isMe] = seat === me ? 1 : 0;
    out[o + S.level + levelIndex(lvl)] = 1;
    out[o + S.levelVsMine] = (levelIndex(lvl) - levelIndex(obs.levels[me]!)) / 12;
    out[o + S.levelsToMax] = (maxIdx - levelIndex(lvl)) / 13;
    out[o + S.isLandlord] = obs.landlord === seat ? 1 : 0;
    out[o + S.knownDefender] = sides[seat] === 'D' ? 1 : 0;
    out[o + S.knownAttacker] = sides[seat] === 'A' ? 1 : 0;
    out[o + S.sideUnknown] = sides[seat] === '?' ? 1 : 0;
    out[o + S.knownPartner] = seat !== me && mySide !== '?' && sides[seat] === mySide ? 1 : 0;
    out[o + S.captured] = obs.capturedPoints[seat]! * pointsScale;
    out[o + S.handLeft] = obs.handSizes[seat]! / initialHand;
    if (obs.trump) {
      for (const v of voids[seat]!) out[o + S.voids + (v === 'T' ? TRUMP_GROUP : Math.max(0, sideBlock(obs.trump, v)))] = 1;
    }
    if (trick) {
      out[o + S.playedThisTrick] = trick.plays.some((p) => p.seat === seat) ? 1 : 0;
      out[o + S.winningTrick] = trick.winner === seat ? 1 : 0;
      out[o + S.nextToPlay] = trick.order[trick.plays.length] === seat ? 1 : 0;
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

  // ---- global
  const g = GLOBAL_OFFSET;
  if (n >= 2 && n <= MAX_SEATS) out[g + G.players + n - 2] = 1;
  out[g + G.decks] = decks / 4;
  out[g + G.mode + (obs.rules.gameMode === 'tractor' ? 0 : 1)] = 1;
  out[g + G.numFriends] = obs.rules.numFriends / 3;
  out[g + G.friendsHidden] = obs.friends.filter((f) => f.seat === null).length / 3;
  const st = STAGES.indexOf(obs.stage);
  if (st >= 0) out[g + G.stage + st] = 1;
  const kind = !obs.trump ? 3 : obs.trump.kind === 'standard' ? 0 : obs.trump.number !== null ? 1 : 2;
  out[g + G.trumpKind + kind] = 1;
  const number = trump.number;
  out[g + G.trumpNumber + (number === null ? 13 : levelIndex(number))] = 1;
  out[g + G.numberIsPoints] = number === '5' || number === 'T' || number === 'K' ? 1 : 0;
  out[g + G.maxRank + (obs.config.maxRank === 'NT' ? 1 : 0)] = 1;
  const landlordLevel: Level | null = obs.landlord !== null ? obs.levels[obs.landlord]! : null;
  out[g + G.landlordLevelsToMax] = landlordLevel ? (maxIdx - levelIndex(landlordLevel)) / 13 : 1;
  for (let seat = 0; seat < n; seat++) {
    if (obs.levels[seat] !== obs.config.maxRank) continue;
    if (sides[seat] === 'D') out[g + G.defendersAtMax] = 1;
    if (sides[seat] === 'A') out[g + G.attackersAtMax] = 1;
  }
  const thr = threshold(obs);
  let attackers = 0;
  for (let seat = 0; seat < n; seat++) if (sides[seat] !== 'D') attackers += obs.capturedPoints[seat]!;
  out[g + G.attackerPoints] = attackers / thr;
  out[g + G.threshold] = thr * pointsScale;
  out[g + G.unseenPoints] = unseenPoints * pointsScale;
  out[g + G.kittySize] = obs.kittySize / 12;
  if (obs.kitty) out[g + G.kittyPoints] = totalPoints(obs.kitty) / 50;
  const totalTricks = obs.tricks.length + Math.max(1, obs.hand.length);
  out[g + G.trickProgress] = obs.tricks.length / totalTricks;
  out[g + G.myHandLeft] = obs.hand.length / initialHand;
  out[g + G.deckLeft] = obs.deckRemaining / Math.max(1, obs.deckRemaining + obs.handSizes.reduce((a, b) => a + b, 0));
  if (trick && obs.trump) {
    const pos = trick.plays.length;
    out[g + G.leading] = pos === 0 ? 1 : 0;
    out[g + G.trickPosition] = pos / n;
    if (trick.format) {
      const led = trick.format.suit === 'T' ? TRUMP_GROUP : Math.max(0, sideBlock(obs.trump, trick.format.suit));
      out[g + G.ledGroup + led] = 1;
      const k = unitKinds(trick.format.units);
      out[g + G.formatCards] = trick.format.units.reduce((a, u) => a + unitSize(u), 0) / 12;
      out[g + G.formatUnits + 0] = k.singles / 6;
      out[g + G.formatUnits + 1] = k.pairs / 6;
      out[g + G.formatUnits + 2] = k.triples / 6;
      out[g + G.formatUnits + 3] = k.tractors / 3;
      out[g + G.formatLongest] = k.longest / 12;
    }
    out[g + G.trickPoints] = totalPoints(trick.plays.flatMap((p) => p.cards)) / 50;
    if (trick.winner !== null) {
      out[g + G.winnerIsMe] = trick.winner === me ? 1 : 0;
      out[g + G.winnerIsPartner] = trick.winner !== me && mySide !== '?' && sides[trick.winner] === mySide ? 1 : 0;
    }
    const after = trick.order.slice(pos + 1);
    out[g + G.toPlayAfterMe] = after.length / n;
    out[g + G.opponentsAfterMe] = after.filter((s) => mySide !== '?' && sides[s] !== '?' && sides[s] !== mySide).length / n;
  }
  if (winning) {
    out[g + G.hasBid] = 1;
    out[g + G.winningBidCount] = winning.count * w;
    out[g + G.winningBidGroup + bidGroup(trump, winning.card)] = 1;
    out[g + G.winningBidMine] = winning.seat === me ? 1 : 0;
  }
  out[g + G.landlordToBeMe] = obs.landlordToBe === me ? 1 : 0;
  out[g + G.passes] = obs.passes / n;
  out[g + G.revealed] = obs.revealedKitty.length / Math.max(1, obs.kittySize);
  out[g + G.role + 0] = obs.landlord === me && obs.stage !== 'draw' && obs.stage !== 'bid' ? 1 : 0;
  out[g + G.role + 1] = mySide === 'D' ? 1 : 0;
  out[g + G.role + 2] = mySide === 'A' ? 1 : 0;
  out[g + G.role + 3] = obs.friends.some((f) => f.seat === null && obs.hand.includes(f.card)) ? 1 : 0;
  out[g + G.drawPolicy + Math.max(0, DRAW_POLICIES.indexOf(obs.config.trickDrawPolicy))] = 1;
  out[g + G.throwEval + Math.max(0, THROW_EVALS.indexOf(obs.config.throwEvaluationPolicy))] = 1;
  out[g + G.kittyPower] = obs.config.kittyPenalty === 'power' ? 1 : 0;
  out[g + G.throwPenalty] = obs.config.throwPenalty === 'tenPointsPerAttempt' ? 1 : 0;
  return out;
}

/** Group (side block or trump) of a card under the slot trump. */
export function groupOf(trump: Trump, card: Card): number {
  if (effectiveSuit(trump, card) === 'T') return TRUMP_GROUP;
  return Math.max(0, sideBlock(trump, card[1] as never));
}
