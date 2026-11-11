// The table: seats round an ellipse (the viewer at the bottom, play going
// clockwise), each seat's bid or trick cards toward the middle, and the
// deck / trick summary in the centre. Played cards spin in from their seat
// and flip face up (the viewer's own fly up from the hand); a finished trick
// flashes and sweeps off toward whoever won it, its points popping up there.
// On the live table cards are also dealt from the deck to each seat, and
// friends found / throws failing get a banner. `peek` shows the previous
// trick in place of the live one (which carries on underneath, hidden, so
// nothing replays when the peek ends).

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { totalPoints, type Card } from '@/game/cards';
import { activeSeat, bidText, drawingSeat } from '@/game/engine';
import type { PlayedCards } from '@/game/trick';
import type { TractorState } from '@/game/types';
import { CardRow, CardView } from '../CardView';
import { SuitText } from '../SuitText';
import { hasLaunch, prefersReducedMotion, takeLaunch } from './flight';
import { levelText, logLine, pointsHidden, roleOf } from './labels';

interface Props {
  state: TractorState;
  /** Seat drawn at the bottom (the viewer), or 0 for a spectator. */
  bottom: number;
  /** The live table (not the replay): deal flights, the viewer's cards flying from the hand, event banners. */
  live?: boolean;
  /** Show the previous trick instead of the current one. */
  peek?: boolean;
}

const BADGE_R: [number, number] = [44, 41];
const PLAY_R: [number, number] = [22, 19];
/** Bids sit a little further out than plays, clear of the deck in the middle. */
const BID_R: [number, number] = [30, 27];

function spot(k: number, n: number, [rx, ry]: [number, number]): { x: number; y: number } {
  const angle = Math.PI / 2 + (k * 2 * Math.PI) / n;
  return { x: 50 + rx * Math.cos(angle), y: 50 + ry * Math.sin(angle) };
}

/** Where seat k (0 = the viewer, at the bottom) sits; badges near the sides are anchored inward so they never clip. */
function badgePlace(k: number, n: number): { style: CSSProperties; side: boolean } {
  const { x, y } = spot(k, n, BADGE_R);
  if (x < 25) return { side: true, style: { left: `${Math.max(1, x - 6)}%`, top: `${y}%`, '--bx': '0%' } as CSSProperties };
  if (x > 75) return { side: true, style: { left: `${Math.min(99, x + 6)}%`, top: `${y}%`, '--bx': '-100%' } as CSSProperties };
  return { side: false, style: { left: `${x}%`, top: `${y}%` } };
}

/** Black or white initials, whichever reads on the seat colour. */
function inkOn(hex: string): string {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return '#fff';
  const [r, g, b] = [m[1], m[2], m[3]].map((h) => parseInt(h!, 16) / 255);
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b! > 0.6 ? '#1a1d24' : '#fff';
}

function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return '?';
  return words.length === 1 ? [...words[0]!].slice(0, 1).join('').toUpperCase() : (words[0]![0]! + words[1]![0]!).toUpperCase();
}

/** A small, stable tilt for a play, so the cards look tossed rather than placed. */
function tilt(seat: number, trick: number): number {
  return (((seat * 7 + trick * 3) % 7) - 3) * 1.4;
}

export function Felt({ state, bottom, live = false, peek = false }: Props) {
  const r = state.round;
  const n = state.seats.length;
  const active = activeSeat(state);
  const drawer = drawingSeat(state);
  const kOf = (seat: number) => (seat - bottom + n) % n;

  // The cards to show at each seat: this trick's plays, or the last trick's while the next is unled.
  let plays: PlayedCards[] = [];
  let showingLast = false;
  let trickWinner: number | null = null;
  let trickNo = r.tricks.length;
  if (r.stage === 'play' || r.stage === 'roundOver') {
    if (r.trick && r.trick.plays.length > 0) {
      plays = r.trick.plays;
      trickWinner = r.trick.winner;
    } else if (r.tricks.length > 0) {
      const last = r.tricks[r.tricks.length - 1]!;
      plays = last.plays;
      trickWinner = last.winner;
      showingLast = true;
      trickNo = r.tricks.length - 1;
    }
  }
  const playOf = new Map(plays.map((p) => [p.seat, p]));
  const lastBidOf = new Map<number, { card: Card; count: number }>();
  for (const b of r.bids) lastBidOf.set(b.seat, b);
  const winningBid = r.autobid ?? r.bids[r.bids.length - 1] ?? null;
  const showBids = r.stage === 'draw' || r.stage === 'bid';
  const lastTrick = r.tricks[r.tricks.length - 1];
  const winnerSpot = trickWinner !== null ? spot(kOf(trickWinner), n, BADGE_R) : null;
  const trickPoints = showingLast && lastTrick && !pointsHidden(state, lastTrick.winner) ? lastTrick.points : 0;
  const peeking = peek && !!lastTrick && (r.stage === 'play' || r.stage === 'roundOver');
  const lastPoints = lastTrick && !pointsHidden(state, lastTrick.winner) ? lastTrick.points : 0;

  return (
    <div className={`felt felt-n${Math.min(n, 9)}${peeking ? ' felt-peeking' : ''}`}>
      <div className="felt-watermark" aria-hidden>
        升级
      </div>
      {state.seats.map((seat, i) => {
        const k = kOf(i);
        const role = roleOf(state, i);
        const isActive = active === i || drawer === i;
        const play = playOf.get(i);
        const bid = showBids ? lastBidOf.get(i) : undefined;
        const friend = r.friends.find((f) => f.seat === i && i !== r.landlord);
        const from = spot(k, n, BADGE_R);
        const at = spot(k, n, PLAY_R);
        const bidAt = spot(k, n, BID_R);
        const points = totalPoints(r.captured[i]!);
        const place = badgePlace(k, n);
        const playStyle = {
          left: `${at.x}%`,
          top: `${at.y}%`,
          '--sl': `${from.x}%`,
          '--st': `${from.y}%`,
          '--tilt': `${tilt(i, trickNo)}deg`,
          ...(winnerSpot ? { '--wl': `${winnerSpot.x}%`, '--wt': `${winnerSpot.y}%` } : {}),
        } as CSSProperties;
        return (
          <div key={seat.id}>
            <div className={`seat-badge${isActive ? ' seat-active' : ''}${place.side ? ' seat-side' : ''} role-${role}`} style={place.style}>
              <span className="seat-avatar" style={{ background: seat.color, color: inkOn(seat.color) }} aria-hidden>
                {initials(seat.name)}
                {role === 'landlord' && (
                  <span className="seat-crown" title="Landlord">
                    👑
                  </span>
                )}
              </span>
              <span className="seat-badge-name">{seat.name}</span>
              <span className="seat-badge-level" title="Level">
                {levelText(state.levels[i]!)}
              </span>
              {friend && (
                <span className="seat-badge-icon" title="Revealed friend">
                  🤝
                </span>
              )}
              <span className="seat-badge-count" title={`${r.hands[i]!.length} cards in hand`}>
                <span className="mini-back" />
                {r.hands[i]!.length}
              </span>
              {r.stage === 'play' && points > 0 && !pointsHidden(state, i) && (
                <span key={points} className="seat-badge-points" title="Points captured">
                  {points}
                </span>
              )}
              {isActive && (
                <span className="seat-badge-turn">
                  <span className="seat-badge-turn-text">{drawer === i ? 'drawing' : seat.isAI ? 'thinking' : 'to play'}</span>
                  {seat.isAI && drawer !== i && (
                    <span className="dots" aria-hidden>
                      <i />
                      <i />
                      <i />
                    </span>
                  )}
                </span>
              )}
            </div>
            {play && (
              <SeatPlay
                key={`t${r.number}-${trickNo}-${i}`}
                seat={i}
                fly={live && i === bottom}
                className={`seat-play${showingLast ? ' seat-play-sweep' : ''}${trickWinner === i ? ' seat-play-winner' : ''}${peeking ? ' seat-play-hidden' : ''}`}
                style={playStyle}
                cards={play.cards}
                highlight={trickWinner === i}
                note={play.badThrowCards.length > 0 ? 'throw failed' : null}
              />
            )}
            {!play && bid && (
              <div key={`b${r.number}-${i}-${bid.card}x${bid.count}`} className="seat-play seat-bid-in" style={{ ...playStyle, left: `${bidAt.x}%`, top: `${bidAt.y}%` }}>
                <span className={`seat-bid${winningBid && winningBid.seat === i ? ' seat-bid-winning' : ''}`}>
                  <CardRow cards={Array<Card>(bid.count).fill(bid.card)} size="xs" />
                  <span className="seat-bid-text">
                    <SuitText text={bidText(bid)} />
                  </span>
                </span>
              </div>
            )}
          </div>
        );
      })}

      {trickPoints > 0 && winnerSpot && (
        <div key={`pp${r.number}-${trickNo}`} className={`points-pop${peeking ? ' is-hidden' : ''}`} style={{ left: `${winnerSpot.x}%`, top: `${winnerSpot.y}%` }}>
          +{trickPoints}
        </div>
      )}

      {peeking &&
        lastTrick!.plays.map((p) => {
          const at = spot(kOf(p.seat), n, PLAY_R);
          const won = lastTrick!.winner === p.seat;
          const style = { left: `${at.x}%`, top: `${at.y}%`, '--tilt': `${tilt(p.seat, r.tricks.length - 1)}deg` } as CSSProperties;
          return (
            <div key={`peek-${r.tricks.length}-${p.seat}`} className={`seat-play seat-play-peek${won ? ' seat-play-winner' : ''}`} style={style}>
              <CardRow cards={p.cards} size="sm" highlight={won} />
              {p.badThrowCards.length > 0 && <span className="seat-play-note">throw failed</span>}
            </div>
          );
        })}

      <div className="felt-center">
        {r.stage === 'draw' && (
          <div className="felt-deck">
            {r.deck.length > 0 ? (
              <>
                <span className="deck-stack">
                  {Array.from({ length: Math.max(1, Math.min(4, Math.ceil(r.deck.length / 10))) }, (_, j) => (
                    <CardView key={j} card={null} size="sm" style={{ '--j': j } as CSSProperties} />
                  ))}
                </span>
                <span className="felt-pill">{r.deck.length} to deal</span>
              </>
            ) : (
              <span className="felt-pill">Deal complete</span>
            )}
          </div>
        )}
        {r.stage === 'bid' && (
          <div className="felt-deck">
            <span className="felt-pill felt-pill-wait">Declaring trump</span>
          </div>
        )}
        {r.stage === 'exchange' && (
          <div className="felt-deck">
            <span className="felt-pill felt-pill-wait">{state.seats[r.landlord!]!.name} is burying the kitty</span>
          </div>
        )}
        {peeking && (
          <div className="felt-deck">
            <span className="felt-pill felt-pill-peek">
              Previous trick · {state.seats[lastTrick!.winner]!.name} won{lastPoints ? ` (+${lastPoints})` : ''}
            </span>
          </div>
        )}
        {!peeking && r.stage === 'play' && showingLast && lastTrick && (
          <div className="felt-deck felt-last">
            <span key={`w${r.number}-${trickNo}`} className="felt-pill felt-pill-won">
              {state.seats[lastTrick.winner]!.name} won the trick{trickPoints ? ` (+${trickPoints})` : ''}
            </span>
          </div>
        )}
        {(r.stage === 'draw' || r.stage === 'bid') && r.revealed > 0 && (
          <div className="felt-revealed">
            <span className="felt-pill">Kitty</span>
            <CardRow cards={r.dealtKitty.slice(0, r.revealed)} size="xs" />
          </div>
        )}
      </div>

      {live && <DealFlights state={state} bottom={bottom} />}
      {live && <EventBanner state={state} />}
    </div>
  );
}

/** One seat's cards in the current (or last) trick. Spins in from the seat, or — the viewer's own play — flies up from the hand. */
function SeatPlay({ seat, fly, className, style, cards, highlight, note }: {
  seat: number;
  fly: boolean;
  className: string;
  style: CSSProperties;
  cards: readonly Card[];
  highlight: boolean;
  note: string | null;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  // Decided once, at mount: did these cards just leave the viewer's hand?
  const [flown] = useState(() => fly && hasLaunch(seat));
  useLayoutEffect(() => {
    const el = ref.current;
    if (!flown || !el || prefersReducedMotion() || !el.animate) return;
    el.querySelectorAll<HTMLElement>('.card').forEach((card, i) => {
      const from = takeLaunch(seat, card.dataset.card ?? '');
      if (!from) return;
      const to = card.getBoundingClientRect();
      const dx = from.left + from.width / 2 - (to.left + to.width / 2);
      const dy = from.top + from.height / 2 - (to.top + to.height / 2);
      const s = to.width > 0 ? from.width / to.width : 1;
      card.animate([{ transform: `translate(${dx}px, ${dy}px) scale(${s}) rotate(${i % 2 ? 4 : -4}deg)` }, { transform: 'none' }], {
        duration: 440,
        delay: i * 35,
        easing: 'cubic-bezier(.22,.9,.3,1.06)',
        fill: 'backwards',
      });
    });
    // Mount only: a launch is taken once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <div ref={ref} className={`${className} ${flown ? 'seat-play-flown' : 'seat-play-in'}`} style={style}>
      <CardRow cards={cards} size="sm" highlight={highlight} />
      {note && <span className="seat-play-note">{note}</span>}
    </div>
  );
}

/** Card backs flying from the deck to whoever just drew (during the live deal). */
function DealFlights({ state, bottom }: { state: TractorState; bottom: number }) {
  const r = state.round;
  const n = state.seats.length;
  const lengths = r.hands.map((h) => h.length).join(',');
  const prev = useRef<{ round: number; lengths: number[] } | null>(null);
  const nextId = useRef(0);
  const [flights, setFlights] = useState<Array<{ id: number; k: number }>>([]);

  useEffect(() => {
    const now = r.hands.map((h) => h.length);
    const before = prev.current;
    prev.current = { round: r.number, lengths: now };
    if (!before || before.round !== r.number || r.stage !== 'draw' || prefersReducedMotion()) return;
    const drew = now.flatMap((len, i) => (len === before.lengths[i]! + 1 ? [i] : []));
    if (drew.length === 0 || drew.length > 3) return;
    setFlights((f) => [...f.slice(-10), ...drew.map((seat) => ({ id: ++nextId.current, k: (seat - bottom + n) % n }))]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lengths]);

  return (
    <>
      {flights.map((f) => {
        // The viewer's cards head down to the hand under the table.
        const to = f.k === 0 ? { x: 50, y: 108 } : spot(f.k, n, BADGE_R);
        const style = { '--tx': `${to.x}%`, '--ty': `${to.y}%`, '--spin': `${f.id % 2 ? 160 : -160}deg` } as CSSProperties;
        return (
          <span key={f.id} className="deal-flight" style={style} onAnimationEnd={() => setFlights((fs) => fs.filter((x) => x.id !== f.id))}>
            <CardView card={null} size="xs" />
          </span>
        );
      })}
    </>
  );
}

/** A banner for the big moments: a friend joining (or being called), a throw that failed. */
function EventBanner({ state }: { state: TractorState }) {
  const recent = state.log
    .slice(-5)
    .reverse()
    .find((e) => e.kind === 'friend' || e.kind === 'throw');
  if (!recent) return null;
  return (
    <div key={recent.n} className={`felt-banner felt-banner-${recent.kind}`} role="status">
      <span className="felt-banner-icon" aria-hidden>
        {recent.kind === 'friend' ? '🤝' : '💥'}
      </span>
      <SuitText text={logLine(state, recent)} />
    </div>
  );
}
