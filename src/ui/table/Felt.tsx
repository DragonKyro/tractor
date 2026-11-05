// The table: seats round an ellipse (the viewer at the bottom, play going
// clockwise), each seat's bid or trick cards toward the middle, and the
// deck / trick summary in the centre. Played cards fly in from their seat;
// a finished trick sweeps off toward whoever won it.

import type { CSSProperties } from 'react';
import { totalPoints, type Card } from '@/game/cards';
import { activeSeat, bidText, drawingSeat } from '@/game/engine';
import type { PlayedCards } from '@/game/trick';
import type { TractorState } from '@/game/types';
import { CardRow, CardView } from '../CardView';
import { levelText, pointsHidden, roleOf } from './labels';

interface Props {
  state: TractorState;
  /** Seat drawn at the bottom (the viewer), or 0 for a spectator. */
  bottom: number;
}

const BADGE_R: [number, number] = [44, 41];
const PLAY_R: [number, number] = [22, 19];

function spot(k: number, n: number, [rx, ry]: [number, number]): { x: number; y: number } {
  const angle = Math.PI / 2 + (k * 2 * Math.PI) / n;
  return { x: 50 + rx * Math.cos(angle), y: 50 + ry * Math.sin(angle) };
}

/** Where seat k (0 = the viewer, at the bottom) sits; badges near the sides are anchored inward so they never clip. */
function badgeStyle(k: number, n: number): CSSProperties {
  const { x, y } = spot(k, n, BADGE_R);
  if (x < 25) return { left: `${Math.max(1, x - 6)}%`, top: `${y}%`, transform: 'translate(0, -50%)' };
  if (x > 75) return { left: `${Math.min(99, x + 6)}%`, top: `${y}%`, transform: 'translate(-100%, -50%)' };
  return { left: `${x}%`, top: `${y}%` };
}

export function Felt({ state, bottom }: Props) {
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

  return (
    <div className={`felt felt-n${Math.min(n, 9)}`}>
      {state.seats.map((seat, i) => {
        const k = kOf(i);
        const role = roleOf(state, i);
        const isActive = active === i || drawer === i;
        const play = playOf.get(i);
        const bid = showBids ? lastBidOf.get(i) : undefined;
        const friend = r.friends.find((f) => f.seat === i && i !== r.landlord);
        const from = spot(k, n, BADGE_R);
        const at = spot(k, n, PLAY_R);
        const playStyle = {
          left: `${at.x}%`,
          top: `${at.y}%`,
          '--sl': `${from.x}%`,
          '--st': `${from.y}%`,
          ...(winnerSpot ? { '--wl': `${winnerSpot.x}%`, '--wt': `${winnerSpot.y}%` } : {}),
        } as CSSProperties;
        return (
          <div key={seat.id}>
            <div className={`seat-badge${isActive ? ' seat-active' : ''} role-${role}`} style={badgeStyle(k, n)}>
              <span className="seat-dot" style={{ background: seat.color }} />
              <span className="seat-badge-name">{seat.name}</span>
              <span className="seat-badge-level" title="Level">{levelText(state.levels[i]!)}</span>
              {role === 'landlord' && <span className="seat-badge-icon" title="Landlord">👑</span>}
              {friend && <span className="seat-badge-icon" title="Revealed friend">🤝</span>}
              <span className="seat-badge-count" title={`${r.hands[i]!.length} cards in hand`}>
                <span className="mini-back" />
                {r.hands[i]!.length}
              </span>
              {r.stage === 'play' && totalPoints(r.captured[i]!) > 0 && !pointsHidden(state, i) && (
                <span className="seat-badge-points" title="Points captured">{totalPoints(r.captured[i]!)}</span>
              )}
              {isActive && <span className="seat-badge-turn">{drawer === i ? 'drawing' : seat.isAI ? 'thinking…' : 'to play'}</span>}
            </div>
            {play && (
              <div
                key={`t${r.number}-${trickNo}-${i}`}
                className={`seat-play seat-play-in${showingLast ? ' seat-play-sweep' : ''}${trickWinner === i ? ' seat-play-winner' : ''}`}
                style={playStyle}
              >
                <CardRow cards={play.cards} size="sm" highlight={trickWinner === i} />
                {play.badThrowCards.length > 0 && <span className="seat-play-note">throw failed</span>}
              </div>
            )}
            {!play && bid && (
              <div key={`b${r.number}-${i}-${bid.card}x${bid.count}`} className="seat-play seat-play-in" style={playStyle}>
                <span className={`seat-bid${winningBid && winningBid.seat === i ? ' seat-bid-winning' : ''}`}>
                  <CardRow cards={Array<Card>(bid.count).fill(bid.card)} size="xs" />
                  <span className="seat-bid-text">{bidText(bid)}</span>
                </span>
              </div>
            )}
          </div>
        );
      })}

      <div className="felt-center">
        {r.stage === 'draw' && (
          <div className="felt-deck">
            {r.deck.length > 0 ? (
              <>
                <CardView card={null} size="sm" />
                <span>{r.deck.length} to deal</span>
              </>
            ) : (
              <span>Deal complete</span>
            )}
          </div>
        )}
        {r.stage === 'bid' && (
          <div className="felt-deck">
            <span>Declaring trump</span>
          </div>
        )}
        {r.stage === 'exchange' && (
          <div className="felt-deck">
            <span>{state.seats[r.landlord!]!.name} is burying the kitty</span>
          </div>
        )}
        {r.stage === 'play' && showingLast && lastTrick && (
          <div className="felt-deck felt-last">
            <span>
              {state.seats[lastTrick.winner]!.name} won the trick{lastTrick.points && !pointsHidden(state, lastTrick.winner) ? ` (+${lastTrick.points})` : ''}
            </span>
          </div>
        )}
        {(r.stage === 'draw' || r.stage === 'bid') && r.revealed > 0 && (
          <div className="felt-revealed">
            <span>Kitty:</span>
            <CardRow cards={r.dealtKitty.slice(0, r.revealed)} size="xs" />
          </div>
        )}
      </div>
    </div>
  );
}
