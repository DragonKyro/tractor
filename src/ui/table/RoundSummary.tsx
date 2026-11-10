// End of a round (and of the match): the score (counting up), the kitty,
// everyone's level change — with confetti when the viewer's side wins.

import { useEffect, useState, type CSSProperties } from 'react';
import { levelLabel, totalPoints } from '@/game/cards';
import { materialize } from '@/game/scoring';
import { scoringParams } from '@/game/engine';
import type { TractorState } from '@/game/types';
import { CardRow } from '../CardView';
import { prefersReducedMotion } from './flight';
import { trumpParts } from './labels';

interface Props {
  state: TractorState;
  /** The seat this screen belongs to (null: spectator / pass screen), for the "you won" touches. */
  viewer?: number | null;
  /** Set when this device may deal the next round. */
  onNextRound?: () => void;
  /** Game over: start a new match / back to the lobby. */
  onPlayAgain?: () => void;
  onLobby?: () => void;
  waitingText?: string;
  /** Hide the summary to look at the table (it comes back with the button). */
  onHide?: () => void;
  onReview?: (kind: 'stats' | 'replay') => void;
}

/** Counts from 0 up to `target` over about a second. */
function useCountUp(target: number): number {
  const [value, setValue] = useState(() => (prefersReducedMotion() ? target : 0));
  useEffect(() => {
    if (prefersReducedMotion() || typeof requestAnimationFrame === 'undefined') {
      setValue(target);
      return;
    }
    const start = performance.now();
    const ms = 900;
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      setValue(Math.round(target * (1 - (1 - t) ** 3)));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return value;
}

const CONFETTI_COLORS = ['#ffd25e', '#ff5e6c', '#4f9bff', '#3ccf7a', '#c08bff', '#ffffff'];

function Confetti() {
  // Random only in the UI: each piece's drift, spin, delay and colour.
  const [pieces] = useState(() =>
    Array.from({ length: 70 }, (_, i) => ({
      left: Math.random() * 100,
      delay: Math.random() * 0.6,
      duration: 2.2 + Math.random() * 1.6,
      drift: (Math.random() - 0.5) * 220,
      spin: (Math.random() < 0.5 ? -1 : 1) * (360 + Math.random() * 720),
      color: CONFETTI_COLORS[i % CONFETTI_COLORS.length]!,
      wide: Math.random() < 0.5,
    })),
  );
  if (prefersReducedMotion()) return null;
  return (
    <div className="confetti" aria-hidden>
      {pieces.map((p, i) => (
        <i
          key={i}
          className={p.wide ? 'confetti-wide' : undefined}
          style={
            {
              left: `${p.left}%`,
              background: p.color,
              animationDelay: `${p.delay}s`,
              animationDuration: `${p.duration}s`,
              '--drift': `${p.drift}px`,
              '--spin': `${p.spin}deg`,
            } as CSSProperties
          }
        />
      ))}
    </div>
  );
}

export function RoundSummary({ state, viewer = null, onNextRound, onPlayAgain, onLobby, waitingText, onHide, onReview }: Props) {
  const res = state.round.result;
  const shown = useCountUp(res?.attackerPoints ?? 0);
  if (!res) return null;
  const m = materialize(scoringParams(state.config), state.rules.numDecks, state.rules.numDecks * 100);
  const threshold = m.landlordWins[m.landlordWins.length - 1]!.end;
  const over = state.status === 'gameOver';
  const team = res.landlordTeam.map((i) => state.seats[i]!.name).join(', ');
  const trump = trumpParts(res.trump);
  const kittyBonus = res.kittyToAttackers ? res.kittyPoints * res.kittyMultiplier : 0;
  const fromTricks = res.attackerPoints - res.penaltyAdjustment - kittyBonus;
  const viewerDefended = viewer !== null && res.landlordTeam.includes(viewer);
  const viewerWon = viewer !== null && (viewerDefended ? res.landlordWon : !res.landlordWon);
  const viewerWonGame = viewer !== null && (state.winners ?? []).includes(viewer);
  const celebrate = over ? viewer === null || viewerWonGame : viewerWon;
  const headline = res.landlordWon ? 'The landlord holds' : 'The attackers take over';
  const meter = Math.min(100, (res.attackerPoints / Math.max(1, threshold)) * 100);
  return (
    <div className="summary-backdrop">
      {celebrate && <Confetti />}
      <div className={`summary-card ${res.landlordWon ? 'summary-defend' : 'summary-attack'}`} role="dialog" aria-label="Round summary">
        {over && <div className="summary-winners">🏆 {(state.winners ?? []).map((w) => state.seats[w]!.name).join(' & ')} win the game!</div>}
        <div className="summary-eyebrow">
          Round {res.round}
          {res.endedEarly && <span className="online-tag summary-early">ended early</span>}
        </div>
        <h3 className="summary-headline">
          {headline}
          {viewer !== null && <span className={`summary-you ${viewerWon ? 'summary-you-won' : 'summary-you-lost'}`}>{viewerWon ? 'you win the round' : 'you lose the round'}</span>}
        </h3>
        <p className="summary-sub">
          Landlord {state.seats[res.landlord]!.name} · trump{' '}
          <span>
            <span className={trump.suitClass}>{trump.symbol}</span>
            {trump.number}
          </span>{' '}
          · defending: {team}
        </p>
        <div className="summary-score">
          <div className="summary-points">{shown}</div>
          <div className="summary-score-text">
            attacker points <span className="lobby-hint">(need {threshold} to take over)</span>
            <span className={`points-meter points-meter-lg${res.attackerPoints >= threshold ? ' points-meter-over' : ''}`} aria-hidden>
              <span style={{ width: `${meter}%` }} />
            </span>
            <div className="summary-breakdown">
              {fromTricks} from tricks
              {kittyBonus > 0 && ` + kitty ${res.kittyPoints} × ${res.kittyMultiplier}`}
              {res.penaltyAdjustment !== 0 && ` ${res.penaltyAdjustment > 0 ? '+' : '−'} ${Math.abs(res.penaltyAdjustment)} penalties`}
            </div>
          </div>
        </div>
        <p className="summary-result">
          {res.landlordWon
            ? res.landlordDelta > 0
              ? `The landlord's team holds and goes up ${res.landlordDelta} level${res.landlordDelta === 1 ? '' : 's'}${res.bonusLevel ? ' (including a short-handed bonus)' : ''}.`
              : "The landlord's team holds."
            : res.attackerDelta > 0
              ? `The attackers take over and go up ${res.attackerDelta} level${res.attackerDelta === 1 ? '' : 's'}.`
              : 'The attackers take over.'}
        </p>
        {state.config.revealKittyAtEnd && (
          <div className="summary-kitty">
            <span>
              Kitty ({totalPoints(res.kitty)} pts){res.kittyToAttackers && res.kittyPoints > 0 ? ' — won by the attackers' : ''}:
            </span>
            <CardRow cards={res.kitty} size="xs" />
          </div>
        )}
        <table className="players-table summary-levels">
          <tbody>
            {state.seats.map((s, i) => {
              const a = res.advancements[i]!;
              return (
                <tr key={s.id} className={res.landlordTeam.includes(i) ? 'role-defender' : 'role-attacker'}>
                  <td>
                    <span className="seat-dot" style={{ background: s.color }} /> {s.name}
                  </td>
                  <td>
                    {levelLabel(a.from)}
                    {a.steps > 0 && !a.finished && ` → ${levelLabel(a.to)}`}
                    {a.finished && ' → 🏆'}
                  </td>
                  <td className="lobby-hint">{a.blocked ? `must defend ${levelLabel(a.to)}` : a.steps > 0 ? `+${a.steps}` : ''}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!over && <p className="summary-sub">Next landlord: {state.seats[res.nextLandlord]!.name}</p>}
        <div className="action-buttons summary-buttons">
          {!over && onNextRound && <button onClick={onNextRound}>Next round →</button>}
          {!over && !onNextRound && <span className="lobby-hint">{waitingText ?? 'Waiting for the next deal…'}</span>}
          {over && onPlayAgain && <button onClick={onPlayAgain}>🔁 Play again</button>}
          {over && onLobby && <button className="secondary" onClick={onLobby}>Lobby</button>}
          {over && !onPlayAgain && <span className="lobby-hint">{waitingText ?? 'Waiting for the host…'}</span>}
          {onReview && <button className="secondary" onClick={() => onReview('stats')}>📊 Stats</button>}
          {onReview && <button className="secondary" onClick={() => onReview('replay')}>⏪ Replay</button>}
          {onHide && <button className="secondary" onClick={onHide}>See the table</button>}
        </div>
      </div>
    </div>
  );
}
