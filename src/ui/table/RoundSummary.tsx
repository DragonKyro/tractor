// End of a round (and of the match): the score, the kitty, everyone's level change.

import { levelLabel, totalPoints } from '@/game/cards';
import { materialize } from '@/game/scoring';
import { scoringParams } from '@/game/engine';
import type { TractorState } from '@/game/types';
import { CardRow } from '../CardView';
import { trumpParts } from './labels';

interface Props {
  state: TractorState;
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

export function RoundSummary({ state, onNextRound, onPlayAgain, onLobby, waitingText, onHide, onReview }: Props) {
  const res = state.round.result;
  if (!res) return null;
  const m = materialize(scoringParams(state.config), state.rules.numDecks, state.rules.numDecks * 100);
  const threshold = m.landlordWins[m.landlordWins.length - 1]!.end;
  const over = state.status === 'gameOver';
  const team = res.landlordTeam.map((i) => state.seats[i]!.name).join(', ');
  const trump = trumpParts(res.trump);
  const kittyBonus = res.kittyToAttackers ? res.kittyPoints * res.kittyMultiplier : 0;
  const fromTricks = res.attackerPoints - res.penaltyAdjustment - kittyBonus;
  return (
    <div className="summary-backdrop">
      <div className="summary-card" role="dialog" aria-label="Round summary">
        {over && <div className="summary-winners">🏆 {(state.winners ?? []).map((w) => state.seats[w]!.name).join(' & ')} win the game!</div>}
        <h3>Round {res.round}{res.endedEarly && <span className="online-tag summary-early">ended early</span>}</h3>
        <p className="summary-sub">
          Landlord {state.seats[res.landlord]!.name} · trump{' '}
          <span className={trump.red ? 'red' : ''}>
            {trump.symbol}
            {trump.number}
          </span>{' '}
          · defending: {team}
        </p>
        <div className="summary-score">
          <div className="summary-points">{res.attackerPoints}</div>
          <div>
            attacker points <span className="lobby-hint">(need {threshold} to take over)</span>
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
