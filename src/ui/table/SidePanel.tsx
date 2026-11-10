// Beside the table: who's who (levels, teams, points), friend calls, the
// game log, and this device's display settings.

import { useEffect, useRef, useState } from 'react';
import { cardLabel, totalPoints } from '@/game/cards';
import { ordinalText } from '@/game/engine';
import type { TractorState } from '@/game/types';
import { usePrefs, type AiSpeed, type BidWindow, type DealSpeed } from '@/store/prefsStore';
import { levelText, logLine, pointsHidden, roleOf } from './labels';

type Tab = 'players' | 'log' | 'settings';

export function SidePanel({ state, canShowAllHands = false, onClose }: { state: TractorState; canShowAllHands?: boolean; onClose?: () => void }) {
  const [tab, setTab] = useState<Tab>('players');
  return (
    <aside className="side-panel">
      <div className="side-tabs" role="tablist">
        {onClose && (
          <button className="side-tab side-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        )}
        {(['players', 'log', 'settings'] as Tab[]).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'side-tab active' : 'side-tab'} onClick={() => setTab(t)}>
            {t === 'players' ? 'Table' : t === 'log' ? 'Log' : 'Settings'}
          </button>
        ))}
      </div>
      {tab === 'players' && <Players state={state} />}
      {tab === 'log' && <Log state={state} />}
      {tab === 'settings' && <Settings canShowAllHands={canShowAllHands} />}
    </aside>
  );
}

const ROLE_LABEL = { landlord: 'Landlord', defender: 'Defending', attacker: 'Attacking', unknown: '' } as const;

function Players({ state }: { state: TractorState }) {
  const r = state.round;
  return (
    <div className="side-body">
      <table className="players-table">
        <thead>
          <tr>
            <th>Player</th>
            <th title="Level">Lvl</th>
            <th>Role</th>
            <th title="Points captured this round">Pts</th>
          </tr>
        </thead>
        <tbody>
          {state.seats.map((s, i) => {
            const role = roleOf(state, i);
            return (
              <tr key={s.id} className={`role-${role}`}>
                <td>
                  <span className="seat-dot" style={{ background: s.color }} /> {s.name}
                  {s.isAI && <span className="online-tag">AI</span>}
                </td>
                <td>{levelText(state.levels[i]!)}</td>
                <td>{ROLE_LABEL[role]}</td>
                <td title={pointsHidden(state, i) ? 'Hidden until the round ends' : undefined}>
                  {pointsHidden(state, i) ? '—' : totalPoints(r.captured[i]!) || ''}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {r.friends.length > 0 && (
        <div className="friend-status">
          <h5>Friend calls</h5>
          <ul>
            {r.friends.map((f, i) => (
              <li key={i}>
                The {ordinalText(f.ordinal)} {cardLabel(f.card)} —{' '}
                {f.seat !== null
                  ? f.seat === r.landlord
                    ? 'played by the landlord (no friend)'
                    : `${state.seats[f.seat]!.name} joined`
                  : f.skip < f.ordinal - 1
                    ? `${f.ordinal - 1 - f.skip} of ${f.ordinal - 1} earlier ${f.ordinal - 1 === 1 ? 'copy' : 'copies'} played`
                    : 'not played yet'}
              </li>
            ))}
          </ul>
        </div>
      )}
      {state.history.length > 0 && (
        <div className="history">
          <h5>Past rounds</h5>
          <ol>
            {state.history.map((h) => (
              <li key={h.round}>
                {state.seats[h.landlord]!.name}: attackers {h.attackerPoints} —{' '}
                {h.landlordWon ? `defenders +${h.landlordDelta}` : h.attackerDelta ? `attackers +${h.attackerDelta}` : 'attackers take over'}
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

function Log({ state }: { state: TractorState }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const last = state.log[state.log.length - 1]?.n;
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [last]);
  return (
    <div className="side-body log" ref={ref}>
      {state.log.slice(-120).map((e) => (
        <div key={e.n} className={`log-row log-${e.kind}`}>
          {e.seat !== undefined && <span className="seat-dot" style={{ background: state.seats[e.seat]?.color }} />}
          {logLine(state, e)}
        </div>
      ))}
    </div>
  );
}

function Settings({ canShowAllHands }: { canShowAllHands: boolean }) {
  const p = usePrefs();
  return (
    <div className="side-body settings">
      <label className="opt-inline">
        AI speed
        <select value={p.aiSpeed} onChange={(e) => p.set({ aiSpeed: e.target.value as AiSpeed })}>
          <option value="slow">Slow</option>
          <option value="normal">Normal</option>
          <option value="fast">Fast</option>
          <option value="instant">Instant</option>
        </select>
      </label>
      <label className="opt-inline">
        Deal speed
        <select value={p.dealSpeed} onChange={(e) => p.set({ dealSpeed: e.target.value as DealSpeed })}>
          <option value="slow">Slow</option>
          <option value="normal">Normal</option>
          <option value="fast">Fast</option>
        </select>
      </label>
      <label className="opt-inline" title="After the deal, how long an AI waits before taking the kitty, so you can still overcall">
        Time to overcall after the deal
        <select value={p.bidWindow} onChange={(e) => p.set({ bidWindow: e.target.value as BidWindow })}>
          <option value="short">3 s</option>
          <option value="normal">6 s</option>
          <option value="long">12 s</option>
        </select>
      </label>
      <label className="opt-inline">
        <input type="checkbox" checked={p.autoDraw} onChange={(e) => p.set({ autoDraw: e.target.checked })} /> Draw my cards automatically
      </label>
      <label className="opt-inline">
        <input type="checkbox" checked={p.sounds} onChange={(e) => p.set({ sounds: e.target.checked })} /> Sounds
      </label>
      <label className="opt-inline">
        <input type="checkbox" checked={p.fourColor} onChange={(e) => p.set({ fourColor: e.target.checked })} /> Four-color suits
      </label>
      {canShowAllHands && (
        <label className="opt-inline">
          <input type="checkbox" checked={p.showAllHands} onChange={(e) => p.set({ showAllHands: e.target.checked })} /> Show all hands (watching AIs)
        </label>
      )}
      <p className="lobby-hint">
        Keys: Enter plays / confirms, Esc clears, H hints, D or Space draws. Drag across cards to select several.
      </p>
      <p className="lobby-hint">Settings are kept on this device only.</p>
    </div>
  );
}
