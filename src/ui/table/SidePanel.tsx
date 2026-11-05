// Beside the table: who's who (levels, teams, points), friend calls, the
// game log, and this device's display settings.

import { useEffect, useRef, useState } from 'react';
import { cardLabel, totalPoints } from '@/game/cards';
import { ordinalText } from '@/game/engine';
import type { TractorState } from '@/game/types';
import { usePrefs, type AiSpeed } from '@/store/prefsStore';
import { levelText, roleOf } from './labels';

type Tab = 'players' | 'log' | 'settings';

export function SidePanel({ state }: { state: TractorState }) {
  const [tab, setTab] = useState<Tab>('players');
  return (
    <aside className="side-panel">
      <div className="side-tabs" role="tablist">
        {(['players', 'log', 'settings'] as Tab[]).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'side-tab active' : 'side-tab'} onClick={() => setTab(t)}>
            {t === 'players' ? 'Table' : t === 'log' ? 'Log' : 'Settings'}
          </button>
        ))}
      </div>
      {tab === 'players' && <Players state={state} />}
      {tab === 'log' && <Log state={state} />}
      {tab === 'settings' && <Settings />}
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
                <td>{totalPoints(r.captured[i]!) || ''}</td>
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
          {e.text}
        </div>
      ))}
    </div>
  );
}

function Settings() {
  const fourColor = usePrefs((s) => s.fourColor);
  const aiSpeed = usePrefs((s) => s.aiSpeed);
  const set = usePrefs((s) => s.set);
  return (
    <div className="side-body settings">
      <label className="opt-inline">
        <input type="checkbox" checked={fourColor} onChange={(e) => set({ fourColor: e.target.checked })} /> Four-color suits
      </label>
      <label className="opt-inline">
        AI / deal speed
        <select value={aiSpeed} onChange={(e) => set({ aiSpeed: e.target.value as AiSpeed })}>
          <option value="slow">Slow</option>
          <option value="normal">Normal</option>
          <option value="fast">Fast</option>
        </select>
      </label>
      <p className="lobby-hint">Settings are kept on this device only.</p>
    </div>
  );
}
