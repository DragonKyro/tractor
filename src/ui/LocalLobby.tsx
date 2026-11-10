// The local lobby: seat builder + match options for hot-seat / vs-AI play,
// plus "Host online", which opens a room carrying these options and AI
// seats over (the online lobby is OnlineLobby, routed by App while this
// device is in a room).

import { useMemo, useState } from 'react';
import type { Seat } from '@/core/types';
import { randomSeed } from '@/core/rng';
import { freeSeatColor, SEAT_COLORS } from '@/core/seatColors';
import { prepareAi } from '@/ai';
import { defaultConfig, MAX_PLAYERS, MIN_PLAYERS, validateConfig, type TractorConfig } from '@/game/config';
import { createInitialState } from '@/game/engine';
import { useGameStore, type MatchSetup } from '@/store/gameStore';
import { useNetworkStore, warmUpNetwork } from '@/store/networkStore';
import { getOrCreateUuid } from '@/net/identity';
import { isOnlineConfigured } from '@/net/firebaseConfig';
import { ColorSwatchPicker, RulesModal } from './lobbyParts';
import { RulesText } from './Rules';
import { TractorOptions } from './TractorOptions';

interface Props {
  /** Seats + options to start from (returning from a finished match). */
  initial?: MatchSetup;
  onBack(): void;
  onStart(): void;
}

function defaultSeats(me: string): Seat[] {
  return [
    { id: me, name: 'You', color: SEAT_COLORS[0]!, isAI: false, isLocal: true },
    { id: 'seat-1', name: 'West', color: SEAT_COLORS[1]!, isAI: true, isLocal: true },
    { id: 'seat-2', name: 'North', color: SEAT_COLORS[2]!, isAI: true, isLocal: true },
    { id: 'seat-3', name: 'East', color: SEAT_COLORS[4]!, isAI: true, isLocal: true },
  ];
}

export function LocalLobby({ initial, onBack, onStart }: Props) {
  const localUuid = useMemo(() => getOrCreateUuid(), []);
  const savedName = useNetworkStore((s) => s.myDisplayName);
  const [seats, setSeats] = useState<Seat[]>(() => initial?.seats ?? defaultSeats(localUuid));
  const [config, setConfig] = useState<TractorConfig>(() => initial?.config ?? defaultConfig(4));
  const [preparing, setPreparing] = useState(false);
  const [showRules, setShowRules] = useState(false);
  const [hostName, setHostName] = useState(savedName);
  const online = isOnlineConfigured();

  function changeSeats(next: Seat[]) {
    // Keep the mode sensible as the table grows / shrinks across odd / even.
    if (next.length % 2 === 1 && config.gameMode === 'tractor') setConfig({ ...config, gameMode: 'findingFriends' });
    if (config.numDecks !== null && config.numDecks > next.length * 2) setConfig((c) => ({ ...c, numDecks: null }));
    setSeats(next);
  }

  function addSeat() {
    if (seats.length >= MAX_PLAYERS) return;
    let n = seats.length;
    while (seats.some((s) => s.id === `seat-${n}`)) n++;
    changeSeats([...seats, { id: `seat-${n}`, name: `Player ${seats.length + 1}`, color: freeSeatColor(seats), isAI: true, isLocal: true }]);
  }

  function removeSeat(id: string) {
    if (seats.length <= MIN_PLAYERS) return;
    changeSeats(seats.filter((s) => s.id !== id));
  }

  function updateSeat(id: string, patch: Partial<Seat>) {
    setSeats(seats.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  }

  function moveSeat(i: number, delta: -1 | 1) {
    const j = i + delta;
    if (j < 0 || j >= seats.length) return;
    const next = seats.slice();
    [next[i], next[j]] = [next[j]!, next[i]!];
    setSeats(next);
  }

  const errors = validateConfig(config, seats);

  async function start() {
    if (errors.length) return;
    setPreparing(true);
    try {
      await prepareAi(seats);
    } catch (e) {
      alert('Could not load the AI: ' + (e as Error).message);
      return;
    } finally {
      setPreparing(false);
    }
    const state = createInitialState(config, randomSeed(), seats);
    const firstHuman = seats.find((s) => !s.isAI)?.id ?? null;
    useGameStore.getState().loadGame(state, firstHuman, { config, seats });
    onStart();
  }

  function hostOnline() {
    const name = hostName.trim();
    if (name) void useNetworkStore.getState().createRoom(name, { config, seats });
  }

  const humans = seats.filter((s) => !s.isAI).length;

  return (
    <div className="lobby-single">
      <div className="lobby-topbar">
        <button className="secondary" onClick={onBack}>← Back</button>
        <button className="secondary" onClick={() => setShowRules(true)}>📖 Rules</button>
      </div>

      <section className="lobby-panel">
        <h3>New game</h3>
        <h4 className="lobby-section-h">Seats · in play order{config.gameMode === 'tractor' ? ' (partners sit opposite)' : ''}</h4>
        <div className="seat-list">
          {seats.map((s, i) => (
            <div key={s.id} className="seat-row">
              <span className="seat-index" title={config.gameMode === 'tractor' ? `Team ${(i % 2) + 1}` : undefined}>
                {i + 1}
              </span>
              <ColorSwatchPicker
                value={s.color}
                takenBy={Object.fromEntries(seats.filter((x) => x.id !== s.id).map((x) => [x.color, x.name]))}
                onChange={(color) => updateSeat(s.id, { color })}
              />
              <input className="seat-name" value={s.name} maxLength={24} onChange={(e) => updateSeat(s.id, { name: e.target.value })} />
              <label className="seat-ai-toggle">
                <input type="checkbox" checked={s.isAI} onChange={(e) => updateSeat(s.id, { isAI: e.target.checked })} />
                AI
              </label>
              <span className="seat-move">
                <button className="secondary tiny" disabled={i === 0} onClick={() => moveSeat(i, -1)} aria-label="Move up">↑</button>
                <button className="secondary tiny" disabled={i === seats.length - 1} onClick={() => moveSeat(i, 1)} aria-label="Move down">↓</button>
              </span>
              <button className="secondary" onClick={() => removeSeat(s.id)} disabled={seats.length <= MIN_PLAYERS} aria-label="Remove seat">×</button>
            </div>
          ))}
        </div>
        <button className="secondary" onClick={addSeat} disabled={seats.length >= MAX_PLAYERS} style={{ marginTop: 8 }}>
          + Add seat
        </button>
        <p className="lobby-hint" style={{ marginTop: 8 }}>
          {humans === 0
            ? 'All AI: sit back and watch.'
            : humans === 1
              ? 'You against the AI.'
              : `Hot-seat: ${humans} people share this screen — it hides hands between turns.`}
        </p>

        <h4 className="lobby-section-h">Options</h4>
        <TractorOptions config={config} seats={seats} onChange={setConfig} />

        <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button onClick={start} disabled={errors.length > 0 || preparing}>
            {preparing ? 'Loading AI…' : 'Start'}
          </button>
        </div>

        {online ? (
          <div className="online-join">
            <h4>Play online</h4>
            <p className="lobby-hint">Open a room with these options and AI seats; friends join from their own devices with the room code.</p>
            <div className="online-join-row" onPointerEnter={warmUpNetwork} onFocus={warmUpNetwork}>
              <input placeholder="Your name" value={hostName} maxLength={24} onChange={(e) => setHostName(e.target.value)} />
              <button className="secondary" onClick={hostOnline} disabled={!hostName.trim()} title={hostName.trim() ? undefined : 'Enter the name friends will see'}>
                Host online
              </button>
            </div>
          </div>
        ) : (
          <p className="lobby-hint" style={{ marginTop: 16 }}>Online play is not set up yet (see src/net/firebaseConfig.ts).</p>
        )}
      </section>

      {showRules && <RulesModal title="Tractor" Rules={RulesText} onClose={() => setShowRules(false)} />}
    </div>
  );
}
