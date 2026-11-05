// Hosts a match: the top bar, the automation driver (live deal, AI seats,
// all-AI round advance), the hot-seat "pass the device" screen, online
// banners and chat, and the Table view.
//
// Only one device drives a match: locally this one; online the networkStore
// picks one (the host while connected) — otherwise every device would deal
// and play the AI seats.

import { useEffect, useMemo, useRef, useState } from 'react';
import { nextAutomatedStep } from '@/ai/driver';
import { randomSeed } from '@/core/rng';
import { activeSeat, createInitialState } from '@/game/engine';
import { useGameStore, type MatchSetup } from '@/store/gameStore';
import { offlineSeats, useNetworkStore } from '@/store/networkStore';
import { paceMs, usePrefs } from '@/store/prefsStore';
import type { Action } from '@/game/types';
import { ChatPanel } from './ChatPanel';
import { RulesModal } from './lobbyParts';
import { RulesText } from './Rules';
import { Table, type SummaryControls } from './table/Table';

interface Props {
  onExit(): void;
  /** Local play: back to the lobby with the finished match's lineup and options. */
  onReturnToLobby(setup: MatchSetup): void;
}

/** How long this device's link may be down before the "Reconnecting…" banner shows. */
const RECONNECT_BANNER_MS = 2000;

export function GameHost({ onExit, onReturnToLobby }: Props) {
  const state = useGameStore((s) => s.state);
  const localPlayerId = useGameStore((s) => s.localPlayerId);
  const aiReady = useGameStore((s) => s.aiReady);
  const role = useNetworkStore((s) => s.role);
  const roomCode = useNetworkStore((s) => s.roomCode);
  const onlineUuids = useNetworkStore((s) => s.onlineUuids);
  const serverConnected = useNetworkStore((s) => s.serverConnected);
  const notice = useNetworkStore((s) => s.notice);
  const aiDriver = useNetworkStore((s) => s.aiDriver);
  const hostUuid = useNetworkStore((s) => s.hostUuid);
  const speed = usePrefs((s) => s.aiSpeed);
  const [rulesOpen, setRulesOpen] = useState(false);
  const [linkDownLong, setLinkDownLong] = useState(false);
  const [busy, setBusy] = useState(false);
  const online = role !== 'solo';
  const driving = !online || aiDriver;

  useEffect(() => {
    if (serverConnected) {
      setLinkDownLong(false);
      return;
    }
    const t = setTimeout(() => setLinkDownLong(true), RECONNECT_BANNER_MS);
    return () => clearTimeout(t);
  }, [serverConnected]);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => useNetworkStore.getState().dismissNotice(), 6000);
    return () => clearTimeout(t);
  }, [notice]);

  // The driver: deal, play AI seats, move all-AI tables on. Re-runs on every
  // state change; a human acting first cancels a pending step.
  useEffect(() => {
    if (!state || !driving || !aiReady || state.status !== 'playing') return;
    const step = nextAutomatedStep(state, (seat) => state.seats[seat]!.isAI);
    if (!step) return;
    const anyHuman = state.seats.some((s) => !s.isAI);
    const delay = anyHuman ? paceMs(step.pace, speed) : step.pace === 'linger' ? 900 : 40;
    const timer = setTimeout(() => {
      if (useGameStore.getState().state !== state) return;
      const err = useGameStore.getState().dispatch(step.action, { ai: true });
      if (err) console.warn('[driver] step refused:', err, step.action);
    }, delay);
    return () => clearTimeout(timer);
  }, [state, driving, aiReady, speed]);

  // ---- whose hand this screen shows
  const humans = useMemo(() => (state ? state.seats.map((s, i) => ({ s, i })).filter((x) => !x.s.isAI) : []), [state]);
  const hotSeat = !online && humans.length >= 2;
  const lastCleared = useRef<number | null>(null);
  const [, force] = useState(0);
  if (!state) {
    return (
      <div>
        <p>No game in progress.</p>
        <button onClick={onExit}>← Back</button>
      </div>
    );
  }
  const active = activeSeat(state);
  const activeHuman = active !== null && !state.seats[active]!.isAI ? active : null;
  let viewer: number | null;
  if (online) {
    const i = state.seats.findIndex((s) => s.id === localPlayerId);
    viewer = i >= 0 ? i : null;
  } else if (humans.length === 0) {
    viewer = null;
  } else if (humans.length === 1) {
    viewer = humans[0]!.i;
  } else {
    viewer = activeHuman ?? lastCleared.current ?? humans[0]!.i;
  }
  const passFor =
    hotSeat && state.status === 'playing' && state.round.stage !== 'roundOver' && activeHuman !== null && lastCleared.current !== activeHuman
      ? activeHuman
      : null;

  const dispatch = (a: Action) => (role === 'spectator' ? 'Spectators cannot move.' : useGameStore.getState().dispatch(a));

  // ---- between rounds / after the match
  const hostName = state.seats.find((s) => s.id === hostUuid)?.name ?? 'the host';
  const summary: SummaryControls = {};
  if (driving) summary.onNextRound = () => void dispatch({ type: 'nextRound' });
  else summary.waitingText = `Waiting for ${hostName} to deal the next round…`;
  if (state.status === 'gameOver') {
    if (!online || role === 'host') {
      summary.onPlayAgain = () => void playAgain();
      summary.onLobby = () => (online ? useNetworkStore.getState().hostReturnToLobby() : backToLobby());
    } else {
      summary.waitingText = `Waiting for ${hostName} to start another game…`;
    }
  }

  async function playAgain() {
    if (busy) return;
    if (online) {
      setBusy(true);
      const err = await useNetworkStore.getState().hostPlayAgain();
      setBusy(false);
      if (err) alert(err);
      return;
    }
    const cur = useGameStore.getState();
    if (!cur.setup) return;
    lastCleared.current = null;
    cur.loadGame(createInitialState(cur.setup.config, randomSeed(), cur.setup.seats), cur.localPlayerId, cur.setup);
  }

  function backToLobby() {
    const setup = useGameStore.getState().setup;
    useGameStore.getState().clear();
    if (setup) onReturnToLobby(setup);
    else onExit();
  }

  function leave() {
    if (online) {
      useNetworkStore.getState().leaveRoom();
      return;
    }
    if (state!.status === 'playing' && !confirm('Leave this game? It will be lost.')) return;
    useGameStore.getState().clear();
    onExit();
  }

  const away = online ? offlineSeats(state.seats, onlineUuids) : [];

  return (
    <div className="game-host">
      <div className="game-host-bar">
        <strong className="game-host-title">Tractor</strong>
        <div className="game-host-actions">
          {online && roomCode && (
            <span className="net-pill" title={role === 'spectator' ? 'Watching: you have no seat in this match' : undefined}>
              <span className={`net-dot net-dot-${role}`} />
              <span className="net-pill-role">{role === 'spectator' ? 'watching' : role} · room</span> <code>{roomCode}</code>
            </span>
          )}
          {aiDriver && role !== 'host' && state.status === 'playing' && (
            <span className="net-pill" title="The host is away: this device runs the deal and the AI seats until they're back">running the table</span>
          )}
          {away.length > 0 && state.status === 'playing' && (
            <span className="net-pill net-pill-warn" title="Their moves wait until they reconnect">
              {away.map((s) => s.name).join(', ')} offline
            </span>
          )}
          <button className="secondary" onClick={() => setRulesOpen(true)}>📖 Rules</button>
          <button className="secondary" onClick={leave}>
            Leave<span className="game-host-wide"> {online ? 'room' : 'game'}</span>
          </button>
        </div>
      </div>
      {online && linkDownLong && <div className="net-banner">Reconnecting to the game server… your moves will go through once it's back.</div>}
      {notice && (
        <div className="net-banner net-banner-notice" role="status" onClick={() => useNetworkStore.getState().dismissNotice()}>
          {notice}
        </div>
      )}
      <Table state={state} viewer={passFor !== null ? null : viewer} dispatch={dispatch} summary={summary} />
      {online && <ChatPanel />}
      {rulesOpen && <RulesModal title="Tractor" Rules={RulesText} onClose={() => setRulesOpen(false)} />}
      {passFor !== null && (
        <div className="hotseat-pass-backdrop">
          <div className="hotseat-pass-card">
            <div className="hotseat-pass-eyebrow">Pass the device</div>
            <div className="hotseat-pass-name" style={{ color: state.seats[passFor]!.color }}>
              {state.seats[passFor]!.name}
            </div>
            <div className="hotseat-pass-sub">Your turn — make sure the others can't see.</div>
            <button
              className="hotseat-pass-ready"
              onClick={() => {
                lastCleared.current = passFor;
                force((n) => n + 1);
              }}
            >
              I'm ready →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
