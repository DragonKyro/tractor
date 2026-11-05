// Hosts a match: the top bar, the driver (AI seats, their draws and bids,
// all-AI round advance), this device's own auto-draw, the hot-seat "pass the
// device" screen, online hand-offs for disconnected players, sounds, online
// banners and chat, the post-game review, and the Table view.
//
// Only one device drives a match: locally this one; online the networkStore
// picks one (the host while connected) — otherwise every device would deal
// and play the AI seats. Each human's own device draws their cards.

import { useEffect, useMemo, useRef, useState } from 'react';
import { chooseAIAction } from '@/ai';
import { nextAutomatedStep } from '@/ai/driver';
import { randomSeed } from '@/core/rng';
import { activeSeat, createInitialState, drawingSeat } from '@/game/engine';
import type { Action, LogKind, TractorState } from '@/game/types';
import { useGameStore, type MatchSetup } from '@/store/gameStore';
import { offlineSeats, useNetworkStore } from '@/store/networkStore';
import { paceMs, usePrefs } from '@/store/prefsStore';
import { ChatPanel } from './ChatPanel';
import { RulesModal } from './lobbyParts';
import { MatchReview } from './review/MatchReview';
import { RulesText } from './Rules';
import { playSound, type Sound } from './sounds';
import { Table, type Automation, type SummaryControls } from './table/Table';

interface Props {
  onExit(): void;
  /** Local play: back to the lobby with the finished match's lineup and options. */
  onReturnToLobby(setup: MatchSetup): void;
}

/** How long this device's link may be down before the "Reconnecting…" banner shows. */
const RECONNECT_BANNER_MS = 2000;
/** Online: a disconnected player's decision is made by the AI after this long. */
const OFFLINE_HANDOFF_MS = 30_000;
/** Online: a connected player who doesn't draw their card for this long gets it drawn for them. */
const IDLE_DRAW_MS = 20_000;

const SOUND_FOR: Partial<Record<LogKind, Sound>> = { play: 'card', throw: 'fail', trick: 'trick', bid: 'bid', friend: 'friend', round: 'round' };
const SOUND_RANK: Sound[] = ['win', 'round', 'friend', 'fail', 'trick', 'bid', 'card'];

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
  const prefs = usePrefs();
  const [rulesOpen, setRulesOpen] = useState(false);
  const [review, setReview] = useState<'stats' | 'replay' | null>(null);
  const [linkDownLong, setLinkDownLong] = useState(false);
  const [busy, setBusy] = useState(false);
  const [automation, setAutomation] = useState<Automation | null>(null);
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

  const isOfflineHuman = (s: TractorState, seat: number) => online && !s.seats[seat]!.isAI && !onlineUuids.has(s.seats[seat]!.id);

  // ---- the driver: AI seats (and disconnected players' draws). Re-runs on
  // every state change; a human acting first cancels the pending step.
  useEffect(() => {
    setAutomation(null);
    if (!state || !driving || !aiReady || state.status !== 'playing') return;
    const step = nextAutomatedStep(state, (seat) => state.seats[seat]!.isAI, {
      drawsFor: (seat) => state.seats[seat]!.isAI || isOfflineHuman(state, seat),
    });
    if (!step) return;
    const delay = paceMs(step.pace, prefs);
    if (step.pace === 'bidWindow') setAutomation({ kind: 'bidWindow', firesAt: Date.now() + delay, seat: activeSeat(state) });
    const timer = setTimeout(() => {
      if (useGameStore.getState().state !== state) return;
      const err = useGameStore.getState().dispatch(step.action, { ai: true });
      if (err) console.warn('[driver] step refused:', err, step.action);
    }, delay);
    return () => clearTimeout(timer);
  }, [state, driving, aiReady, prefs.aiSpeed, prefs.dealSpeed, prefs.bidWindow, onlineUuids]);

  // ---- whose hand this screen shows
  const humans = useMemo(() => (state ? state.seats.map((s, i) => ({ s, i })).filter((x) => !x.s.isAI) : []), [state]);
  const hotSeat = !online && humans.length >= 2;
  const lastCleared = useRef<number | null>(null);
  const [, force] = useState(0);
  const active = state ? activeSeat(state) : null;
  const activeHuman = state && active !== null && !state.seats[active]!.isAI ? active : null;
  let viewer: number | null = null;
  if (state) {
    if (online) {
      const i = state.seats.findIndex((s) => s.id === localPlayerId);
      viewer = i >= 0 ? i : null;
    } else if (humans.length === 1) {
      viewer = humans[0]!.i;
    } else if (humans.length >= 2) {
      viewer = activeHuman ?? lastCleared.current ?? humans[0]!.i;
    }
  }
  const mySeat = online ? viewer : humans.length === 1 ? humans[0]!.i : null;

  // ---- this device's own draws: automatic at the deal pace (or the Draw button).
  const drawer = state ? drawingSeat(state) : null;
  useEffect(() => {
    if (!state || mySeat === null || drawer !== mySeat || !prefs.autoDraw || role === 'spectator') return;
    const t = setTimeout(() => {
      if (useGameStore.getState().state !== state) return;
      useGameStore.getState().dispatch({ type: 'draw', seat: mySeat });
    }, paceMs('deal', prefs));
    return () => clearTimeout(t);
  }, [state, mySeat, drawer, prefs.autoDraw, prefs.dealSpeed, role]);

  // ---- online: hand a disconnected player's decision (or an idle draw) to the AI.
  const waitingOn = drawer ?? active;
  const waitingHuman = state && waitingOn !== null && !state.seats[waitingOn]!.isAI ? waitingOn : null;
  const waitingOffline = state !== null && waitingHuman !== null && isOfflineHuman(state, waitingHuman);
  useEffect(() => {
    if (!state || !online || !driving || waitingHuman === null || waitingHuman === mySeat) return;
    const drawing = drawer === waitingHuman;
    if (drawing && waitingOffline) return; // the driver draws for them at once
    if (!drawing && !waitingOffline) return; // connected players decide for themselves
    const delay = drawing ? IDLE_DRAW_MS : OFFLINE_HANDOFF_MS;
    if (!drawing) setAutomation({ kind: 'handoff', firesAt: Date.now() + delay, seat: waitingHuman });
    const t = setTimeout(() => {
      const cur = useGameStore.getState().state;
      if (cur !== state) return;
      const a: Action | null = drawing ? { type: 'draw', seat: waitingHuman } : chooseAIAction(cur, waitingHuman, 'normal');
      if (a) useGameStore.getState().dispatch(a, { ai: true });
    }, delay);
    return () => clearTimeout(t);
  }, [state, online, driving, waitingHuman, waitingOffline, drawer, mySeat]);

  // ---- sounds: the most notable new log line, and a ping when it's your turn.
  const lastLog = useRef<number | null>(null);
  useEffect(() => {
    if (!state) return;
    const seen = lastLog.current;
    lastLog.current = state.logCount;
    if (seen === null || state.logCount < seen || !prefs.sounds) return;
    const fresh = state.log.filter((e) => e.n >= seen);
    if (fresh.length === 0 || fresh.length > 8) return; // nothing new, or catching up after a rejoin
    const kinds = new Set<Sound>(fresh.map((e) => SOUND_FOR[e.kind]).filter((k): k is Sound => !!k));
    if (state.status === 'gameOver') kinds.add('win');
    const pick = SOUND_RANK.find((k) => kinds.has(k));
    if (pick) playSound(pick);
  }, [state?.logCount]);
  const lastTurn = useRef<number | null>(null);
  useEffect(() => {
    const mine = viewer !== null && (active === viewer || drawer === viewer);
    if (mine && lastTurn.current !== viewer && prefs.sounds && !(drawer === viewer && prefs.autoDraw)) playSound('turn');
    lastTurn.current = mine ? viewer : null;
  }, [active, drawer, viewer, prefs.sounds, prefs.autoDraw]);

  if (!state) {
    return (
      <div>
        <p>No game in progress.</p>
        <button onClick={onExit}>← Back</button>
      </div>
    );
  }

  const passFor =
    hotSeat && state.status === 'playing' && state.round.stage !== 'roundOver' && activeHuman !== null && lastCleared.current !== activeHuman
      ? activeHuman
      : null;

  const dispatch = (a: Action) => (role === 'spectator' ? 'Spectators cannot move.' : useGameStore.getState().dispatch(a));

  // ---- between rounds / after the match
  const hostName = state.seats.find((s) => s.id === hostUuid)?.name ?? 'the host';
  const summary: SummaryControls = { onReview: (kind) => setReview(kind) };
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
    setReview(null);
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
  const watching = !online && humans.length === 0;

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
            <span className="net-pill net-pill-warn" title="The AI plays for them if they're needed for 30 seconds">
              {away.map((s) => s.name).join(', ')} offline
            </span>
          )}
          {state.history.length > 0 && (
            <button className="secondary" onClick={() => setReview('stats')} title="Rounds so far, levels and points">📊<span className="game-host-wide"> Stats</span></button>
          )}
          <button className="secondary" onClick={() => setRulesOpen(true)}>📖<span className="game-host-wide"> Rules</span></button>
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
      <Table
        state={state}
        viewer={passFor !== null ? null : viewer}
        dispatch={dispatch}
        summary={summary}
        automation={automation}
        showAllHands={watching && prefs.showAllHands}
        canShowAllHands={watching}
      />
      {online && <ChatPanel />}
      {rulesOpen && <RulesModal title="Tractor" Rules={RulesText} onClose={() => setRulesOpen(false)} />}
      {review && <MatchReview state={state} initialTab={review} onClose={() => setReview(null)} viewer={viewer} />}
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

