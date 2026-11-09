// Online play: the room lifecycle, role, lobby, presence and chat, and the
// bridge that keeps gameStore in step with the room's action log. Adapted
// from the card-collecting / catan projects (same room layout and rules).
//
// Roles:
//   solo       — no room. Hot-seat or AI play.
//   host       — made the room; edits the lobby, starts matches.
//   guest      — in the room with a seat.
//   spectator  — in the room without a seat (read-only).
//
// A room holds a lobby (seats, options) and a list of matches. The lobby's
// `match` says which one is being played (null = everyone is in the lobby).
// Every device watches that match: it builds the opening state from the start
// record (createInitialState is pure) and applies the action log in slot order
// via ActionSync, so a device that joins or reloads mid-match rebuilds it
// exactly. "Play again" is a new match; returning to the lobby sets `match`
// back to null.
//
// One device *drives* the match — deals the live deal, plays the AI seats,
// deals the next round: the host's while it's connected, else the first
// connected seated human's (in seat order), so a match goes on when the host
// drops. Every device works this out from presence, so they agree; in the
// moment it changes hands two devices may both act, and the log keeps the
// first (the other's move is dropped quietly).

import { create } from 'zustand';
import type { Seat } from '@/core/types';
import { randomSeed } from '@/core/rng';
import { freeSeatColor, SEAT_COLORS } from '@/core/seatColors';
import { prepareAi } from '@/ai';
import { defaultConfig, MAX_PLAYERS, MIN_PLAYERS, normalizeConfig, validateConfig, type TractorConfig } from '@/game/config';
import { createInitialState } from '@/game/engine';
import type { TractorState } from '@/game/types';
import {
  type ChatMessage,
  type ConnectionState,
  type LobbyState,
  type Presence,
  type Role,
  type RoomConnection,
  type RoomEvents,
  type RoomTransport,
  type StartRecord,
  ServerUnreachableError,
  generateRoomCode,
  getOrCreateUuid,
  getSavedDisplayName,
  normalizeRoomCode,
  saveDisplayName,
  setRoomCodeInUrl,
} from '@/net';
import { useGameStore } from './gameStore';
import { ActionSync } from './actionSync';

/** The build this page runs (rooms record the build that made them). */
export const APP_BUILD = typeof __APP_BUILD__ === 'string' ? __APP_BUILD__ : 'dev';
const SYSTEM_NAME = 'System';
const MAX_NAME = 24;
const UNREACHABLE_TEXT = "Couldn't reach the game server. Check this device's internet connection, then try again.";

/** Why a join gave up, for the error screen. */
export interface JoinFailure {
  code: string;
  /** notFound: the server has no room with this code. unreachable: this device couldn't reach the server. */
  reason: 'notFound' | 'unreachable';
  detail?: string;
}

/** FNV-1a over the state's JSON: devices compare it to know they built the same opening. */
export function stateChecksum(state: unknown): string {
  const json = JSON.stringify(state);
  let h = 0x811c9dc5;
  for (let i = 0; i < json.length; i++) {
    h ^= json.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

export function lobbyConfig(lobby: LobbyState): TractorConfig {
  try {
    return normalizeConfig(JSON.parse(lobby.configJson) as Partial<TractorConfig>, lobby.seats.length);
  } catch {
    return defaultConfig(lobby.seats.length);
  }
}

/** Why the host can't start this lobby yet, or null when it's ready. */
export function lobbyStartProblem(lobby: LobbyState): string | null {
  const n = lobby.seats.length;
  if (n < MIN_PLAYERS || n > MAX_PLAYERS) return `Tractor takes ${MIN_PLAYERS}–${MAX_PLAYERS} players (${n} seated).`;
  const errors = validateConfig(lobbyConfig(lobby), lobby.seats);
  return errors.length ? errors.join(' ') : null;
}

interface NetworkStore {
  connection: ConnectionState;
  errorMessage: string | null;
  role: Role;
  myUuid: string;
  myDisplayName: string;
  roomCode: string | null;
  hostUuid: string | null;
  joinFailure: JoinFailure | null;
  /** This device's link to the server; false while it reconnects (moves wait, then go through). */
  serverConnected: boolean;
  lobby: LobbyState | null;
  /** Everyone connected to the room right now. */
  people: Presence[];
  onlineUuids: Set<string>;
  chat: ChatMessage[];
  /** A one-off message about this device's moves (one taken back). */
  notice: string | null;
  /** This device drives the match being played: deals, AI seats, next round (see the header). */
  aiDriver: boolean;

  setDisplayName(name: string): void;
  /** Open a room. `setup` carries a local lobby's options and AI seats over. */
  createRoom(name: string, setup?: { config: TractorConfig; seats: Seat[] }): Promise<void>;
  joinRoom(code: string, name: string): Promise<void>;
  /** Join the room of the last failed join again. */
  retryJoin(): void;
  leaveRoom(): void;
  clearError(): void;

  hostSetConfig(config: TractorConfig): void;
  hostAddAISeat(): void;
  /** Host: remove a seat (not the host's own). */
  hostRemoveSeat(seatId: string): void;
  /** Host: move a seat earlier / later round the table (partners sit opposite). */
  hostMoveSeat(seatId: string, delta: -1 | 1): void;
  /** Rename / recolor a seat: your own, or (host) an AI seat. A color another seat wears swaps. */
  editSeat(seatId: string, patch: { name?: string; color?: string }): void;
  /** Sit at the table (between matches, while there's room). */
  takeSeat(): void;
  /** Host: start a match from the lobby. Resolves with why it couldn't, or null. */
  hostStartMatch(): Promise<string | null>;
  /** Host: start a fresh match with the same lobby, straight from the game-over screen. */
  hostPlayAgain(): Promise<string | null>;
  /** Host: end the match for everyone and go back to the lobby. */
  hostReturnToLobby(): void;

  sendChat(text: string): void;
  dismissNotice(): void;
}

// Loaded on first use so the Firebase SDK isn't in the main bundle.
// (Tests swap the module for src/net/fakeTransport.ts.)
type LoadedTransport = RoomTransport & { warmUp: () => void };
let transportPromise: Promise<LoadedTransport> | null = null;
function loadTransport(): Promise<LoadedTransport> {
  transportPromise ??= import('@/net/firebaseRoom')
    .then((m) => m.getTransport())
    .catch((e: unknown) => {
      transportPromise = null;
      throw new ServerUnreachableError(`Could not load the online code (${(e as Error).message}).`);
    });
  return transportPromise;
}

/** Load the online code and sign in ahead of time (call when the online UI opens). */
export function warmUpNetwork(): void {
  loadTransport().then((t) => t.warmUp(), () => {});
}

// Internal (non-reactive) session state.
let conn: RoomConnection | null = null;
let session = 0; // bumped on every create / join / leave; stale callbacks compare against it
let watchedMatch: number | null = null;
let stopWatch: (() => void) | null = null;
let sync: ActionSync<TractorState> | null = null;
/** The watched match's start record has been applied (gameStore holds it). */
let matchLoaded = false;
/** The match this device has prepared the AI for. */
let preparedMatch: number | null = null;
let preparingMatch: number | null = null;
/** The watched match's start record, once applied. */
let currentStart: StartRecord | null = null;
let lastPresence: Map<string, string> | null = null;
let divergeWarned = false;

function messageId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function cleanName(name: string): string {
  return name.trim().slice(0, MAX_NAME);
}

/** Take (or keep) a seat for `uuid` while the lobby is open. */
function seatEdit(lobby: LobbyState, uuid: string, name: string): LobbyState | null {
  if (lobby.match !== null) return null;
  const i = lobby.seats.findIndex((s) => s.id === uuid);
  if (i >= 0) {
    if (lobby.seats[i]!.name === name) return null;
    return { ...lobby, seats: lobby.seats.map((s, j) => (j === i ? { ...s, name } : s)) };
  }
  if (lobby.seats.length >= MAX_PLAYERS) return null;
  return {
    ...lobby,
    seats: [...lobby.seats, { id: uuid, name, color: freeSeatColor(lobby.seats), isAI: false, isLocal: false }],
  };
}

/** The online lobby's opening seats: the host, then any AI seats carried over from a local lobby. */
function openingSeats(me: string, name: string, local?: Seat[]): Seat[] {
  const firstHuman = local?.find((s) => !s.isAI);
  const host: Seat = { id: me, name, color: firstHuman?.color ?? SEAT_COLORS[0]!, isAI: false, isLocal: false };
  const seats: Seat[] = [host];
  for (const s of local ?? []) {
    if (!s.isAI || seats.length >= MAX_PLAYERS) continue;
    const n = seats.filter((x) => x.isAI).length + 1;
    const color = seats.some((x) => x.color === s.color) ? freeSeatColor(seats) : s.color;
    seats.push({ id: `ai-${n}`, name: s.name || `AI ${n}`, color, isAI: true, isLocal: false });
  }
  return seats;
}

function nextAiSeat(seats: Seat[]): Seat {
  for (let n = 1; ; n++) {
    if (seats.some((s) => s.id === `ai-${n}`)) continue;
    return { id: `ai-${n}`, name: `AI ${n}`, color: freeSeatColor(seats), isAI: true, isLocal: false };
  }
}

export const useNetworkStore = create<NetworkStore>((set, get) => {
  // Every move this device makes in an online match goes to the log; a spectator makes none.
  useGameStore.getState().registerBroadcastHandler((action, before, fromAI) => sync?.local(action, before, fromAI));
  useGameStore.getState().registerDispatchGuard(() => get().connection === 'in-game' && get().role === 'spectator');

  const myUuid = getOrCreateUuid();

  /** End the current session (if any) and start a new one; returns its id. */
  function beginSession(): number {
    session++;
    stopWatch?.();
    stopWatch = null;
    sync?.close();
    sync = null;
    conn?.leave();
    conn = null;
    if (matchLoaded) useGameStore.getState().clear();
    watchedMatch = null;
    matchLoaded = false;
    preparedMatch = null;
    preparingMatch = null;
    currentStart = null;
    lastPresence = null;
    divergeWarned = false;
    if (get().aiDriver) set({ aiDriver: false });
    return session;
  }

  function sysMessage(text: string): ChatMessage {
    return { id: messageId(), byUuid: get().myUuid, name: SYSTEM_NAME, text, ts: Date.now(), kind: 'system' };
  }

  function addSystem(text: string): void {
    set({ chat: [...get().chat, sysMessage(text)] });
  }

  function makeSync(match: number): ActionSync<TractorState> {
    return new ActionSync<TractorState>({
      apply: (entry) => {
        if (useGameStore.getState().applyLocal(entry.action as never) || divergeWarned) return;
        divergeWarned = true;
        addSystem(
          "A move from the room didn't fit this device's game, so it may be out of step. "
            + "Reload every device's page so they all run the same version.",
        );
      },
      reapply: (entry) => useGameStore.getState().applyLocal(entry.action as never),
      snapshot: () => useGameStore.getState().state!,
      restore: (s) => useGameStore.getState().restoreState(s),
      write: (seq, entry) => (conn ? conn.appendAction(match, seq, entry) : Promise.resolve(false)),
      dropped: (_action, why, quiet) => {
        // A driver move that lost its slot (two devices briefly both driving):
        // nobody to tell, and the driver decides again from the new state.
        if (quiet) return;
        set({
          notice:
            why === 'overtaken'
              ? "Another player's move went through first, so yours was taken back."
              : 'The server refused that move, so it was taken back.',
        });
      },
    });
  }

  /** Who drives the match now: the host while connected, else the first connected seated human. */
  function driverUuid(): string | null {
    const rec = currentStart;
    if (!rec) return null;
    const online = get().onlineUuids;
    const order = [rec.hostUuid, ...rec.seats.filter((x) => !x.isAI && x.id !== rec.hostUuid).map((x) => x.id)];
    return order.find((uuid) => online.has(uuid)) ?? null;
  }

  /** Recompute whether this device drives the match (match start, presence changes). */
  function updateDriver(my: number): void {
    if (my !== session) return;
    const s = get();
    const rec = currentStart;
    const me = s.myUuid;
    // A device that isn't the host waits to hear who's connected before
    // taking over (right after joining it only knows about itself).
    const driving = !!rec && (rec.hostUuid === me || lastPresence !== null) && driverUuid() === me;
    if (driving !== s.aiDriver) {
      set({ aiDriver: driving });
      if (rec && rec.hostUuid !== me) {
        const host = rec.seats.find((x) => x.id === rec.hostUuid)?.name ?? 'The host';
        addSystem(driving ? `${host} is away, so this device is running the table until they're back.` : 'Handed the table back to the host.');
      }
    }
    if (driving) ensureAiPrepared(my);
  }

  /** Prepare the AI once per match on the driving device; the AI driver waits on it. */
  function ensureAiPrepared(my: number): void {
    const rec = currentStart;
    const match = watchedMatch;
    if (!rec || match === null || preparedMatch === match || preparingMatch === match) return;
    preparingMatch = match;
    useGameStore.getState().setAiReady(false);
    prepareAi(rec.seats).then(
      () => {
        if (my !== session || watchedMatch !== match) return;
        preparedMatch = match;
        preparingMatch = null;
        useGameStore.getState().setAiReady(true);
      },
      (e: unknown) => {
        if (my !== session || watchedMatch !== match) return;
        preparingMatch = null;
        addSystem(`Couldn't load the AI (${(e as Error).message}); its seats are paused. Reload to try again.`);
      },
    );
  }

  /** Watch the lobby's current match (or none), dropping the previous one. */
  function followMatch(my: number): void {
    const lobby = get().lobby;
    if (!conn || !lobby || my !== session) return;
    const match = lobby.match;
    if (match === watchedMatch) return;
    stopWatch?.();
    stopWatch = null;
    sync?.close();
    sync = null;
    const wasLoaded = matchLoaded;
    watchedMatch = match;
    matchLoaded = false;
    currentStart = null;
    divergeWarned = false;
    if (get().aiDriver) set({ aiDriver: false });
    if (match === null) {
      // Back to the lobby. (A new match replaces the old one's state when its
      // start record lands, so the game-over screen stays up until then.)
      if (wasLoaded) useGameStore.getState().clear();
      return;
    }
    sync = makeSync(match);
    stopWatch = conn.watchMatch(match, {
      start: (rec) => my === session && watchedMatch === match && onStart(rec, my),
      action: (seq, entry) => my === session && watchedMatch === match && sync?.receive(seq, entry),
    });
  }

  function onLobby(lobby: LobbyState, my: number): void {
    const s = get();
    const me = s.myUuid;
    const lobbyRole: Role = lobby.hostUuid === me ? 'host' : lobby.seats.some((x) => x.id === me) ? 'guest' : 'spectator';
    const toLobby = lobby.match === null && (s.connection === 'connecting' || s.connection === 'in-game');
    set({
      lobby,
      hostUuid: lobby.hostUuid,
      // During a match the role comes from the match's seats (onStart).
      role: lobby.match !== null && matchLoaded ? s.role : lobbyRole,
      connection: toLobby ? 'lobby' : s.connection,
    });
    if (toLobby && s.connection === 'in-game') addSystem('Back in the lobby');
    followMatch(my);
  }

  function onStart(rec: StartRecord, my: number): void {
    if (matchLoaded) return; // written once; ignore repeats
    let config: TractorConfig;
    let state: TractorState;
    try {
      config = normalizeConfig(JSON.parse(rec.configJson) as Partial<TractorConfig>, rec.seats.length);
      state = createInitialState(config, rec.seed, rec.seats);
    } catch (e) {
      addSystem(`This device couldn't set up the match (${(e as Error).message}). Reload the page.`);
      return;
    }
    if (stateChecksum(state) !== rec.check) {
      addSystem(
        "This device set up a different opening from the host's, so moves won't line up. "
          + "Reload every device's page so they all run the same version.",
      );
    }
    matchLoaded = true;
    const s = get();
    const me = s.myUuid;
    const seated = rec.seats.some((x) => !x.isAI && x.id === me);
    const role: Role = rec.hostUuid === me ? 'host' : seated ? 'guest' : 'spectator';
    const wasPlaying = s.connection === 'lobby' || s.connection === 'in-game';
    useGameStore.getState().loadGame(state, seated ? me : null, { config, seats: rec.seats });
    currentStart = rec;
    set({ connection: 'in-game', role, hostUuid: rec.hostUuid, errorMessage: null, joinFailure: null });
    addSystem(wasPlaying ? 'Match started' : seated ? 'Rejoined the match' : 'Watching the match');
    updateDriver(my);
    // Replay the log from the start: every device ends up with the same state.
    sync?.begin();
  }

  function onChat(msg: ChatMessage): void {
    if (get().chat.some((c) => c.id === msg.id)) return;
    set({ chat: [...get().chat, { ...msg, kind: 'user' }] });
  }

  function onPresence(people: Presence[]): void {
    const me = get().myUuid;
    const now = new Map<string, string>(people.map((p) => [p.uuid, p.name]));
    const online = new Set(now.keys());
    online.add(me);
    const lines: string[] = [];
    if (lastPresence) {
      for (const [uuid, name] of now) {
        if (!lastPresence.has(uuid) && uuid !== me) lines.push(`${name || 'A player'} joined`);
      }
      for (const [uuid, name] of lastPresence) {
        if (!now.has(uuid) && uuid !== me) lines.push(`${name || 'A player'} disconnected`);
      }
    }
    lastPresence = now;
    // One entry per person (a device can briefly hold two connections).
    const unique = [...new Map(people.map((p) => [p.uuid, p])).values()];
    set({
      people: unique,
      onlineUuids: online,
      chat: lines.length ? [...get().chat, ...lines.map(sysMessage)] : get().chat,
    });
    updateDriver(session);
  }

  function roomEvents(my: number): RoomEvents {
    const live = () => my === session;
    return {
      lobby: (lobby) => live() && onLobby(lobby, my),
      chat: (msg) => live() && onChat(msg),
      presence: (people) => live() && onPresence(people),
      connected: (up) => live() && set({ serverConnected: up }),
    };
  }

  function editLobby(edit: (lobby: LobbyState) => LobbyState | null): Promise<LobbyState | null> {
    if (!conn) return Promise.resolve(null);
    return conn.updateLobby(edit).catch((e: unknown) => {
      console.warn('[net] lobby update failed:', (e as Error).message);
      return null;
    });
  }

  function failJoin(code: string, reason: JoinFailure['reason'], detail?: string): void {
    beginSession();
    if (reason === 'notFound') setRoomCodeInUrl(null);
    set({
      connection: 'error',
      joinFailure: { code, reason, detail },
      errorMessage:
        reason === 'notFound'
          ? `There's no room with code ${code}. Check the code with the host (rooms are cleared a week after they're made).`
          : UNREACHABLE_TEXT,
    });
  }

  async function startMatch(from: number | null): Promise<string | null> {
    const c = conn;
    const my = session;
    const lobby = get().lobby;
    if (get().role !== 'host' || !c || !lobby || lobby.match !== from) return null;
    const problem = lobbyStartProblem(lobby);
    if (problem) return problem;
    try {
      await prepareAi(lobby.seats);
    } catch (e) {
      return `Could not load the AI: ${(e as Error).message}`;
    }
    if (my !== session) return null;
    // Lock the seats and options first, atomically: a guest taking a seat at
    // the same moment either makes it in or finds the lobby closed.
    let refused: string | null = null;
    const final = await editLobby((l) => {
      refused = l.match !== from ? 'The match already started.' : lobbyStartProblem(l);
      return refused ? null : { ...l, match: l.matchCount, matchCount: l.matchCount + 1 };
    });
    if (my !== session) return null;
    if (!final || final.match === null) return refused ?? "Couldn't reach the room. Try again.";
    const match = final.match;
    preparedMatch = match;
    const reopen = () => void editLobby((l) => (l.match === match ? { ...l, match: from } : null));
    const seed = randomSeed();
    let check: string;
    try {
      check = stateChecksum(createInitialState(lobbyConfig(final), seed, final.seats));
    } catch (e) {
      reopen();
      return `Couldn't set up the match: ${(e as Error).message}`;
    }
    try {
      await c.startMatch(match, {
        configJson: JSON.stringify(lobbyConfig(final)),
        seed,
        seats: final.seats,
        hostUuid: get().myUuid,
        startedAt: Date.now(),
        check,
      });
    } catch (e) {
      if (my === session) reopen();
      return `Couldn't start the match: ${(e as Error).message}`;
    }
    return null;
  }

  return {
    connection: 'disconnected',
    errorMessage: null,
    role: 'solo',
    myUuid,
    myDisplayName: getSavedDisplayName(),
    roomCode: null,
    hostUuid: null,
    joinFailure: null,
    serverConnected: true,
    lobby: null,
    people: [],
    onlineUuids: new Set(),
    chat: [],
    notice: null,
    aiDriver: false,

    setDisplayName: (name) => {
      saveDisplayName(name);
      set({ myDisplayName: name });
    },

    createRoom: async (rawName, setup) => {
      const name = cleanName(rawName) || 'Host';
      saveDisplayName(name);
      const my = beginSession();
      const me = get().myUuid;
      const seats = openingSeats(me, name, setup?.seats);
      const lobby: LobbyState = {
        hostUuid: me,
        configJson: JSON.stringify(setup ? setup.config : defaultConfig(seats.length)),
        seats,
        match: null,
        matchCount: 0,
      };
      set({
        connection: 'connecting',
        role: 'host',
        myDisplayName: name,
        roomCode: null,
        hostUuid: me,
        joinFailure: null,
        errorMessage: null,
        serverConnected: true,
        lobby,
        people: [],
        onlineUuids: new Set([me]),
        chat: [],
        notice: null,
      });
      try {
        const transport = await loadTransport();
        let c: RoomConnection | null = null;
        // A code still in use by a live room is refused; draw another.
        for (let attempt = 0; attempt < 6 && !c && my === session; attempt++) {
          c = await transport.createRoom(
            generateRoomCode(),
            { hostUuid: me, createdAt: Date.now(), build: APP_BUILD },
            lobby,
            { uuid: me, name },
            roomEvents(my),
          );
        }
        if (my !== session) {
          c?.leave();
          return;
        }
        if (!c) throw new Error('Could not find a free room code. Try again.');
        conn = c;
        setRoomCodeInUrl(c.code);
        set({ roomCode: c.code, connection: get().connection === 'connecting' ? 'lobby' : get().connection });
        followMatch(my);
      } catch (e) {
        if (my !== session) return;
        console.error('[net] failed to create room', e);
        beginSession();
        set({
          connection: 'error',
          joinFailure: null,
          errorMessage: e instanceof ServerUnreachableError ? UNREACHABLE_TEXT : (e as Error).message,
        });
      }
    },

    joinRoom: async (code, rawName) => {
      const name = cleanName(rawName) || 'Guest';
      saveDisplayName(name);
      const upper = normalizeRoomCode(code);
      const my = beginSession();
      const me = get().myUuid;
      set({
        connection: 'connecting',
        role: 'guest',
        myDisplayName: name,
        roomCode: upper,
        hostUuid: null,
        lobby: null,
        people: [],
        chat: [],
        notice: null,
        errorMessage: null,
        joinFailure: null,
        serverConnected: true,
        onlineUuids: new Set([me]),
      });
      let c: RoomConnection | null;
      try {
        const transport = await loadTransport();
        c = await transport.joinRoom(upper, { uuid: me, name }, roomEvents(my));
      } catch (e) {
        if (my !== session) return;
        console.error('[net] failed to join room', e);
        failJoin(upper, 'unreachable', (e as Error).message);
        return;
      }
      if (my !== session) {
        c?.leave();
        return;
      }
      if (!c) {
        failJoin(upper, 'notFound');
        return;
      }
      conn = c;
      setRoomCodeInUrl(upper);
      followMatch(my);
      if (c.meta.build !== APP_BUILD) {
        addSystem(
          `This room was made by another version of the app (${c.meta.build}; this device runs ${APP_BUILD}). `
            + "If moves stop lining up, reload every device's page.",
        );
      }
      // Take a seat while the lobby is open (rejoining keeps yours).
      await editLobby((lobby) => seatEdit(lobby, me, name));
    },

    retryJoin: () => {
      const code = get().joinFailure?.code ?? get().roomCode;
      if (code) void get().joinRoom(code, get().myDisplayName);
    },

    leaveRoom: () => {
      const s = get();
      const c = conn;
      conn = null;
      // Leaving the lobby gives up your seat (the host keeps theirs).
      if (c && s.connection === 'lobby' && s.role !== 'host') {
        const me = s.myUuid;
        c.updateLobby((l) =>
          l.match !== null || !l.seats.some((x) => x.id === me) ? null : { ...l, seats: l.seats.filter((x) => x.id !== me) },
        )
          .catch(() => null)
          .finally(() => c.leave());
      } else {
        c?.leave();
      }
      beginSession();
      setRoomCodeInUrl(null);
      set({
        connection: 'disconnected',
        errorMessage: null,
        role: 'solo',
        roomCode: null,
        hostUuid: null,
        joinFailure: null,
        serverConnected: true,
        lobby: null,
        people: [],
        onlineUuids: new Set(),
        chat: [],
        notice: null,
      });
    },

    clearError: () => {
      beginSession();
      setRoomCodeInUrl(null);
      set({ connection: 'disconnected', role: 'solo', errorMessage: null, joinFailure: null, roomCode: null });
    },

    hostSetConfig: (config) => {
      const json = JSON.stringify(config);
      if (get().role !== 'host' || get().lobby?.configJson === json) return;
      void editLobby((l) => (l.match !== null || l.configJson === json ? null : { ...l, configJson: json }));
    },

    hostAddAISeat: () => {
      if (get().role !== 'host') return;
      void editLobby((l) => (l.match !== null || l.seats.length >= MAX_PLAYERS ? null : { ...l, seats: [...l.seats, nextAiSeat(l.seats)] }));
    },

    hostRemoveSeat: (seatId) => {
      if (get().role !== 'host' || seatId === get().myUuid) return; // the host's seat stays
      void editLobby((l) =>
        l.match !== null || !l.seats.some((s) => s.id === seatId) ? null : { ...l, seats: l.seats.filter((s) => s.id !== seatId) },
      );
    },

    hostMoveSeat: (seatId, delta) => {
      if (get().role !== 'host') return;
      void editLobby((l) => {
        const i = l.seats.findIndex((s) => s.id === seatId);
        const j = i + delta;
        if (l.match !== null || i < 0 || j < 0 || j >= l.seats.length) return null;
        const seats = l.seats.slice();
        [seats[i], seats[j]] = [seats[j]!, seats[i]!];
        return { ...l, seats };
      });
    },

    editSeat: (seatId, patch) => {
      const s = get();
      const target = s.lobby?.seats.find((x) => x.id === seatId);
      if (!target || s.connection !== 'lobby') return;
      const mine = target.id === s.myUuid;
      if (!mine && !(target.isAI && s.role === 'host')) return;
      const name = patch.name === undefined ? undefined : cleanName(patch.name);
      if (name === '') return;
      if (mine && name) {
        saveDisplayName(name);
        set({ myDisplayName: name });
      }
      void editLobby((l) => {
        const seat = l.seats.find((x) => x.id === seatId);
        if (l.match !== null || !seat) return null;
        const next = {
          ...seat,
          ...(name ? { name } : {}),
          ...(patch.color ? { color: patch.color } : {}),
        };
        if (next.name === seat.name && next.color === seat.color) return null;
        return {
          ...l,
          seats: l.seats.map((x) => (x.id === seatId ? next : patch.color && x.color === patch.color ? { ...x, color: seat.color } : x)),
        };
      });
    },

    takeSeat: () => {
      const s = get();
      if (s.connection !== 'lobby') return;
      void editLobby((l) => seatEdit(l, s.myUuid, s.myDisplayName || 'Guest'));
    },

    hostStartMatch: () => startMatch(null),

    hostPlayAgain: () => {
      const current = get().lobby?.match;
      return current === null || current === undefined ? Promise.resolve(null) : startMatch(current);
    },

    hostReturnToLobby: () => {
      if (get().role !== 'host') return;
      void editLobby((l) => (l.match === null ? null : { ...l, match: null }));
    },

    sendChat: (text) => {
      const trimmed = text.trim();
      const s = get();
      if (!trimmed || !conn) return;
      const seatName = s.lobby?.seats.find((x) => x.id === s.myUuid)?.name;
      const msg: ChatMessage = {
        id: messageId(),
        byUuid: s.myUuid,
        name: seatName || s.myDisplayName || 'Player',
        text: trimmed.slice(0, 500),
        ts: Date.now(),
        kind: 'user',
      };
      // Shown at once; the room's copy is de-duplicated by id.
      set({ chat: [...s.chat, msg] });
      conn.sendChat(msg).catch((e: unknown) => console.warn('[net] chat send failed:', (e as Error).message));
    },

    dismissNotice: () => set({ notice: null }),
  };
});

/** In a room (connecting, lobby, playing, or an error screen for one). */
export function isOnline(connection: ConnectionState): boolean {
  return connection !== 'disconnected';
}

/** Human seats of the current match whose owner isn't connected right now. */
export function offlineSeats(seats: Seat[], onlineUuids: Set<string>): Seat[] {
  return seats.filter((s) => !s.isAI && !onlineUuids.has(s.id));
}
