// Test-only: an in-memory stand-in for firebaseRoom.ts
// (vi.mock('@/net/firebaseRoom', () => import('@/net/fakeTransport'))).
// Each "device" in a test is its own module graph (vi.resetModules) with its
// own stores; they all share one server on globalThis, which behaves like the
// Realtime Database as the app uses it: write-once start records and log
// slots, atomic lobby edits, presence that disappears when a device leaves,
// payloads through JSON, and every event delivered asynchronously (the
// author's own writes too).
// `fakeServer().up = false` models a device that can't reach the server.
// `fakeServer().holdWrites = true` queues log writes until `releaseWrites()`,
// so tests can make two devices race for the same slot.
// Not imported by production code.

import {
  ROOM_LIFETIME_MS,
  SWEEP_BATCH,
  ServerUnreachableError,
  decodeAction,
  encodeAction,
  type MatchEvents,
  type RoomConnection,
  type RoomEvents,
  type RoomTransport,
} from './transport';
import type { LobbyState, Presence, RoomMeta, StartRecord } from './types';

interface Listener {
  id: number;
  me: Presence;
  events: RoomEvents;
}

interface MatchWatcher {
  id: number;
  listener: number;
  events: MatchEvents;
}

export interface FakeMatch {
  start: string | null;
  actions: Map<number, string>;
}

export interface FakeRoom {
  meta: RoomMeta;
  lobby: string;
  matches: Map<number, FakeMatch>;
  chat: string[];
  listeners: Map<number, Listener>;
  watchers: Map<number, Map<number, MatchWatcher>>;
}

interface FakeServer {
  rooms: Map<string, FakeRoom>;
  up: boolean;
  holdWrites: boolean;
  held: (() => void)[];
  nextId: number;
}

const KEY = '__tractorFakeServer';

export function fakeServer(): FakeServer {
  const g = globalThis as unknown as Record<string, FakeServer | undefined>;
  return (g[KEY] ??= { rooms: new Map(), up: true, holdWrites: false, held: [], nextId: 0 });
}

export function resetFakeServer(): void {
  delete (globalThis as unknown as Record<string, FakeServer | undefined>)[KEY];
}

/** Let held log writes reach the server, in the order they were made. */
export function releaseWrites(): void {
  const s = fakeServer();
  s.holdWrites = false;
  const held = s.held.splice(0);
  held.forEach((f) => f());
}

const later = (f: () => void) => {
  void Promise.resolve().then(f);
};

function presenceOf(room: FakeRoom): Presence[] {
  return [...room.listeners.values()].map((l) => ({ ...l.me }));
}

function broadcast(room: FakeRoom, f: (l: Listener) => void): void {
  for (const l of [...room.listeners.values()]) later(() => room.listeners.has(l.id) && f(l));
}

function matchOf(room: FakeRoom, match: number): FakeMatch {
  let m = room.matches.get(match);
  if (!m) {
    m = { start: null, actions: new Map() };
    room.matches.set(match, m);
  }
  return m;
}

function watchersOf(room: FakeRoom, match: number): Map<number, MatchWatcher> {
  let w = room.watchers.get(match);
  if (!w) {
    w = new Map();
    room.watchers.set(match, w);
  }
  return w;
}

function toWatchers(room: FakeRoom, match: number, f: (w: MatchWatcher) => void): void {
  const ws = watchersOf(room, match);
  for (const w of [...ws.values()]) later(() => ws.has(w.id) && room.listeners.has(w.listener) && f(w));
}

function requireServer(): FakeServer {
  const s = fakeServer();
  if (!s.up) throw new ServerUnreachableError('Fake server is down.');
  return s;
}

function open(code: string, room: FakeRoom, me: Presence, events: RoomEvents): RoomConnection {
  const server = fakeServer();
  const id = server.nextId++;
  const listener: Listener = { id, me: { ...me }, events };
  room.listeners.set(id, listener);
  const mine = new Set<() => void>();

  // Initial load, as the database's listeners deliver it.
  later(() => {
    if (!room.listeners.has(id)) return;
    events.connected(true);
    events.lobby(JSON.parse(room.lobby) as LobbyState);
    room.chat.forEach((json) => events.chat(JSON.parse(json)));
  });
  broadcast(room, (l) => l.events.presence(presenceOf(room)));

  return {
    code,
    meta: { ...room.meta },
    updateLobby: async (edit) => {
      requireServer();
      const next = edit(JSON.parse(room.lobby) as LobbyState);
      if (!next) return null;
      room.lobby = JSON.stringify(next);
      broadcast(room, (l) => l.events.lobby(JSON.parse(room.lobby) as LobbyState));
      return JSON.parse(room.lobby) as LobbyState;
    },
    watchMatch: (match, matchEvents) => {
      const watcher: MatchWatcher = { id: server.nextId++, listener: id, events: matchEvents };
      const ws = watchersOf(room, match);
      ws.set(watcher.id, watcher);
      later(() => {
        if (!ws.has(watcher.id)) return;
        const m = room.matches.get(match);
        if (!m) return;
        if (m.start) matchEvents.start(JSON.parse(m.start) as StartRecord);
        [...m.actions.entries()]
          .sort((a, b) => a[0] - b[0])
          .forEach(([seq, value]) => matchEvents.action(seq, decodeAction(value)!));
      });
      const stop = () => {
        ws.delete(watcher.id);
        mine.delete(stop);
      };
      mine.add(stop);
      return stop;
    },
    startMatch: async (match, record) => {
      requireServer();
      const m = matchOf(room, match);
      if (m.start) throw new Error('PERMISSION_DENIED: already started');
      if (me.uuid !== room.meta.hostUuid) throw new Error('PERMISSION_DENIED: not the host');
      m.start = JSON.stringify(record);
      const json = m.start;
      toWatchers(room, match, (w) => w.events.start(JSON.parse(json) as StartRecord));
    },
    appendAction: (match, seq, entry) =>
      new Promise<boolean>((resolve, reject) => {
        const write = () => {
          if (!fakeServer().up) {
            reject(new ServerUnreachableError());
            return;
          }
          const m = room.matches.get(match);
          if (!m?.start) {
            reject(new Error('PERMISSION_DENIED: not started'));
            return;
          }
          if (m.actions.has(seq)) {
            resolve(false);
            return;
          }
          const value = encodeAction(entry);
          m.actions.set(seq, value);
          toWatchers(room, match, (w) => w.events.action(seq, decodeAction(value)!));
          resolve(true);
        };
        if (fakeServer().holdWrites) fakeServer().held.push(write);
        else later(write);
      }),
    sendChat: async (msg) => {
      requireServer();
      const json = JSON.stringify(msg);
      room.chat.push(json);
      broadcast(room, (l) => l.events.chat(JSON.parse(json)));
    },
    leave: () => {
      [...mine].forEach((stop) => stop());
      if (!room.listeners.delete(id)) return;
      broadcast(room, (l) => l.events.presence(presenceOf(room)));
    },
  };
}

export function createFakeTransport(): RoomTransport & { warmUp: () => void } {
  return {
    warmUp: () => {},
    createRoom: async (code, meta, lobby, me, events) => {
      const server = requireServer();
      if (server.rooms.has(code)) return null;
      const room: FakeRoom = {
        meta: { ...meta },
        lobby: JSON.stringify(lobby),
        matches: new Map(),
        chat: [],
        listeners: new Map(),
        watchers: new Map(),
      };
      server.rooms.set(code, room);
      // Like the real transport: the new room's host clears week-old rooms nobody is in.
      const cutoff = Date.now() - ROOM_LIFETIME_MS;
      [...server.rooms.entries()]
        .filter(([, r]) => r.meta.createdAt < cutoff)
        .sort((a, b) => a[1].meta.createdAt - b[1].meta.createdAt)
        .slice(0, SWEEP_BATCH)
        .forEach(([c, r]) => r.listeners.size === 0 && server.rooms.delete(c));
      return open(code, room, me, events);
    },
    joinRoom: async (code, me, events) => {
      const server = requireServer();
      const room = server.rooms.get(code);
      return room ? open(code, room, me, events) : null;
    },
  };
}

let instance: ReturnType<typeof createFakeTransport> | null = null;

/** Same export as firebaseRoom.ts, so the module can be swapped in with vi.mock. */
export function getTransport() {
  return (instance ??= createFakeTransport());
}
