// The interface between networkStore and the server. `firebaseRoom.ts` is the
// real one (Firebase Realtime Database); `fakeTransport.ts` is an in-memory
// stand-in for tests. Nothing here knows about Tractor's rules.

import type { ChatMessage, LobbyState, LoggedAction, Presence, RoomMeta, StartRecord } from './types';

/** Everything a room pushes to a device. Callbacks may fire before `createRoom` / `joinRoom` resolve. */
export interface RoomEvents {
  lobby: (lobby: LobbyState) => void;
  chat: (msg: ChatMessage) => void;
  /** Everyone connected right now (this device included once it's registered). */
  presence: (people: Presence[]) => void;
  /** This device's link to the server went down / came back. */
  connected: (up: boolean) => void;
}

/** What one match pushes to a device that watches it. */
export interface MatchEvents {
  start: (start: StartRecord) => void;
  /** Log entry `seq` (0-based). The initial load arrives in order; later entries normally do too. */
  action: (seq: number, entry: LoggedAction) => void;
}

export interface RoomConnection {
  readonly code: string;
  readonly meta: RoomMeta;
  /**
   * Edit the lobby atomically: `edit` gets the current lobby (and may be
   * re-run if another device edits at the same moment) and returns the new
   * one, or null to change nothing. Resolves with the committed lobby, or
   * null when `edit` declined.
   */
  updateLobby: (edit: (current: LobbyState) => LobbyState | null) => Promise<LobbyState | null>;
  /** Listen to match `match`'s start record and action log. Returns the way to stop. */
  watchMatch: (match: number, events: MatchEvents) => () => void;
  /** Write match `match`'s start record (host only, once). */
  startMatch: (match: number, record: StartRecord) => Promise<void>;
  /**
   * Claim log slot `seq` of match `match` for `entry`. Resolves true once the
   * server holds it, false when another device's move already took the slot.
   * Rejects when the server refuses the write.
   */
  appendAction: (match: number, seq: number, entry: LoggedAction) => Promise<boolean>;
  sendChat: (msg: ChatMessage) => Promise<void>;
  /** Stop listening and drop this device's presence. */
  leave: () => void;
}

export interface RoomTransport {
  /** Open a new room under `code`. Resolves null when the code is already in use. */
  createRoom: (
    code: string,
    meta: RoomMeta,
    lobby: LobbyState,
    me: Presence,
    events: RoomEvents,
  ) => Promise<RoomConnection | null>;
  /** Join room `code`. Resolves null when there's no such room. */
  joinRoom: (code: string, me: Presence, events: RoomEvents) => Promise<RoomConnection | null>;
}

/** The server couldn't be reached (offline, blocked, or timed out). */
export class ServerUnreachableError extends Error {
  constructor(message = 'Could not reach the game server.') {
    super(message);
    this.name = 'ServerUnreachableError';
  }
}

export const DAY_MS = 24 * 60 * 60 * 1000;
/**
 * Rooms are deleted once the day they were made ended this long ago (so 7-8
 * days after they're made), and only when nobody is connected
 * (database.rules.json uses the same number).
 */
export const ROOM_LIFETIME_MS = 7 * DAY_MS;
/** Stale rooms each new room's host deletes. More than one, so a backlog can't build up. */
export const SWEEP_BATCH = 10;

/** Match slot keys: `m0`, `m1`, … (not bare numbers, which the database would read as array indexes). */
export function matchKey(match: number): string {
  return `m${match}`;
}

/** Log slot keys: `a000042`. Fixed-width so key order is log order. */
export function seqKey(seq: number): string {
  return `a${String(seq).padStart(6, '0')}`;
}

export function parseSeqKey(key: string): number | null {
  return /^a\d{6}$/.test(key) ? Number(key.slice(1)) : null;
}

// A log slot holds one short JSON string: {"a":<action>,"n":<nonce>}. Every
// device downloads every move, so the encoding stays small.
export function encodeAction(entry: LoggedAction): string {
  return JSON.stringify({ a: entry.action, n: entry.nonce });
}

export function decodeAction(value: unknown): LoggedAction | null {
  if (typeof value !== 'string') return null;
  try {
    const v = JSON.parse(value) as { a?: unknown; n?: unknown };
    if (v.a === undefined || typeof v.n !== 'string') return null;
    return { action: v.a, nonce: v.n };
  } catch {
    return null;
  }
}

/** Parse a JSON-string payload, or null when it isn't one. */
export function parseJson<T>(json: unknown): T | null {
  if (typeof json !== 'string') return null;
  try {
    return JSON.parse(json) as T;
  } catch {
    return null;
  }
}
