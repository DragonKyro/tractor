// What a room stores and what devices exchange through it. Every payload is
// JSON-serializable; the engine never holds a function in its state.

import type { Seat } from '@/core/types';

export type ConnectionState = 'disconnected' | 'connecting' | 'lobby' | 'in-game' | 'error';

/**
 * solo: no room (hot-seat / AI play). host: made the room, runs the AI seats.
 * guest: in the room with a seat. spectator: in the room without one (read-only).
 */
export type Role = 'solo' | 'host' | 'guest' | 'spectator';

/**
 * The room's lobby. Edited by transaction, so guests seat themselves while the
 * host edits the rest; between matches it's also where the next match is set up.
 */
export interface LobbyState {
  hostUuid: string;
  /** The match options (a TractorConfig), as JSON. */
  configJson: string;
  /** A human seat's id is its owner's UUID; an AI seat's id is `ai-N`. */
  seats: Seat[];
  /** The match being played (its slot under `matches/`), null while everyone is in the lobby. */
  match: number | null;
  /** Matches started so far, i.e. the next match's slot. */
  matchCount: number;
}

export interface ChatMessage {
  id: string;
  byUuid: string;
  name: string;
  text: string;
  ts: number;
  /** System lines (joins, leaves, game start) are made on each device, never sent. */
  kind: 'user' | 'system';
}

/** Written once, by the host, when the room is made. */
export interface RoomMeta {
  hostUuid: string;
  createdAt: number;
  /** The app build that made the room (devices on another build may not stay in step). */
  build: string;
}

/**
 * How a match began. It carries createInitialState's inputs rather than the
 * opening state (a few hundred bytes instead of tens of KB): the module's
 * createInitialState is pure, so every device builds the same state from them,
 * then applies the match's action log.
 */
export interface StartRecord {
  configJson: string;
  seed: number;
  seats: Seat[];
  hostUuid: string;
  /** When the host started the match (ms since epoch). */
  startedAt: number;
  /** Checksum of the opening state, so a device that built a different one (another app version) can tell. */
  check: string;
}

/** One move in a match's action log. */
export interface LoggedAction {
  action: unknown;
  /** Short random id, so the author recognises its own move in the log. */
  nonce: string;
}

/** A device connected to the room right now. */
export interface Presence {
  uuid: string;
  name: string;
}
