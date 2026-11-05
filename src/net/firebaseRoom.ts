// The real RoomTransport: Firebase Realtime Database (project
// github-games-e7c7d, shared with the other web apps: catan, card-collecting,
// ...). Loaded lazily (dynamic import from networkStore) so the Firebase SDK
// only downloads when someone goes online.
//
// The database is shared, so everything this app stores lives under one
// top-level key, DB_ROOT ("tractor"), and database.rules.json only covers that
// key (scripts/merge-db-rules.mjs merges it into the live rules on deploy).
// The paths below are relative to it.
//
// Each device signs in anonymously and silently (no account, no prompt); the
// uid exists only so the security rules can tie writes to a device. A room
// lives at rooms/{CODE}:
//
//   meta       { hostUuid, hostUid, createdAt, build, v }  written once, with the room
//   lobby      JSON string of LobbyState                   edited by transaction
//   matches/
//     m{N}/start      JSON string of StartRecord          written once, by the host
//     m{N}/actions/   aNNNNNN -> JSON string {a, n}       write-once log slots
//   chat/      pushId -> JSON string of ChatMessage
//   presence/  pushId -> { uid, uuid, name }               removed by onDisconnect
//
// plus, written atomically with each room, a per-day index for clean-up
// (the scheme catan uses on the same database):
//
//   roomDays/{day}          -> when that (UTC) day ends    anyone can read: no codes
//   roomIndex/{day}/{CODE}  -> true                        readable only a week after the day ends
//
// Rooms are cleared a week after their day ends: each new room's host runs
// sweepStaleRooms over the oldest days. The rules let anyone delete a room
// over a week old that nobody is connected to, and never expose the codes of
// younger rooms. (Query bounds can't be range-checked in rules, hence day
// buckets rather than one index sorted by time.)
//
// Payloads are JSON strings: the database drops empty arrays / objects and
// turns arrays into keyed objects, which would corrupt game data.
//
// Usage is billed by bytes stored and downloaded, plus open connections, not
// by operation count. So payloads carry only what devices read (a start
// record is createInitialState's inputs, not the state), and the database
// connection is closed whenever this device isn't in a room.

import { getApps, initializeApp, type FirebaseApp } from 'firebase/app';
import {
  browserLocalPersistence,
  browserSessionPersistence,
  getAuth,
  indexedDBLocalPersistence,
  initializeAuth,
  inMemoryPersistence,
  signInAnonymously,
  type Auth,
  type Persistence,
} from 'firebase/auth';
import {
  get,
  getDatabase,
  goOffline,
  goOnline,
  limitToFirst,
  onChildAdded,
  onDisconnect,
  onValue,
  orderByValue,
  push,
  query,
  ref,
  remove,
  runTransaction,
  serverTimestamp,
  set,
  update,
  type Database,
  type DatabaseReference,
  type DataSnapshot,
  type Unsubscribe,
} from 'firebase/database';
import { freshTabMode } from './identity';
import {
  DAY_MS,
  ROOM_LIFETIME_MS,
  ServerUnreachableError,
  decodeAction,
  encodeAction,
  matchKey,
  parseJson,
  parseSeqKey,
  seqKey,
  type RoomConnection,
  type RoomEvents,
  type RoomTransport,
} from './transport';
import type { ChatMessage, LobbyState, Presence, RoomMeta, StartRecord } from './types';

import { FIREBASE_CONFIG } from './firebaseConfig';

/**
 * This app's top-level key in the shared database: every path it reads or
 * writes is under it (must match the key in database.rules.json).
 */
export const DB_ROOT = 'tractor';

/** Bump when the room layout changes (also checked by the rules). */
export const PROTOCOL_VERSION = 1;

/** Oldest days looked at per sweep. */
const SWEEP_DAYS = 5;

/** How long to wait for sign-in or the first read before calling the server unreachable. */
const SERVER_TIMEOUT_MS = 15000;

/** This device's estimate of the server's clock offset (local, no network). */
function serverTimeOffset(db: Database): Promise<number> {
  return new Promise((resolve) => {
    onValue(ref(db, '.info/serverTimeOffset'), (s) => resolve(Number(s.val()) || 0), { onlyOnce: true });
  });
}

/** The (UTC) day bucket a room made at server time `serverNow` goes in. */
function dayBucket(serverNow: number): { day: string; endsAt: number } {
  const day = Math.floor(serverNow / DAY_MS);
  return { day: String(day), endsAt: (day + 1) * DAY_MS };
}

/**
 * Delete the rooms of every day (among the SWEEP_DAYS oldest) that ended over
 * a week ago, then the emptied days. A room the rules keep (someone is still
 * connected) keeps its day too, to be tried by a later sweep. Resolves with
 * the codes removed.
 */
export async function sweepStaleRooms(db: Database): Promise<string[]> {
  // A minute's margin so a fast local clock can't ask for a day the rules
  // still consider fresh.
  const cutoff = Date.now() + (await serverTimeOffset(db)) - ROOM_LIFETIME_MS - 60_000;
  const days = await get(query(ref(db, `${DB_ROOT}/roomDays`), orderByValue(), limitToFirst(SWEEP_DAYS)));
  const stale: string[] = [];
  days.forEach((d) => {
    if (d.key && Number(d.val()) <= cutoff) stale.push(d.key);
  });
  const removed: string[] = [];
  for (const day of stale) {
    const codes: string[] = [];
    (await get(ref(db, `${DB_ROOT}/roomIndex/${day}`))).forEach((c) => {
      if (c.key) codes.push(c.key);
    });
    let kept = 0;
    await Promise.all(
      codes.map((code) =>
        update(ref(db, DB_ROOT), { [`rooms/${code}`]: null, [`roomIndex/${day}/${code}`]: null }).then(
          () => removed.push(code),
          () => kept++, // still in use: kept for now
        ),
      ),
    );
    if (kept === 0) await remove(ref(db, `${DB_ROOT}/roomDays/${day}`)).catch(() => {});
  }
  return removed;
}

export interface FirebaseTransport extends RoomTransport {
  /** Start signing in now so the first create / join is quick. Never throws. */
  warmUp: () => void;
}

export interface FirebaseTransportOptions {
  /** A separate Firebase app (tests run several "devices" in one process). */
  appName?: string;
  /** Where the anonymous sign-in is kept. Default: per tab with ?fresh, else per browser. */
  persistence?: 'local' | 'session' | 'memory';
}

function withTimeout<T>(p: Promise<T>, what: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new ServerUnreachableError(`Timed out ${what}.`)), SERVER_TIMEOUT_MS);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

function errorText(e: unknown): string {
  const err = e as { code?: string; message?: string };
  return `${err.code ?? ''} ${err.message ?? String(e)}`;
}

const isPermissionDenied = (e: unknown) => /permission.denied/i.test(errorText(e));

function unreachable(e: unknown): Error {
  if (e instanceof ServerUnreachableError) return e;
  return new ServerUnreachableError(`Could not reach the game server (${errorText(e).trim()}).`);
}

function persistenceFor(kind: FirebaseTransportOptions['persistence']): Persistence | Persistence[] {
  const resolved = kind ?? (typeof window === 'undefined' ? 'memory' : freshTabMode() ? 'session' : 'local');
  if (resolved === 'memory') return inMemoryPersistence;
  if (resolved === 'session') return browserSessionPersistence;
  return [indexedDBLocalPersistence, browserLocalPersistence];
}

export function createFirebaseTransport(opts: FirebaseTransportOptions = {}): FirebaseTransport {
  const name = opts.appName ?? '[DEFAULT]';
  let client: { app: FirebaseApp; auth: Auth; db: Database } | null = null;
  let signIn: Promise<string> | null = null;
  // Rooms open on this transport; the database connection is up only while > 0.
  let openRooms = 0;
  let online = true;

  function connect(): Database {
    const { db } = firebase();
    if (!online) {
      goOnline(db);
      online = true;
    }
    return db;
  }

  function disconnectIfIdle(): void {
    if (openRooms > 0 || !online || !client) return;
    // The server notices the closed socket and runs this device's
    // onDisconnect (dropping its presence entry).
    goOffline(client.db);
    online = false;
  }

  function firebase() {
    if (client) return client;
    const app = getApps().find((a) => a.name === name) ?? initializeApp(FIREBASE_CONFIG, name);
    let auth: Auth;
    try {
      auth = initializeAuth(app, { persistence: persistenceFor(opts.persistence) });
    } catch {
      auth = getAuth(app); // already initialised (hot reload)
    }
    client = { app, auth, db: getDatabase(app) };
    return client;
  }

  /** This device's anonymous uid, signing in silently the first time. */
  function uid(): Promise<string> {
    signIn ??= withTimeout(
      (async () => {
        const { auth } = firebase();
        await auth.authStateReady();
        const user = auth.currentUser ?? (await signInAnonymously(auth)).user;
        return user.uid;
      })(),
      'signing in',
    ).catch((e: unknown) => {
      signIn = null;
      throw unreachable(e);
    });
    return signIn;
  }

  function openRoom(code: string, meta: RoomMeta, myUid: string, me: Presence, events: RoomEvents): RoomConnection {
    const db = connect();
    openRooms++;
    const base = `${DB_ROOT}/rooms/${code}`;
    const offs: Unsubscribe[] = [];
    let presenceRef: DatabaseReference | null = null;
    let closed = false;
    const cancelled = (what: string) => (e: Error) => console.warn(`[net] ${what} listener stopped:`, e.message);

    offs.push(
      onValue(
        ref(db, `${base}/lobby`),
        (s) => {
          const lobby = parseJson<LobbyState>(s.val());
          if (lobby && !closed) events.lobby(lobby);
        },
        cancelled('lobby'),
      ),
    );
    offs.push(
      onChildAdded(
        ref(db, `${base}/chat`),
        (s) => {
          const msg = parseJson<ChatMessage>(s.val());
          if (msg && !closed) events.chat(msg);
        },
        cancelled('chat'),
      ),
    );
    offs.push(
      onValue(
        ref(db, `${base}/presence`),
        (s) => {
          const people: Presence[] = [];
          s.forEach((c: DataSnapshot) => {
            const v = c.val() as { uuid?: unknown; name?: unknown } | null;
            if (v && typeof v.uuid === 'string') people.push({ uuid: v.uuid, name: String(v.name ?? '') });
          });
          if (!closed) events.presence(people);
        },
        cancelled('presence'),
      ),
    );
    // Presence: on every (re)connection add an entry the server deletes when
    // this connection drops.
    offs.push(
      onValue(ref(db, '.info/connected'), (s) => {
        const up = s.val() === true;
        if (closed) return;
        events.connected(up);
        if (!up) return;
        const p = push(ref(db, `${base}/presence`));
        presenceRef = p;
        onDisconnect(p)
          .remove()
          .then(() => {
            if (!closed && presenceRef === p) {
              return set(p, { uid: myUid, uuid: me.uuid, name: me.name });
            }
          })
          .catch((e: Error) => console.warn('[net] presence failed:', e.message));
      }),
    );

    return {
      code,
      meta,
      updateLobby: async (edit) => {
        let result: LobbyState | null = null;
        const res = await runTransaction(ref(db, `${base}/lobby`), (current: unknown) => {
          result = null;
          // Not cached yet: send a placeholder; the server answers with the real value and this re-runs.
          if (current === null) return null;
          const lobby = parseJson<LobbyState>(current);
          if (!lobby) return undefined;
          result = edit(lobby);
          return result ? JSON.stringify(result) : undefined;
        });
        return res.committed ? result : null;
      },
      watchMatch: (match, matchEvents) => {
        const at = `${base}/matches/${matchKey(match)}`;
        let stopped = false;
        const stops = [
          onValue(
            ref(db, `${at}/start`),
            (s) => {
              const start = parseJson<StartRecord>(s.val());
              if (start && !stopped && !closed) matchEvents.start(start);
            },
            cancelled('start'),
          ),
          onChildAdded(
            ref(db, `${at}/actions`),
            (s) => {
              const seq = parseSeqKey(s.key ?? '');
              const entry = decodeAction(s.val());
              if (seq !== null && entry && !stopped && !closed) matchEvents.action(seq, entry);
            },
            cancelled('actions'),
          ),
        ];
        const stop = () => {
          if (stopped) return;
          stopped = true;
          stops.forEach((off) => off());
        };
        offs.push(stop);
        return stop;
      },
      startMatch: async (match, record) => {
        await set(ref(db, `${base}/matches/${matchKey(match)}/start`), JSON.stringify(record));
      },
      appendAction: async (match, seq, entry) => {
        const value = encodeAction(entry);
        const res = await runTransaction(
          ref(db, `${base}/matches/${matchKey(match)}/actions/${seqKey(seq)}`),
          (current: unknown) => (current === null ? value : undefined),
          // Don't show the move locally until the server has it: the log
          // listener is how every device, the author included, learns the order.
          { applyLocally: false },
        );
        return res.committed;
      },
      sendChat: async (msg) => {
        await push(ref(db, `${base}/chat`), JSON.stringify(msg));
      },
      leave: () => {
        if (closed) return;
        closed = true;
        offs.forEach((off) => off());
        const p = presenceRef;
        presenceRef = null;
        openRooms--;
        if (openRooms > 0 && p) {
          // Staying connected for another room: take this presence down by hand.
          onDisconnect(p).cancel().catch(() => {});
          remove(p).catch(() => {});
        }
        disconnectIfIdle();
      },
    };
  }

  return {
    warmUp: () => {
      uid().catch(() => {});
    },

    createRoom: async (code, meta, lobby, me, events) => {
      const myUid = await uid();
      const db = connect();
      const { day, endsAt } = dayBucket(Date.now() + (await serverTimeOffset(db)));
      try {
        // The room and its index entries, atomically.
        await withTimeout(
          update(ref(db, DB_ROOT), {
            [`rooms/${code}`]: {
              meta: {
                hostUuid: meta.hostUuid,
                hostUid: myUid,
                createdAt: serverTimestamp(),
                build: meta.build,
                v: PROTOCOL_VERSION,
              },
              lobby: JSON.stringify(lobby),
            },
            [`roomIndex/${day}/${code}`]: true,
            [`roomDays/${day}`]: endsAt,
          }),
          'creating the room',
        );
      } catch (e) {
        disconnectIfIdle();
        // The rules refuse to overwrite an existing room: the code is taken.
        if (isPermissionDenied(e)) return null;
        throw unreachable(e);
      }
      const room = openRoom(code, meta, myUid, me, events);
      // Housekeeping while connected anyway; never holds up the new room.
      sweepStaleRooms(db).catch((e: Error) => console.warn('[net] stale-room sweep failed:', e.message));
      return room;
    },

    joinRoom: async (code, me, events) => {
      const myUid = await uid();
      const db = connect();
      let snap: DataSnapshot;
      try {
        snap = await withTimeout(get(ref(db, `${DB_ROOT}/rooms/${code}/meta`)), 'looking up the room');
      } catch (e) {
        disconnectIfIdle();
        throw unreachable(e);
      }
      const v = snap.val() as Partial<RoomMeta> | null;
      if (!v || typeof v.hostUuid !== 'string') {
        disconnectIfIdle();
        return null;
      }
      const meta: RoomMeta = {
        hostUuid: v.hostUuid,
        createdAt: typeof v.createdAt === 'number' ? v.createdAt : 0,
        build: typeof v.build === 'string' ? v.build : 'unknown',
      };
      return openRoom(code, meta, myUid, me, events);
    },
  };
}

let defaultTransport: FirebaseTransport | null = null;

/** The app's transport (one Firebase app per page). */
export function getTransport(): FirebaseTransport {
  return (defaultTransport ??= createFirebaseTransport());
}
