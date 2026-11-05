// Keeps this device's game in step with a match's action log.
//
// The server's log (write-once slots 0, 1, 2, …) is the single order every
// device applies moves in. This device's own moves are applied at once (no
// waiting on the network) and queued as *pending*; one at a time, the head of
// the queue is written to the next free slot. When its slot comes back in the
// log the move is confirmed. When another device's move lands in that slot
// first (two players acting at the same moment: simultaneous picks, a
// click-through screen), this device rolls back to just before its pending
// moves, applies the other move, and replays its own on top: the ones still
// legal are written again, the rest are dropped.
//
// Invariant: with moves pending, the confirmed state (the log applied in
// order) is `pending[0].before`; with none pending, it's the live state.
//
// Game-agnostic: actions are opaque, and the state is whatever `snapshot`
// returns (game states are immutable, so a snapshot is just a reference).

import type { LoggedAction } from '@/net/types';

export interface ActionSyncDeps<S> {
  /** Apply a confirmed move from the log. */
  apply: (entry: LoggedAction) => void;
  /** Re-apply one of this device's pending moves after a rollback; false when it's no longer legal. */
  reapply: (entry: LoggedAction) => boolean;
  /** The live state, to roll back to. */
  snapshot: () => S;
  restore: (snapshot: S) => void;
  /** Claim log slot `seq` (true), or find it taken (false). Rejects when the server refuses. */
  write: (seq: number, entry: LoggedAction) => Promise<boolean>;
  /** One of this device's moves didn't make it into the game. `quiet`: it was
   *  marked so when made (an AI move — nobody to tell; the AI decides again). */
  dropped: (action: unknown, why: 'overtaken' | 'refused', quiet: boolean) => void;
}

interface Pending<S> {
  entry: LoggedAction;
  before: S;
  quiet: boolean;
}

const RETRY_MS = 1000;

/** The server said no (security rules), as opposed to a failed connection. */
function isRefusal(err: unknown): boolean {
  const e = err as { code?: unknown; message?: unknown } | null;
  return /permission.denied/i.test(`${String(e?.code ?? '')} ${String(e?.message ?? err)}`);
}

/** 8 random base-36 characters: enough to tell this device's next move from anyone else's. */
function newNonce(): string {
  return Math.random().toString(36).slice(2, 10).padEnd(8, '0');
}

export class ActionSync<S> {
  private confirmed = 0;
  private started = false;
  private closed = false;
  private readonly inbox = new Map<number, LoggedAction>();
  private pending: Pending<S>[] = [];
  private inFlight: number | null = null;
  private writeToken = 0;

  constructor(private readonly deps: ActionSyncDeps<S>) {}

  /** Log entries applied so far. */
  get confirmedCount(): number {
    return this.confirmed;
  }

  /** This device's moves not yet in the log. */
  get pendingCount(): number {
    return this.pending.length;
  }

  /** The start state is in place: apply the log (entries that arrived early were held). */
  begin(): void {
    if (this.started) return;
    this.started = true;
    this.drain();
  }

  close(): void {
    this.closed = true;
    this.pending = [];
    this.inbox.clear();
  }

  /** A move this device just applied; `before` is the state it was applied to.
   *  `quiet` moves (the AI's) are dropped without telling anyone. */
  local(action: unknown, before: S, quiet = false): void {
    if (this.closed || !this.started) return;
    this.pending.push({ entry: { action, nonce: newNonce() }, before, quiet });
    this.pump();
  }

  /** Log entry `seq` arrived from the server. */
  receive(seq: number, entry: LoggedAction): void {
    if (this.closed || seq < this.confirmed) return;
    this.inbox.set(seq, entry);
    this.drain();
  }

  private drain(): void {
    if (!this.started || this.closed) return;
    while (this.inbox.has(this.confirmed)) {
      const entry = this.inbox.get(this.confirmed)!;
      this.inbox.delete(this.confirmed);
      this.confirmed++;
      const head = this.pending[0];
      if (head && head.entry.nonce === entry.nonce) {
        // Ours, already applied here.
        this.pending.shift();
        this.inFlight = null;
        continue;
      }
      if (!head) {
        this.deps.apply(entry);
        continue;
      }
      // Another device's move came first: roll back, apply it, replay ours.
      const replay = this.pending;
      this.pending = [];
      this.inFlight = null;
      this.deps.restore(head.before);
      this.deps.apply(entry);
      this.replay(replay);
    }
    this.pump();
  }

  private replay(moves: Pending<S>[]): void {
    for (const move of moves) {
      const before = this.deps.snapshot();
      if (this.deps.reapply(move.entry)) {
        this.pending.push({ entry: { ...move.entry, nonce: newNonce() }, before, quiet: move.quiet });
      } else {
        this.deps.dropped(move.entry.action, 'overtaken', move.quiet);
      }
    }
  }

  /** Write the head of the queue to the next slot, one write at a time. */
  private pump(): void {
    if (!this.started || this.closed || this.inFlight !== null) return;
    const head = this.pending[0];
    if (!head) return;
    const token = ++this.writeToken;
    this.inFlight = token;
    this.deps.write(this.confirmed, head.entry).then(
      // Written: the log listener confirms it. Taken: the other device's
      // entry is on its way and its arrival triggers the rollback.
      () => {},
      (err: unknown) => {
        if (this.closed || this.inFlight !== token || this.pending[0] !== head) return;
        if (!isRefusal(err)) {
          // A network hiccup, not a "no": try the same move again shortly.
          this.inFlight = null;
          setTimeout(() => this.pump(), RETRY_MS);
          return;
        }
        console.warn('[net] move refused by the server:', (err as Error)?.message ?? err);
        const rest = this.pending.slice(1);
        this.pending = [];
        this.inFlight = null;
        this.deps.restore(head.before);
        this.deps.dropped(head.entry.action, 'refused', head.quiet);
        this.replay(rest);
        this.pump();
      },
    );
  }
}
