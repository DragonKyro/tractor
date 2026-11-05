import { afterEach, describe, expect, it, vi } from 'vitest';
import type { LoggedAction } from '@/net/types';
import { ActionSync } from './actionSync';

// A toy game: the state is the list of moves applied, and a move is legal
// only once (like two players claiming the same thing).
type State = string[];
const move = (tag: string) => ({ type: 'claim', tag });
const tagOf = (a: unknown) => (a as { tag: string }).tag;

interface Write {
  seq: number;
  entry: LoggedAction;
  resolve: (ok: boolean) => void;
  reject: (e: unknown) => void;
}

function harness() {
  let state: State = [];
  const writes: Write[] = [];
  const dropped: [string, string][] = [];
  const legal = (s: State, a: unknown) => !s.includes(tagOf(a));
  const sync = new ActionSync<State>({
    apply: (e) => {
      if (legal(state, e.action)) state = [...state, tagOf(e.action)];
    },
    reapply: (e) => {
      if (!legal(state, e.action)) return false;
      state = [...state, tagOf(e.action)];
      return true;
    },
    snapshot: () => state,
    restore: (s) => {
      state = s;
    },
    write: (seq, entry) => new Promise<boolean>((resolve, reject) => writes.push({ seq, entry, resolve, reject })),
    dropped: (a, why, quiet) => dropped.push([tagOf(a), quiet ? `${why} (quiet)` : why]),
  });
  sync.begin();
  return {
    sync,
    writes,
    dropped,
    state: () => state,
    /** This device makes a move (applied at once, as gameStore.dispatch does). */
    act(tag: string, quiet = false) {
      const before = state;
      state = [...state, tag];
      sync.local(move(tag), before, quiet);
    },
    /** The server log delivers entry `seq` from another device. */
    deliver(seq: number, tag: string) {
      sync.receive(seq, { action: move(tag), nonce: `other-${tag}` });
    },
    /** The server echoes our write `i` back as log entry `seq`. */
    echo(i: number) {
      const w = writes[i]!;
      sync.receive(w.seq, w.entry);
    },
  };
}

const settle = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
};

afterEach(() => {
  vi.useRealTimers();
});

describe('ActionSync', () => {
  it('writes our moves one at a time to the next slot, confirming each from the log', () => {
    const h = harness();
    h.act('a');
    h.act('b');
    expect(h.state()).toEqual(['a', 'b']);
    expect(h.writes.map((w) => w.seq)).toEqual([0]); // b waits for a
    expect(h.writes[0]!.entry.nonce).toMatch(/^[0-9a-z]{8}$/);
    h.echo(0);
    expect(h.writes.map((w) => w.seq)).toEqual([0, 1]);
    h.echo(1);
    expect(h.sync.confirmedCount).toBe(2);
    expect(h.sync.pendingCount).toBe(0);
    expect(h.state()).toEqual(['a', 'b']);
  });

  it("applies other devices' moves in log order, holding early arrivals", () => {
    const h = harness();
    h.deliver(1, 'y');
    expect(h.state()).toEqual([]);
    h.deliver(0, 'x');
    expect(h.state()).toEqual(['x', 'y']);
    h.deliver(0, 'x'); // a repeat is ignored
    expect(h.state()).toEqual(['x', 'y']);
  });

  it('holds the log until the game has started', () => {
    const applied: string[] = [];
    const late = new ActionSync<State>({
      apply: (e) => applied.push(tagOf(e.action)),
      reapply: () => true,
      snapshot: () => [],
      restore: () => {},
      write: async () => true,
      dropped: () => {},
    });
    late.receive(0, { action: move('x'), nonce: 'n' });
    expect(applied).toEqual([]);
    late.begin();
    expect(applied).toEqual(['x']);
  });

  it("replays our move after another device's move takes its slot", async () => {
    const h = harness();
    h.act('mine');
    expect(h.writes[0]!.seq).toBe(0);
    h.deliver(0, 'theirs');
    h.writes[0]!.resolve(false);
    await settle();
    expect(h.state()).toEqual(['theirs', 'mine']);
    expect(h.writes.map((w) => w.seq)).toEqual([0, 1]);
    h.echo(1);
    expect(h.sync.pendingCount).toBe(0);
    expect(h.dropped).toEqual([]);
  });

  it('drops our move when the other device made the same one first', () => {
    const h = harness();
    h.act('claim');
    h.act('after');
    h.deliver(0, 'claim');
    expect(h.state()).toEqual(['claim', 'after']);
    expect(h.dropped).toEqual([['claim', 'overtaken']]);
    expect(h.writes.at(-1)!.seq).toBe(1);
    expect(h.writes.at(-1)!.entry.action).toEqual(move('after'));
  });

  it('a quiet move (an AI move) stays quiet through a replay and when dropped', () => {
    const h = harness();
    h.act('ai-claim', true);
    h.act('ai-next', true);
    h.deliver(0, 'other');
    h.deliver(1, 'ai-claim');
    expect(h.dropped).toEqual([['ai-claim', 'overtaken (quiet)']]);
    expect(h.state()).toEqual(['other', 'ai-claim', 'ai-next']);
  });

  it('rolls back a move the server refuses outright, keeping the ones after it', async () => {
    const h = harness();
    h.act('bad');
    h.act('good');
    h.writes[0]!.reject(new Error('PERMISSION_DENIED: Permission denied'));
    await settle();
    expect(h.dropped).toEqual([['bad', 'refused']]);
    expect(h.state()).toEqual(['good']);
    expect(h.writes.at(-1)!.seq).toBe(0);
    expect(h.writes.at(-1)!.entry.action).toEqual(move('good'));
  });

  it('retries a write that failed for a network reason', async () => {
    vi.useFakeTimers();
    const h = harness();
    h.act('a');
    h.writes[0]!.reject(new Error('disconnect'));
    await vi.advanceTimersByTimeAsync(0);
    expect(h.writes).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(h.writes).toHaveLength(2);
    expect(h.writes[1]!.seq).toBe(0);
    expect(h.dropped).toEqual([]);
    expect(h.state()).toEqual(['a']);
  });

  it('ignores local moves and log entries once closed', () => {
    const h = harness();
    h.sync.close();
    h.act('a');
    h.deliver(0, 'x');
    expect(h.writes).toHaveLength(0);
    expect(h.state()).toEqual(['a']); // the harness applied it; nothing was queued
  });
});
