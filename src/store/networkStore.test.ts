import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fakeServer, releaseWrites, resetFakeServer } from '@/net/fakeTransport';
import type { Action, TractorState } from '@/game/types';

// Online play end to end over an in-memory server (src/net/fakeTransport.ts):
// every device is a fresh copy of the stores, as if on its own phone.
vi.mock('@/net/firebaseRoom', () => import('@/net/fakeTransport'));

async function device(uuid: string) {
  vi.resetModules();
  localStorage.setItem('tractor.uuid', uuid);
  const netModule = await import('@/store/networkStore');
  const { useGameStore } = await import('@/store/gameStore');
  const driver = await import('@/ai/driver');
  const engine = await import('@/game/engine');
  await import('@/test/setup'); // each device is a fresh module graph: give its AI levels the tiny test networks
  return {
    uuid,
    net: () => netModule.useNetworkStore.getState(),
    state: () => useGameStore.getState().state as TractorState | null,
    dispatch: (a: Action, ai = false) => useGameStore.getState().dispatch(a, { ai }),
    driver,
    engine,
  };
}
type Device = Awaited<ReturnType<typeof device>>;

/** Let every write in flight (and the events it triggers) land. */
async function flush() {
  for (let round = 0; round < 3; round++) {
    for (let i = 0; i < 30; i++) await Promise.resolve();
    await new Promise((r) => setTimeout(r, 0));
  }
}

function memoryStorage(): Storage {
  const items = new Map<string, string>();
  return {
    get length() {
      return items.size;
    },
    clear: () => items.clear(),
    getItem: (k) => items.get(k) ?? null,
    key: (i) => [...items.keys()][i] ?? null,
    removeItem: (k) => void items.delete(k),
    setItem: (k, v) => void items.set(k, String(v)),
  };
}

beforeEach(() => {
  resetFakeServer();
  vi.stubGlobal('localStorage', memoryStorage());
});
afterEach(() => {
  vi.unstubAllGlobals();
});

async function table() {
  const host = await device('host-uuid');
  await host.net().createRoom('Hana');
  await flush();
  const code = host.net().roomCode!;
  const guest = await device('guest-uuid');
  await guest.net().joinRoom(code, 'Gus');
  await flush();
  host.net().hostAddAISeat();
  host.net().hostAddAISeat();
  await flush();
  return { host, guest, code };
}

/**
 * One step of the match: the driving device runs the deal / AI seats; a
 * human seat's decision is made by its owner with the same AI.
 */
async function step(devices: Device[]): Promise<boolean> {
  const host = devices[0]!;
  const s = host.state()!;
  if (s.status !== 'playing') return false;
  const isAI = (seat: number) => s.seats[seat]!.isAI;
  const auto = host.driver.nextAutomatedStep(s, isAI);
  if (auto) {
    expect(host.dispatch(auto.action, true)).toBeNull();
    await flush();
    return true;
  }
  if (s.round.stage === 'roundOver') return false;
  // A human seat's turn to draw: its own device draws.
  const drawer = host.engine.drawingSeat(s);
  if (drawer !== null) {
    const owner = devices.find((d) => d.uuid === s.seats[drawer]!.id)!;
    expect(owner.dispatch({ type: 'draw', seat: drawer })).toBeNull();
    await flush();
    return true;
  }
  const seat = host.engine.activeSeat(s);
  if (seat === null) return false;
  const owner = devices.find((d) => d.uuid === s.seats[seat]!.id)!;
  const { chooseAIAction } = await import('@/ai');
  const action = chooseAIAction(owner.state()!, seat);
  expect(action).not.toBeNull();
  expect(owner.dispatch(action!)).toBeNull();
  await flush();
  return true;
}

describe('online tractor', () => {
  it('seats a guest and two AIs, and lets the host reorder the table', async () => {
    const { host, guest } = await table();
    expect(host.net().lobby!.seats.map((s) => s.name)).toEqual(['Hana', 'Gus', 'AI 1', 'AI 2']);
    host.net().hostMoveSeat('guest-uuid', 1);
    await flush();
    expect(guest.net().lobby!.seats.map((s) => s.name)).toEqual(['Hana', 'AI 1', 'Gus', 'AI 2']);
  });

  it('plays a whole round with every device in step', async () => {
    const { host, guest } = await table();
    expect(await host.net().hostStartMatch()).toBeNull();
    await flush();
    expect(host.net().connection).toBe('in-game');
    expect(guest.net().connection).toBe('in-game');
    expect(host.net().aiDriver).toBe(true);
    expect(guest.net().aiDriver).toBe(false);
    expect(guest.state()!.rules.biddingStyle).toBe('live');
    let guard = 0;
    while (host.state()!.history.length === 0 && guard++ < 2000) {
      if (!(await step([host, guest]))) break;
    }
    expect(host.state()!.history.length).toBe(1);
    expect(JSON.stringify(guest.state())).toBe(JSON.stringify(host.state()));
  }, 60_000);

  it('a guest bid racing the deal is replayed on top of it', async () => {
    const { host, guest } = await table();
    await host.net().hostStartMatch();
    await flush();
    // Deal (each seat's own device draws for it) until the guest can bid.
    const drawNext = async () => {
      const seat = host.engine.drawingSeat(host.state()!)!;
      (seat === 1 ? guest : host).dispatch({ type: 'draw', seat }, seat !== 1);
      await flush();
    };
    let guard = 0;
    while (guest.engine.validBidsFor(guest.state()!, 1).length === 0 && guard++ < 120) await drawNext();
    // Line the deal up so the host's next draw is an AI seat's.
    while (host.engine.drawingSeat(host.state()!) === 1 && guard++ < 200) await drawNext();
    const bid = guest.engine.validBidsFor(guest.state()!, 1)[0];
    if (!bid) return; // no bidding card this deal — nothing to race
    fakeServer().holdWrites = true;
    host.dispatch({ type: 'draw', seat: host.engine.drawingSeat(host.state()!)! }, true);
    guest.dispatch({ type: 'bid', seat: 1, card: bid.card, count: bid.count });
    releaseWrites();
    await flush();
    expect(JSON.stringify(guest.state())).toBe(JSON.stringify(host.state()));
    expect(host.state()!.round.bids.some((b) => b.seat === 1)).toBe(true);
  }, 30_000);

  it('another device takes over the table when the host drops', async () => {
    const { host, guest } = await table();
    await host.net().hostStartMatch();
    await flush();
    host.net().leaveRoom();
    await flush();
    expect(guest.net().aiDriver).toBe(true);
  });
});
