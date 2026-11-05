// @vitest-environment node
//
// Online play against the real Firebase project (github-games-e7c7d, shared
// with other apps; everything here lives under /tractor), each "device" with
// its own Firebase app and anonymous sign-in. Skipped unless FIREBASE_LIVE=1:
//
//   FIREBASE_LIVE=1 npx vitest run src/store/networkStore.live.test.ts
//
// The rules don't let clients delete fresh rooms, so it prints the room code
// it used; remove it afterwards with
// `MSYS_NO_PATHCONV=1 npx firebase-tools database:remove /tractor/rooms/<CODE> --force`
// (its index entry goes a week later, with the next sweep of its day). Each
// run also leaves a couple of anonymous users in Firebase Auth (harmless).

import { afterAll, describe, expect, it, vi } from 'vitest';
import type { Action, TractorState } from '@/game/types';

const LIVE = process.env.FIREBASE_LIVE === '1';

vi.mock('@/net/firebaseRoom', async (importOriginal) => {
  const real = await importOriginal<typeof import('@/net/firebaseRoom')>();
  let transport: ReturnType<typeof real.createFirebaseTransport> | null = null;
  return {
    ...real,
    getTransport: () =>
      (transport ??= real.createFirebaseTransport({
        appName: `device-${Math.random().toString(36).slice(2)}`,
        persistence: 'memory',
      })),
  };
});

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
vi.stubGlobal('localStorage', memoryStorage());

const run = Date.now().toString(36);
const rooms: string[] = [];

async function device(name: string) {
  vi.resetModules();
  const uuid = `live-${run}-${name}`;
  localStorage.setItem('tractor.uuid', uuid);
  const netModule = await import('@/store/networkStore');
  const { useGameStore } = await import('@/store/gameStore');
  const driver = await import('@/ai/driver');
  const ai = await import('@/ai');
  const engine = await import('@/game/engine');
  return {
    uuid,
    net: () => netModule.useNetworkStore.getState(),
    state: () => useGameStore.getState().state as TractorState | null,
    dispatch: (a: Action, fromAI = false) => useGameStore.getState().dispatch(a, { ai: fromAI }),
    driver,
    ai,
    engine,
  };
}

const until = (f: () => void, timeout = 20000) => vi.waitFor(f, { timeout, interval: 50 });

afterAll(() => {
  if (rooms.length) console.log(`[live] rooms used: ${rooms.join(', ')} (remove with firebase-tools database:remove /tractor/rooms/<CODE>)`);
});

describe.skipIf(!LIVE)('online tractor on the real Firebase project', () => {
  it('hosts, joins, starts and plays a dozen moves with both devices in step', async () => {
    const host = await device('host');
    await host.net().createRoom('Hana');
    await until(() => expect(host.net().roomCode).toBeTruthy());
    const code = host.net().roomCode!;
    rooms.push(code);
    const guest = await device('guest');
    await guest.net().joinRoom(code, 'Gus');
    await until(() => expect(host.net().lobby!.seats.length).toBe(2));
    host.net().hostAddAISeat();
    host.net().hostAddAISeat();
    await until(() => expect(guest.net().lobby!.seats.length).toBe(4));
    expect(await host.net().hostStartMatch()).toBeNull();
    await until(() => expect(guest.net().connection).toBe('in-game'));
    await until(() => expect(host.net().aiDriver).toBe(true));

    // Twelve steps: the host drives (deal / AI), human seats move on their own device.
    for (let i = 0; i < 12; i++) {
      const s = host.state()!;
      const auto = host.driver.nextAutomatedStep(s, (seat) => s.seats[seat]!.isAI);
      const before = host.state()!.logCount;
      if (auto) {
        expect(host.dispatch(auto.action, true)).toBeNull();
      } else {
        const seat = host.engine.activeSeat(s)!;
        const owner = [host, guest].find((d) => d.uuid === s.seats[seat]!.id)!;
        const a = owner.ai.chooseAIAction(owner.state()!, seat, 'normal')!;
        expect(owner.dispatch(a)).toBeNull();
      }
      await until(() => {
        expect(host.state()!.logCount).toBeGreaterThanOrEqual(before);
        expect(JSON.stringify(guest.state())).toBe(JSON.stringify(host.state()));
      });
    }
    host.net().leaveRoom();
    guest.net().leaveRoom();
  }, 120_000);
});
