// Seeded RNG (mulberry32). Deterministic — every peer reduces to the same state
// when fed the same actions, so RNG-bearing actions (card draws, shuffles) carry
// their seed inline rather than relying on Math.random().

export type RngState = { seed: number };

export function createRng(seed: number): RngState {
  return { seed: seed >>> 0 };
}

export function rngNext(state: RngState): number {
  state.seed = (state.seed + 0x6d2b79f5) >>> 0;
  let t = state.seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

export function rngInt(state: RngState, maxExclusive: number): number {
  return Math.floor(rngNext(state) * maxExclusive);
}

export function shuffle<T>(state: RngState, arr: readonly T[]): T[] {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = rngInt(state, i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export function randomSeed(): number {
  return (Math.random() * 0x100000000) >>> 0;
}
