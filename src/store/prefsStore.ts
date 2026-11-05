// Per-device display preferences, kept in localStorage. Never part of a match.

import { create } from 'zustand';

export type AiSpeed = 'slow' | 'normal' | 'fast';

interface Prefs {
  /** Four-color suits (♦ orange, ♣ green) instead of red/black. */
  fourColor: boolean;
  aiSpeed: AiSpeed;
  set(patch: Partial<Pick<Prefs, 'fourColor' | 'aiSpeed'>>): void;
}

const KEY = 'tractor.prefs';

function load(): Partial<Prefs> {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<Prefs>;
  } catch {
    return {};
  }
}

export const usePrefs = create<Prefs>((set, get) => ({
  fourColor: false,
  aiSpeed: 'normal',
  ...load(),
  set: (patch) => {
    set(patch);
    try {
      const { fourColor, aiSpeed } = get();
      localStorage.setItem(KEY, JSON.stringify({ fourColor, aiSpeed }));
    } catch {
      /* ignore */
    }
  },
}));

/** Milliseconds to wait before each kind of automated step when a human is watching. */
export function paceMs(pace: 'deal' | 'ai' | 'linger', speed: AiSpeed): number {
  const f = speed === 'slow' ? 1.7 : speed === 'fast' ? 0.45 : 1;
  const base = pace === 'deal' ? 260 : pace === 'ai' ? 650 : 2200;
  return Math.round(base * f);
}
