// Per-device preferences, kept in localStorage. Never part of a match: two
// devices at one online table can deal / draw at their own pace settings
// (the driving device's settings pace the AI seats).

import { create } from 'zustand';

export type AiSpeed = 'slow' | 'normal' | 'fast' | 'instant';
export type DealSpeed = 'slow' | 'normal' | 'fast';
/** How long an AI waits after the deal before taking the kitty, so humans can still overcall. */
export type BidWindow = 'short' | 'normal' | 'long';

export interface PrefValues {
  /** Classic red / black suits instead of the four-colour deck (♠ ink, ♥ red, ♦ blue, ♣ green). */
  twoColor: boolean;
  aiSpeed: AiSpeed;
  dealSpeed: DealSpeed;
  bidWindow: BidWindow;
  /** Draw your cards automatically during a live deal (off: click Draw for each card). */
  autoDraw: boolean;
  sounds: boolean;
  /** Watching an all-AI table: show every hand face up. */
  showAllHands: boolean;
}

interface Prefs extends PrefValues {
  set(patch: Partial<PrefValues>): void;
}

const KEY = 'tractor.prefs';
const DEFAULTS: PrefValues = {
  twoColor: false,
  aiSpeed: 'normal',
  dealSpeed: 'normal',
  bidWindow: 'normal',
  autoDraw: true,
  sounds: true,
  showAllHands: false,
};

function load(): Partial<PrefValues> {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? '{}') as Record<string, unknown>;
    // Only known keys (older builds stored others, e.g. `fourColor`).
    return Object.fromEntries(Object.entries(saved).filter(([k]) => k in DEFAULTS)) as Partial<PrefValues>;
  } catch {
    return {};
  }
}

export const usePrefs = create<Prefs>((set, get) => ({
  ...DEFAULTS,
  ...load(),
  set: (patch) => {
    set(patch);
    try {
      const { set: _omit, ...values } = get();
      localStorage.setItem(KEY, JSON.stringify(values));
    } catch {
      /* ignore */
    }
  },
}));

const AI_MS: Record<AiSpeed, number> = { slow: 1100, normal: 650, fast: 300, instant: 40 };
const DEAL_MS: Record<DealSpeed, number> = { slow: 420, normal: 180, fast: 60 };
const WINDOW_MS: Record<BidWindow, number> = { short: 3000, normal: 6000, long: 12000 };

/** Milliseconds to wait before an automated step of this kind. */
export function paceMs(pace: 'deal' | 'ai' | 'lead' | 'bidWindow' | 'roundEnd', p: Pick<PrefValues, 'aiSpeed' | 'dealSpeed' | 'bidWindow'>): number {
  switch (pace) {
    case 'deal':
      return DEAL_MS[p.dealSpeed];
    case 'ai':
      return AI_MS[p.aiSpeed];
    case 'lead':
      // The trick just won stays in view a moment longer.
      return p.aiSpeed === 'instant' ? 40 : AI_MS[p.aiSpeed] + 700;
    case 'bidWindow':
      return WINDOW_MS[p.bidWindow];
    case 'roundEnd':
      return p.aiSpeed === 'instant' ? 300 : 1500;
  }
}
