// Cards flying from the viewer's hand to the table. Right before a play is
// dispatched, the hand's selected cards are measured (captureLaunch); when the
// play shows up on the felt in the same update, each card starts its
// animation from where it sat in the hand (takeLaunch). UI only.

import type { Card } from '@/game/cards';

type Measure = () => Array<[Card, DOMRect]>;

let measureHand: Measure | null = null;
let pending: { seat: number; at: number; items: Array<[Card, DOMRect]> } | null = null;

/** Our window for a launch to be picked up (the play lands in the same React update, well within this). */
const FRESH_MS = 1500;

/** The hand registers how to measure its selected cards; returns the unregister function. */
export function registerHand(fn: Measure): () => void {
  measureHand = fn;
  return () => {
    if (measureHand === fn) measureHand = null;
  };
}

/** Note where the selected cards are, just before `seat` plays them. */
export function captureLaunch(seat: number): void {
  pending = measureHand ? { seat, at: performance.now(), items: measureHand() } : null;
}

export function hasLaunch(seat: number): boolean {
  return !!pending && pending.seat === seat && performance.now() - pending.at < FRESH_MS && pending.items.length > 0;
}

/** Where this card of `seat`'s play left the hand from (each copy is taken once), or null. */
export function takeLaunch(seat: number, card: Card): DOMRect | null {
  if (!hasLaunch(seat)) return null;
  const i = pending!.items.findIndex(([c]) => c === card);
  if (i < 0) return null;
  return pending!.items.splice(i, 1)[0]![1];
}

export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}
