// The viewer's hand: sorted by suit with trump last, grouped with small
// gaps, overlapped to fit the width (wrapping into extra rows when the hand
// is big) and fanned slightly. Tap a card to select it, or press and drag
// across several; on a focused card Enter / Space toggles it. Newly drawn
// cards drop in; when the order changes (trump declared, a gap closing after
// a play) the cards slide to their new places.

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type PointerEvent } from 'react';
import type { Card } from '@/game/cards';
import { effectiveSuit, type Trump } from '@/game/trump';
import { CardView } from '../CardView';
import { prefersReducedMotion, registerHand } from './flight';

interface Props {
  cards: Card[];
  trump: Trump;
  selected: ReadonlySet<number>;
  /** Set to make the hand selectable. */
  onSelect?: (next: Set<number>) => void;
  /** Indices to mark (e.g. cards that came from the kitty). */
  marked?: ReadonlySet<number>;
  markLabel?: string;
}

function coarsePointer(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(pointer: coarse)').matches;
}

const SLIDE_MS = 340;
/** The fan's widest tilt, in degrees. */
const MAX_FAN = 6;

export function Hand({ cards, trump, selected, onSelect, marked, markLabel }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(800);
  const [cardW, setCardW] = useState(72);
  const drag = useRef<{ mode: boolean; touched: Set<number>; current: Set<number> } | null>(null);
  const minStep = coarsePointer() ? 30 : 22;
  const slots = useRef(new Map<string, HTMLElement>());
  const prevPos = useRef(new Map<string, [number, number]>());
  const firstPaint = useRef(true);
  /** Each card's drop-in delay, fixed when it mounts: staggered for the cards of the first paint, none for cards drawn later. */
  const entryDelay = useRef(new Map<string, number>());

  // Stable keys per physical card (card + which copy), so a new card mounts — and drops in — where it sorts.
  const seen = new Map<Card, number>();
  const keys = cards.map((c) => {
    const k = seen.get(c) ?? 0;
    seen.set(c, k + 1);
    return `${c}#${k}`;
  });
  const latest = useRef({ cards, keys, selected });
  latest.current = { cards, keys, selected };

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      setWidth(el.clientWidth);
      const px = parseFloat(getComputedStyle(el).getPropertyValue('--card-w-md'));
      if (px > 0) setCardW(px);
    };
    measure();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, []);

  // Where the selected cards are on screen, for the flight to the table when they're played.
  useEffect(
    () =>
      registerHand(() => {
        const { cards: cs, keys: ks, selected: sel } = latest.current;
        const out: Array<[Card, DOMRect]> = [];
        for (const i of sel) {
          const card = slots.current.get(ks[i]!)?.querySelector('.card');
          if (card) out.push([cs[i]!, card.getBoundingClientRect()]);
        }
        return out;
      }),
    [],
  );

  // Slide cards whose place in the layout changed (offsets ignore transforms, so the fan and lifts don't count).
  useLayoutEffect(() => {
    firstPaint.current = false;
    const next = new Map<string, [number, number]>();
    const moves: Array<{ el: HTMLElement; dx: number; dy: number; from: [number, number] }> = [];
    for (const [key, el] of slots.current) {
      const pos: [number, number] = [el.offsetLeft, el.offsetTop];
      next.set(key, pos);
      const p = prevPos.current.get(key);
      if (p && (Math.abs(p[0] - pos[0]) > 1 || Math.abs(p[1] - pos[1]) > 1)) moves.push({ el, dx: p[0] - pos[0], dy: p[1] - pos[1], from: [0, 0] });
    }
    prevPos.current = next;
    if (moves.length === 0 || prefersReducedMotion() || typeof Element === 'undefined' || !Element.prototype.animate) return;
    // Continue from wherever a running slide has the card now.
    for (const m of moves) {
      if (!m.el.getAnimations().some((a) => a.id === 'slide')) continue;
      const [x, y] = getComputedStyle(m.el).translate.split(' ').map((v) => parseFloat(v) || 0);
      m.from = [x ?? 0, y ?? 0];
    }
    for (const m of moves) {
      for (const a of m.el.getAnimations()) if (a.id === 'slide') a.cancel();
      const anim = m.el.animate(
        [{ translate: `${m.dx + m.from[0]}px ${m.dy + m.from[1]}px` }, { translate: '0px 0px' }],
        { duration: SLIDE_MS, easing: 'cubic-bezier(.2,.8,.2,1)' },
      );
      anim.id = 'slide';
    }
  });

  const indexAt = (x: number, y: number): number | null => {
    const el = document.elementFromPoint(x, y)?.closest('[data-hand-index]');
    return el ? Number(el.getAttribute('data-hand-index')) : null;
  };
  const apply = (i: number) => {
    const d = drag.current!;
    if (d.mode) d.current.add(i);
    else d.current.delete(i);
    onSelect?.(new Set(d.current));
  };
  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (!onSelect || e.button > 0) return;
    const i = indexAt(e.clientX, e.clientY);
    if (i === null) return;
    e.preventDefault();
    drag.current = { mode: !selected.has(i), touched: new Set([i]), current: new Set(selected) };
    apply(i);
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    const i = indexAt(e.clientX, e.clientY);
    if (i === null || d.touched.has(i)) return;
    d.touched.add(i);
    apply(i);
  };
  const endDrag = () => {
    drag.current = null;
  };

  if (cards.length === 0) return <div className="hand hand-empty" ref={ref}>No cards in hand.</div>;

  // Split into rows when one row would overlap too tightly. The room left for the cards is the width less
  // the tray's padding and what the outermost cards' tilt sticks out.
  const overhang = Math.ceil(cardW * 1.55 * 1.15 * Math.sin((MAX_FAN * Math.PI) / 180));
  const room = width - 16 - 2 * overhang;
  let rows = 1;
  const stepFor = (perRow: number, groups: number) => (room - cardW - groups * 10) / Math.max(1, perRow - 1);
  const groupsIn = (list: number[]) => new Set(list.map((i) => effectiveSuit(trump, cards[i]!))).size - 1;
  const all = cards.map((_, i) => i);
  while (rows < 4 && stepFor(Math.ceil(cards.length / rows), groupsIn(all)) < minStep) rows++;
  const perRow = Math.ceil(cards.length / rows);
  const rowLists: number[][] = [];
  for (let r = 0; r < rows; r++) rowLists.push(all.slice(r * perRow, (r + 1) * perRow));

  return (
    <div
      className={`hand${onSelect ? ' hand-selectable' : ''}`}
      ref={ref}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      {rowLists.map((list, ri) => {
        const groups = groupsIn(list);
        const step = Math.min(cardW * 0.62, Math.max(12, stepFor(list.length, groups)));
        // A gentle fan: tilt out from the middle, dipping at the ends.
        const mid = (list.length - 1) / 2;
        const spread = Math.min(MAX_FAN, list.length * 0.55);
        return (
          <div className="hand-row" key={ri}>
            {list.map((i, k) => {
              const prev = k > 0 ? list[k - 1]! : null;
              const newGroup = prev !== null && effectiveSuit(trump, cards[prev]!) !== effectiveSuit(trump, cards[i]!);
              const marginLeft = k === 0 ? 0 : -(cardW - step) + (newGroup ? 10 : 0);
              const t = mid > 0 ? (k - mid) / mid : 0;
              const style = {
                marginLeft,
                zIndex: k,
                '--fan-r': `${(t * spread).toFixed(2)}deg`,
                '--fan-y': `${(t * t * spread * 1.3).toFixed(1)}px`,
              } as CSSProperties;
              const key = keys[i]!;
              if (!entryDelay.current.has(key)) entryDelay.current.set(key, firstPaint.current ? Math.min(i, 40) * 18 : 0);
              (style as Record<string, string>)['--delay'] = `${entryDelay.current.get(key)}ms`;
              return (
                <span
                  key={key}
                  className="hand-slot"
                  data-hand-index={i}
                  style={style}
                  ref={(el) => {
                    if (el) slots.current.set(key, el);
                    else slots.current.delete(key);
                  }}
                >
                  {/* A stable wrapper carries the drop-in: the card inside switches between button and span with the turn. */}
                  <span className="hand-card">
                  <CardView
                    card={cards[i]!}
                    size="md"
                    selected={selected.has(i)}
                    trump={effectiveSuit(trump, cards[i]!) === 'T'}
                    onClick={
                      onSelect
                        ? (e) => {
                            // Pointer presses are handled above; this is keyboard activation.
                            if (e.detail !== 0) return;
                            const next = new Set(selected);
                            if (next.has(i)) next.delete(i);
                            else next.add(i);
                            onSelect(next);
                          }
                        : undefined
                    }
                    highlight={marked?.has(i)}
                    badge={marked?.has(i) ? markLabel : undefined}
                  />
                  </span>
                </span>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
