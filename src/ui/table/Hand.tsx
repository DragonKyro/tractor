// The viewer's hand: sorted by suit with trump last, grouped with small
// gaps, overlapped to fit the width (wrapping into extra rows when the hand
// is big). Tap a card to select it, or press and drag across several; on a
// focused card Enter / Space toggles it. Newly drawn cards pop in.

import { useLayoutEffect, useRef, useState, type PointerEvent } from 'react';
import type { Card } from '@/game/cards';
import { effectiveSuit, type Trump } from '@/game/trump';
import { CardView } from '../CardView';

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

export function Hand({ cards, trump, selected, onSelect, marked, markLabel }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(800);
  const [cardW, setCardW] = useState(72);
  const drag = useRef<{ mode: boolean; touched: Set<number>; current: Set<number> } | null>(null);
  const minStep = coarsePointer() ? 30 : 22;

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

  // Split into rows when one row would overlap too tightly.
  let rows = 1;
  const stepFor = (perRow: number, groups: number) => (width - cardW - groups * 10) / Math.max(1, perRow - 1);
  const groupsIn = (list: number[]) => new Set(list.map((i) => effectiveSuit(trump, cards[i]!))).size - 1;
  const all = cards.map((_, i) => i);
  while (rows < 4 && stepFor(Math.ceil(cards.length / rows), groupsIn(all)) < minStep) rows++;
  const perRow = Math.ceil(cards.length / rows);
  const rowLists: number[][] = [];
  for (let r = 0; r < rows; r++) rowLists.push(all.slice(r * perRow, (r + 1) * perRow));

  // Stable keys per physical card (card + which copy), so a new card mounts — and pops in — where it sorts.
  const seen = new Map<Card, number>();
  const keys = cards.map((c) => {
    const k = seen.get(c) ?? 0;
    seen.set(c, k + 1);
    return `${c}#${k}`;
  });

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
        return (
          <div className="hand-row" key={ri}>
            {list.map((i, k) => {
              const prev = k > 0 ? list[k - 1]! : null;
              const newGroup = prev !== null && effectiveSuit(trump, cards[prev]!) !== effectiveSuit(trump, cards[i]!);
              const marginLeft = k === 0 ? 0 : -(cardW - step) + (newGroup ? 10 : 0);
              return (
                <span key={keys[i]} className="hand-slot" data-hand-index={i} style={{ marginLeft, zIndex: k }}>
                  <CardView
                    card={cards[i]!}
                    size="md"
                    selected={selected.has(i)}
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
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
