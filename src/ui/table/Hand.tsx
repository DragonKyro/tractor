// The viewer's hand: sorted by suit with trump last, grouped with small
// gaps, overlapped to fit the width (wrapping into extra rows when the hand
// is big), tap to select.

import { useLayoutEffect, useRef, useState } from 'react';
import type { Card } from '@/game/cards';
import { effectiveSuit, type Trump } from '@/game/trump';
import { CardView } from '../CardView';

interface Props {
  cards: Card[];
  trump: Trump;
  selected: ReadonlySet<number>;
  onToggle?: (index: number) => void;
  /** Indices to mark (e.g. cards that came from the kitty). */
  marked?: ReadonlySet<number>;
  markLabel?: string;
}

const MIN_STEP = 22;

export function Hand({ cards, trump, selected, onToggle, marked, markLabel }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(800);
  const [cardW, setCardW] = useState(72);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      setWidth(el.clientWidth);
      const v = getComputedStyle(el).getPropertyValue('--card-w-md');
      const px = parseFloat(v);
      if (px > 0) setCardW(px);
    };
    measure();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, []);

  if (cards.length === 0) return <div className="hand hand-empty" ref={ref}>No cards in hand.</div>;

  // Split into rows when one row would overlap too tightly.
  let rows = 1;
  const stepFor = (perRow: number, groups: number) => (width - cardW - groups * 10) / Math.max(1, perRow - 1);
  const groupsIn = (list: number[]) => new Set(list.map((i) => effectiveSuit(trump, cards[i]!))).size - 1;
  const all = cards.map((_, i) => i);
  while (rows < 3 && stepFor(Math.ceil(cards.length / rows), groupsIn(all)) < MIN_STEP) rows++;
  const perRow = Math.ceil(cards.length / rows);
  const rowLists: number[][] = [];
  for (let r = 0; r < rows; r++) rowLists.push(all.slice(r * perRow, (r + 1) * perRow));

  return (
    <div className="hand" ref={ref}>
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
                <span key={i} className="hand-slot" style={{ marginLeft, zIndex: k }}>
                  <CardView
                    card={cards[i]!}
                    size="md"
                    selected={selected.has(i)}
                    onClick={onToggle ? () => onToggle(i) : undefined}
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
