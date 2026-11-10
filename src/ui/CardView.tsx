// One playing card, drawn by cardArt (the big sizes get the full face, the
// small ones a compact face with a giant index). `card` null draws the back.

import type { CSSProperties, MouseEvent } from 'react';
import { cardLabel, type Card } from '@/game/cards';
import { usePrefs } from '@/store/prefsStore';
import { cardSrc } from './cardArt';

export type CardSize = 'lg' | 'md' | 'sm' | 'xs';

export function CardView({
  card,
  size = 'md',
  selected,
  dim,
  highlight,
  trump,
  badge,
  onClick,
  title,
  style,
}: {
  card: Card | null;
  size?: CardSize;
  selected?: boolean;
  dim?: boolean;
  /** A gold outline (e.g. a trick's winning play, a card from the kitty). */
  highlight?: boolean;
  /** Mark the card as trump (a gold edge along the top). */
  trump?: boolean;
  badge?: string;
  onClick?: (e: MouseEvent<HTMLButtonElement>) => void;
  title?: string;
  style?: CSSProperties;
}) {
  const twoColor = usePrefs((s) => s.twoColor);
  const cls = [
    'card',
    `card-${size}`,
    selected && 'card-selected',
    dim && 'card-dim',
    highlight && 'card-highlight',
    trump && 'card-trump',
    onClick && 'card-clickable',
  ]
    .filter(Boolean)
    .join(' ');
  const label = card === null ? 'face-down card' : cardLabel(card);
  const src = cardSrc(card, size === 'lg' || size === 'md' ? 'full' : 'compact', twoColor ? 'two' : 'four');
  const img = <img src={src} alt={label} draggable={false} />;
  return onClick ? (
    <button type="button" className={cls} onClick={onClick} title={title ?? label} aria-pressed={selected} data-card={card ?? undefined} style={style}>
      {img}
      {badge && <span className="card-badge">{badge}</span>}
    </button>
  ) : (
    <span className={cls} title={title ?? label} data-card={card ?? undefined} style={style}>
      {img}
      {badge && <span className="card-badge">{badge}</span>}
    </span>
  );
}

/** A row of small cards (trick plays, kitty, log details). */
export function CardRow({ cards, size = 'sm', highlight }: { cards: readonly Card[]; size?: CardSize; highlight?: boolean }) {
  return (
    <span className={`card-row card-row-${size}`}>
      {cards.map((c, i) => (
        <CardView key={`${c}-${i}`} card={c} size={size} highlight={highlight} style={{ '--i': i } as CSSProperties} />
      ))}
    </span>
  );
}
