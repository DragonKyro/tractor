// One playing card, drawn from the SVG faces in public/cards (Adrian
// Kennard's CC0 deck, via the shengji project). `card` null draws the back.

import type { MouseEvent } from 'react';
import { cardLabel, type Card } from '@/game/cards';
import { usePrefs } from '@/store/prefsStore';

export type CardSize = 'lg' | 'md' | 'sm' | 'xs';

export function cardImageUrl(card: Card | null, fourColor: boolean): string {
  const file = card === null ? '1B' : card === 'LJ' ? '2J' : card === 'BJ' ? '1J' : card;
  return `${import.meta.env.BASE_URL}${fourColor ? 'cards-4c' : 'cards'}/${file}.svg`;
}

export function CardView({
  card,
  size = 'md',
  selected,
  dim,
  highlight,
  badge,
  onClick,
  title,
}: {
  card: Card | null;
  size?: CardSize;
  selected?: boolean;
  dim?: boolean;
  /** A gold outline (e.g. a trick's winning play, a card from the kitty). */
  highlight?: boolean;
  badge?: string;
  onClick?: (e: MouseEvent<HTMLButtonElement>) => void;
  title?: string;
}) {
  const fourColor = usePrefs((s) => s.fourColor);
  const cls = ['card', `card-${size}`, selected && 'card-selected', dim && 'card-dim', highlight && 'card-highlight', onClick && 'card-clickable']
    .filter(Boolean)
    .join(' ');
  const label = card === null ? 'face-down card' : cardLabel(card);
  const img = <img src={cardImageUrl(card, fourColor)} alt={label} draggable={false} />;
  return onClick ? (
    <button type="button" className={cls} onClick={onClick} title={title ?? label} aria-pressed={selected}>
      {img}
      {badge && <span className="card-badge">{badge}</span>}
    </button>
  ) : (
    <span className={cls} title={title ?? label}>
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
        <CardView key={`${c}-${i}`} card={c} size={size} highlight={highlight} />
      ))}
    </span>
  );
}
