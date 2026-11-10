// Text with its suit symbols in their suit colours (the same four colours as
// the cards, lightened for the dark UI; red / black with the classic deck).

import { Fragment } from 'react';
import type { Suit } from '@/game/cards';

const SUIT_OF: Record<string, Suit> = { '♠': 'S', '♥': 'H', '♦': 'D', '♣': 'C' };

export function SuitText({ text }: { text: string }) {
  if (!/[♠♥♦♣]/.test(text)) return <>{text}</>;
  return (
    <>
      {text.split(/([♠♥♦♣])/).map((part, i) => {
        const suit = SUIT_OF[part];
        return suit ? (
          <span key={i} className={`suit suit-${suit}`}>
            {part}
          </span>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        );
      })}
    </>
  );
}
