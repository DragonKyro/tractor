// The deck's art, drawn in code: each card is a small standalone SVG (cached
// as a data URL, so a card on screen is one <img>). Two layouts — `full`
// (pips, court portraits, jesters) for the big sizes and `compact` (a giant
// index and a big suit) for the small ones, where pips would be specks — and
// two colour schemes: four-colour suits (♠ ink, ♥ red, ♦ blue, ♣ green) or
// the classic red / black. Rank glyphs are stroked paths, so they look the
// same on every system.

import { BIG_JOKER, SMALL_JOKER, cardRank, cardSuit, type Card, type Rank, type Suit } from '@/game/cards';

export type CardLayout = 'full' | 'compact';
export type SuitColors = 'four' | 'two';

/** viewBox is 0 0 W H; the ratio matches the layout's --card-ratio. */
const W = 100;
const H = 155;
const CX = W / 2;
const CY = H / 2;

export const SUIT_INK: Record<SuitColors, Record<Suit, string>> = {
  four: { S: '#1d2230', H: '#d92f3c', D: '#1c6ce0', C: '#11904a' },
  two: { S: '#1d2230', H: '#d92f3c', D: '#d92f3c', C: '#1d2230' },
};
/** Face colour per suit: a whisper of the suit's colour (four-colour), so even a sliver of an overlapped card reads. */
const FACE: Record<SuitColors, Record<Suit, [string, string]>> = {
  four: { S: ['#ffffff', '#eceef4'], H: ['#ffffff', '#fdebec'], D: ['#ffffff', '#e9f1fd'], C: ['#ffffff', '#e7f5ec'] },
  two: { S: ['#fffefb', '#f4f1ea'], H: ['#fffefb', '#f4f1ea'], D: ['#fffefb', '#f4f1ea'], C: ['#fffefb', '#f4f1ea'] },
};
const PANEL: Record<SuitColors, Record<Suit, string>> = {
  four: { S: '#e2e5ee', H: '#fbdfe1', D: '#dce8fb', C: '#d9efe1' },
  two: { S: '#ece8de', H: '#f6e0dd', D: '#f6e0dd', C: '#ece8de' },
};

const GOLD = '#f2b928';
const GOLD_DARK = '#b98110';
const SKIN = '#f4cfab';
const BLUSH = '#f08f8f';
const EYE = '#262833';

// ---------------------------------------------------------------- shapes

/** Suit symbols in a 100 × 100 box. */
const SUIT_PATH: Record<Suit, string> = {
  H: 'M50 94C24 74 3 57 3 32C3 15 15 5 29 5C39 5 46 11 50 20C54 11 61 5 71 5C85 5 97 15 97 32C97 57 76 74 50 94Z',
  D: 'M50 2C60 20 73 36 91 50C73 64 60 80 50 98C40 80 27 64 9 50C27 36 40 20 50 2Z',
  S: 'M50 3C62 22 96 39 96 62C96 77 85 85 73 85C64 85 57 80 53 73C54 84 58 92 67 98H33C42 92 46 84 47 73C43 80 36 85 27 85C15 85 4 77 4 62C4 39 38 22 50 3Z',
  C:
    circle(50, 28, 20.5) +
    circle(27, 58, 20.5) +
    circle(73, 58, 20.5) +
    'M50 38L33 62H67Z' +
    'M45.5 58C45.5 80 40 90 31 98H69C60 90 54.5 80 54.5 58Z',
};

function circle(cx: number, cy: number, r: number): string {
  return `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0Z`;
}

/** Stroked glyphs in a 10 × 14 box (round caps), for ranks and the joker's letters. */
const GLYPH: Record<string, string> = {
  A: 'M0.8 14L5 0.2L9.2 14M2.3 9.2H7.7',
  '2': 'M1.1 3.7C1.3 1.3 3 0 5 0C7.4 0 8.9 1.6 8.9 3.8C8.9 6.3 6.9 7.8 5.2 9.4L1 13.8H9.2',
  '3': 'M1.3 0.4H8.5L4.7 5.6C7.6 5.5 9.3 7.3 9.3 9.7C9.3 12.4 7.3 14 5 14C3 14 1.6 13.1 0.8 11.8',
  '4': 'M7 14V0.2L0.6 9.7H9.6',
  '5': 'M8.6 0.4H2.1L1.4 6.3C2.3 5.5 3.6 5.1 5 5.1C7.6 5.1 9.3 7 9.3 9.5C9.3 12.2 7.4 14 4.9 14C3 14 1.6 13.2 0.8 12',
  '6': 'M8.3 1.3C7.4 0.4 6.3 0 5 0C2.3 0 0.9 2.9 0.9 7.4C0.9 11.6 2.5 14 5.1 14C7.6 14 9.2 12.2 9.2 9.6C9.2 7 7.5 5.4 5.2 5.4C3.2 5.4 1.8 6.5 1.1 8.1',
  '7': 'M0.8 0.4H9.3C6.6 4 5 8.6 4.2 14',
  '8': 'M5 6.5C3 6.5 1.4 5.3 1.4 3.3C1.4 1.3 3 0 5 0S8.6 1.3 8.6 3.3C8.6 5.3 7 6.5 5 6.5ZM5 6.5C2.6 6.5 0.8 8 0.8 10.2C0.8 12.5 2.6 14 5 14S9.2 12.5 9.2 10.2C9.2 8 7.4 6.5 5 6.5Z',
  '9': 'M1.7 12.7C2.6 13.6 3.7 14 5 14C7.7 14 9.1 11.1 9.1 6.6C9.1 2.4 7.5 0 4.9 0C2.4 0 0.8 1.8 0.8 4.4C0.8 7 2.5 8.6 4.8 8.6C6.8 8.6 8.2 7.5 8.9 5.9',
  '1': 'M0.3 2.7L2.8 0.2V14',
  '0': 'M3.2 0C1.4 0 0.3 2.8 0.3 7S1.4 14 3.2 14S6.1 11.2 6.1 7S5 0 3.2 0Z',
  J: 'M3.2 0.4H9M7.3 0.4V10C7.3 12.6 6 14 4 14C2.4 14 1.2 13 0.8 11.4',
  Q: 'M5 0C2.2 0 0.6 2.9 0.6 7S2.2 14 5 14S9.4 11.1 9.4 7S7.8 0 5 0ZM6.1 10.2L9.6 14.6',
  K: 'M1.2 0.2V14M8.8 0.2L1.5 8.4M4 5.7L9.2 14',
  O: 'M5 0C2.2 0 0.6 2.9 0.6 7S2.2 14 5 14S9.4 11.1 9.4 7S7.8 0 5 0Z',
  E: 'M8.8 0.4H1.2V13.6H8.8M1.2 7H7.4',
  R: 'M1.2 14V0.4H5.4C7.7 0.4 9 1.8 9 3.9S7.7 7.4 5.4 7.4H1.2M5.4 7.4L9 14',
};

/** A rank as stroked glyphs in a box at (x, y) of height h ("10" is a little wider than the rest). */
function rankGlyphs(rank: Rank, x: number, y: number, h: number, color: string, weight = 2.5): string {
  const s = h / 14;
  if (rank === 'T') {
    // "10": a narrow 1 and 0, slightly condensed.
    return (
      `<g transform="translate(${x - 0.6} ${y}) scale(${s * 0.88} ${s})" fill="none" stroke="${color}" stroke-width="${weight * 0.94}" stroke-linecap="round" stroke-linejoin="round">` +
      `<path d="${GLYPH['1']}"/><path transform="translate(6.3 0)" d="${GLYPH['0']}"/></g>`
    );
  }
  return (
    `<path transform="translate(${x} ${y}) scale(${s})" d="${GLYPH[rank]}" fill="none" stroke="${color}" ` +
    `stroke-width="${weight}" stroke-linecap="round" stroke-linejoin="round"/>`
  );
}

/** A suit symbol (the <symbol id="p"> of this card) centred at (cx, cy), `size` wide. */
function pip(cx: number, cy: number, size: number, flip = false, fill?: string): string {
  const x = cx - size / 2;
  const y = cy - size / 2;
  return `<use href="#p" x="${x}" y="${y}" width="${size}" height="${size}"${flip ? ` transform="rotate(180 ${cx} ${cy})"` : ''}${fill ? ` fill="${fill}"` : ''}/>`;
}

// ---------------------------------------------------------------- pips

const COL = { L: 34.5, M: 50, R: 65.5 };
const TOP = 31;
const BOT = 124;
const MID = (TOP + BOT) / 2;

/** Pip centres (and whether each is upside down) for 2–10. */
function pipLayout(rank: Rank): Array<[number, number, boolean]> {
  const p = (x: number, y: number): [number, number, boolean] => [x, y, y > MID + 0.5];
  const { L, M, R } = COL;
  const q1 = TOP + (BOT - TOP) / 3;
  const q2 = TOP + ((BOT - TOP) * 2) / 3;
  switch (rank) {
    case '2':
      return [p(M, TOP), p(M, BOT)];
    case '3':
      return [p(M, TOP), p(M, MID), p(M, BOT)];
    case '4':
      return [p(L, TOP), p(R, TOP), p(L, BOT), p(R, BOT)];
    case '5':
      return [p(L, TOP), p(R, TOP), p(M, MID), p(L, BOT), p(R, BOT)];
    case '6':
      return [p(L, TOP), p(R, TOP), p(L, MID), p(R, MID), p(L, BOT), p(R, BOT)];
    case '7':
      return [p(L, TOP), p(R, TOP), p(M, (TOP + MID) / 2), p(L, MID), p(R, MID), p(L, BOT), p(R, BOT)];
    case '8':
      return [p(L, TOP), p(R, TOP), p(M, (TOP + MID) / 2), p(L, MID), p(R, MID), p(M, (MID + BOT) / 2), p(L, BOT), p(R, BOT)];
    case '9':
      return [p(L, TOP), p(R, TOP), p(L, q1), p(R, q1), p(M, MID), p(L, q2), p(R, q2), p(L, BOT), p(R, BOT)];
    case 'T':
      return [p(L, TOP), p(R, TOP), p(M, (TOP + q1) / 2), p(L, q1), p(R, q1), p(L, q2), p(R, q2), p(M, (q2 + BOT) / 2), p(L, BOT), p(R, BOT)];
    default:
      return [];
  }
}

// ---------------------------------------------------------------- court portraits

/** A bust for J / Q / K, drawn in the court panel (x 23–77, y 17–138). */
function portrait(rank: 'J' | 'Q' | 'K', ink: string): string {
  const eyes =
    `<circle cx="44.5" cy="70" r="1.7" fill="${EYE}"/><circle cx="55.5" cy="70" r="1.7" fill="${EYE}"/>` +
    `<circle cx="45" cy="69.4" r="0.5" fill="#fff"/><circle cx="56" cy="69.4" r="0.5" fill="#fff"/>`;
  const cheeks = `<circle cx="40.5" cy="76" r="2.6" fill="${BLUSH}" opacity=".55"/><circle cx="59.5" cy="76" r="2.6" fill="${BLUSH}" opacity=".55"/>`;
  const head = `<rect x="45" y="80" width="10" height="20" rx="3" fill="${SKIN}"/><circle cx="50" cy="71" r="15" fill="${SKIN}"/>`;
  const body =
    `<path d="M21 140C21 112 33 99 50 99C67 99 79 112 79 140Z" fill="${ink}"/>` +
    `<path d="M41 99.6L50 112L59 99.6" fill="none" stroke="#fff" stroke-width="2.4" stroke-linejoin="round"/>` +
    pip(50, 125, 13, false, '#fff');
  if (rank === 'K') {
    return (
      body +
      head +
      eyes +
      cheeks +
      `<path d="M39 64L46.5 63.2M53.5 63.2L61 64" stroke="#a8a8a8" stroke-width="1.6" stroke-linecap="round"/>` +
      // beard and moustache
      `<path d="M34.5 71C34.5 92 43 101.5 50 101.5S65.5 92 65.5 71C61.5 79.5 56.5 82.5 50 82.5S38.5 79.5 34.5 71Z" fill="#fbfbfb" stroke="#d5d5d5" stroke-width=".8"/>` +
      `<path d="M41.5 80.5C44 76.8 47.6 77 50 79.2C52.4 77 56 76.8 58.5 80.5C55 82.6 52.4 82 50 81C47.6 82 45 82.6 41.5 80.5Z" fill="#fbfbfb" stroke="#d5d5d5" stroke-width=".8"/>` +
      // crown
      `<path d="M35.5 58L33.5 37L43 47L50 32L57 47L66.5 37L64.5 58Z" fill="${GOLD}" stroke="${GOLD_DARK}" stroke-width=".9" stroke-linejoin="round"/>` +
      `<rect x="35" y="53" width="30" height="7" rx="1.5" fill="${GOLD}" stroke="${GOLD_DARK}" stroke-width=".9"/>` +
      `<circle cx="33.5" cy="37" r="2.8" fill="${ink}"/><circle cx="50" cy="32" r="3" fill="${ink}"/><circle cx="66.5" cy="37" r="2.8" fill="${ink}"/>` +
      `<circle cx="50" cy="56.5" r="2.3" fill="${ink}"/><circle cx="41.5" cy="56.5" r="1.4" fill="#fff" opacity=".85"/><circle cx="58.5" cy="56.5" r="1.4" fill="#fff" opacity=".85"/>`
    );
  }
  if (rank === 'Q') {
    const hair = '#6b3f26';
    return (
      `<path d="M32.5 72C30.5 52 40 51 50 51S69.5 52 67.5 72C69.5 89 66 99 61 104H39C34 99 30.5 89 32.5 72Z" fill="${hair}"/>` +
      body +
      `<path d="M42 100.5Q50 107 58 100.5" fill="none" stroke="${GOLD}" stroke-width="1.5"/><circle cx="50" cy="105.4" r="2" fill="${GOLD}"/>` +
      head +
      eyes +
      `<path d="M42.9 69.3L41.2 67.9M57.1 69.3L58.8 67.9" stroke="${EYE}" stroke-width="1" stroke-linecap="round"/>` +
      cheeks +
      `<path d="M47.4 78.6Q50 81.6 52.6 78.6Q50 77.6 47.4 78.6Z" fill="#d6455a"/>` +
      `<path d="M35 69C35 57.5 42 54.5 50 54.5S65 57.5 65 69C60 63 55 61 50 60.4C46 62.4 40.5 65 35 69Z" fill="${hair}"/>` +
      `<circle cx="35.4" cy="77" r="1.9" fill="${GOLD}"/><circle cx="64.6" cy="77" r="1.9" fill="${GOLD}"/>` +
      // tiara
      `<path d="M38.5 57.5L41 47.5L45.5 52.5L50 42L54.5 52.5L59 47.5L61.5 57.5Q50 54.5 38.5 57.5Z" fill="${GOLD}" stroke="${GOLD_DARK}" stroke-width=".8" stroke-linejoin="round"/>` +
      `<circle cx="50" cy="49.5" r="2.3" fill="${ink}"/>`
    );
  }
  const hair = '#8a5a2b';
  return (
    body +
    `<circle cx="44" cy="104.5" r="1.5" fill="#fff"/><circle cx="56" cy="104.5" r="1.5" fill="#fff"/>` +
    head +
    `<path d="M35.2 68C35 59 41 55.5 50 55.5S65 59 64.8 68L62.5 66C56 63 44 63 37.5 66Z" fill="${hair}"/>` +
    eyes +
    cheeks +
    `<path d="M45 77Q50 81.5 55 77" fill="none" stroke="${EYE}" stroke-width="1.4" stroke-linecap="round"/>` +
    // feathered cap
    `<path d="M59 51C64 40 71 33 81 28C78 37 71 46 62.5 54Z" fill="${GOLD}" stroke="${GOLD_DARK}" stroke-width=".8" stroke-linejoin="round"/>` +
    `<path d="M62 50.5Q69 40 77 33" fill="none" stroke="${GOLD_DARK}" stroke-width=".7"/>` +
    `<path d="M31.5 63C31 50 41 44 51.5 44C62 44 70.5 49.5 70.5 57C70.5 61 67 62.4 63 61.4C53 59 42 60 33 65.5Z" fill="${ink}"/>` +
    `<path d="M33.5 63.5C43 58.5 54 57.8 64.5 60.5" fill="none" stroke="${GOLD}" stroke-width="1.6" stroke-linecap="round"/>`
  );
}

// ---------------------------------------------------------------- jokers

interface JokerTheme {
  ink: string;
  hat: [string, string, string];
  bell: string;
  ray: string;
  base: string;
  face: string;
  nose: string;
}

const BIG: JokerTheme = {
  ink: '#d92f3c',
  hat: ['#d92f3c', '#f2b928', '#1c6ce0'],
  bell: '#f2b928',
  ray: '#ffe6a3',
  base: '#fff9e8',
  face: SKIN,
  nose: '#e8414d',
};
const SMALL: JokerTheme = {
  ink: '#1d2230',
  hat: ['#2b3142', '#8b93a5', '#2b3142'],
  bell: '#b7bdc9',
  ray: '#e4e7ed',
  base: '#f8f9fb',
  face: '#f3f3f1',
  nose: '#5d6474',
};

/** Jester head (hat, face, ruff), centred on (50, 80). */
function jester(t: JokerTheme): string {
  const ruff: string[] = [];
  for (let i = 0; i < 6; i++) {
    const x0 = 29 + i * 7;
    ruff.push(`<path d="M${x0} 101L${x0 + 3.5} 113L${x0 + 7} 101Z" fill="${i % 2 ? t.hat[1] : t.hat[0]}"/>`);
  }
  return (
    `<path d="M28 102Q50 94 72 102" fill="none" stroke="${t.hat[0]}" stroke-width="3" stroke-linecap="round"/>` +
    ruff.join('') +
    `<circle cx="50" cy="83" r="17" fill="${t.face}" stroke="${t.ink}" stroke-opacity=".12" stroke-width="1"/>` +
    // hat: three floppy points with bells
    `<path d="M33 76C26 62 20 55 12.5 51C26 47.5 39 53 46 66Z" fill="${t.hat[0]}"/>` +
    `<path d="M67 76C74 62 80 55 87.5 51C74 47.5 61 53 54 66Z" fill="${t.hat[2]}"/>` +
    `<path d="M39 70C41 54 45 42 50 30C55 42 59 54 61 70Z" fill="${t.hat[1]}"/>` +
    `<path d="M32 77.5Q50 64 68 77.5L68.5 72Q50 59 31.5 72Z" fill="${t.hat[0]}"/>` +
    `<circle cx="12.5" cy="51" r="4.2" fill="${t.bell}"/><circle cx="50" cy="30" r="4.2" fill="${t.bell}"/><circle cx="87.5" cy="51" r="4.2" fill="${t.bell}"/>` +
    // face
    `<circle cx="43.5" cy="82" r="1.9" fill="${EYE}"/><circle cx="56.5" cy="82" r="1.9" fill="${EYE}"/>` +
    `<circle cx="38.5" cy="89" r="3" fill="${t.nose}" opacity=".35"/><circle cx="61.5" cy="89" r="3" fill="${t.nose}" opacity=".35"/>` +
    `<circle cx="50" cy="87" r="2.6" fill="${t.nose}"/>` +
    `<path d="M42 91.5Q50 100 58 91.5" fill="none" stroke="${EYE}" stroke-width="1.6" stroke-linecap="round"/>`
  );
}

function sunburst(t: JokerTheme): string {
  const rays: string[] = [];
  const n = 18;
  for (let i = 0; i < n; i += 2) {
    const a0 = (i / n) * Math.PI * 2;
    const a1 = ((i + 1) / n) * Math.PI * 2;
    const p = (a: number) => `${(CX + 130 * Math.cos(a)).toFixed(1)} ${(CY + 130 * Math.sin(a)).toFixed(1)}`;
    rays.push(`M${CX} ${CY}L${p(a0)}L${p(a1)}Z`);
  }
  return `<path d="${rays.join('')}" fill="${t.ray}" opacity=".75"/>`;
}

function jokerLetters(x: number, y: number, h: number, color: string, weight: number): string {
  return ['J', 'O', 'K', 'E', 'R']
    .map(
      (g, i) =>
        `<path transform="translate(${x} ${y + i * h * 1.22}) scale(${h / 14})" d="${GLYPH[g]}" fill="none" stroke="${color}" stroke-width="${weight}" stroke-linecap="round" stroke-linejoin="round"/>`,
    )
    .join('');
}

const STAR = 'M50 4L61 37H96L68 58L79 92L50 71L21 92L32 58L4 37H39Z';

// ---------------------------------------------------------------- whole cards

const CARD_SHAPE = `x="0.75" y="0.75" width="${W - 1.5}" height="${H - 1.5}" rx="7.5"`;
const EDGE = `<rect ${CARD_SHAPE} fill="none" stroke="#000" stroke-opacity=".16" stroke-width="1.5"/>`;

function wrap(defs: string, body: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}"><defs>${defs}</defs>${body}</svg>`;
}

/** A face's background (defs, body): a soft wash of the suit's tint and an inner hairline. */
function faceBase(top: string, bottom: string, line: string): [string, string] {
  return [
    `<linearGradient id="f" x1="0" y1="0" x2=".35" y2="1"><stop offset="0" stop-color="${top}"/><stop offset="1" stop-color="${bottom}"/></linearGradient>`,
    `<rect ${CARD_SHAPE} fill="url(#f)"/><rect x="4" y="4" width="${W - 8}" height="${H - 8}" rx="5" fill="none" stroke="${line}" stroke-opacity=".22" stroke-width=".8"/>`,
  ];
}

function suitedCard(card: Card, layout: CardLayout, colors: SuitColors): string {
  const rank = cardRank(card)!;
  const suit = cardSuit(card)!;
  const ink = SUIT_INK[colors][suit];
  const [top, bottom] = FACE[colors][suit];
  const [baseDefs, baseBody] = faceBase(top, bottom, ink);
  let defs = `<symbol id="p" viewBox="0 0 100 100"><path d="${SUIT_PATH[suit]}" fill="${ink}"/></symbol>` + baseDefs;
  let body = baseBody;
  const court = rank === 'J' || rank === 'Q' || rank === 'K';

  if (layout === 'compact') {
    body += rankGlyphs(rank, 8, 8, 40, ink, 2.35);
    body += pip(22.5, 66, 27);
    if (court) {
      // The portrait's head and shoulders, cropped into the lower right.
      const s = 0.92;
      const frame = 'x="23" y="26" width="54" height="84" rx="7"';
      defs += `<clipPath id="c"><rect ${frame}/></clipPath>`;
      body +=
        `<g transform="translate(${(44 - 23 * s).toFixed(2)} ${(67 - 26 * s).toFixed(2)}) scale(${s})">` +
        `<rect ${frame} fill="${PANEL[colors][suit]}"/><g clip-path="url(#c)">${portrait(rank, ink)}</g>` +
        `<rect ${frame} fill="none" stroke="${ink}" stroke-width="1.6"/></g>`;
    } else {
      body += pip(70, 120, 46, false);
    }
    return wrap(defs, body + EDGE);
  }

  // Full layout: corner indices (top left, and upside down bottom right).
  const index = rankGlyphs(rank, 5.5, 6.5, 21, ink, 2.6) + pip(13.1, 37, 14);
  body += index + `<g transform="rotate(180 ${CX} ${CY})">${index}</g>`;

  if (rank === 'A') {
    body += `<circle cx="${CX}" cy="${CY}" r="31" fill="${PANEL[colors][suit]}"/>`;
    body += `<circle cx="${CX}" cy="${CY}" r="31" fill="none" stroke="${ink}" stroke-opacity=".35" stroke-width="1" stroke-dasharray="2.2 3"/>`;
    body += pip(CX, CY + 1, 42);
  } else if (court) {
    const panel = PANEL[colors][suit];
    defs +=
      `<clipPath id="c"><rect x="23" y="17" width="54" height="121" rx="5"/></clipPath>` +
      `<pattern id="d" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="2" height="5" fill="${ink}" opacity=".07"/></pattern>`;
    body +=
      `<rect x="23" y="17" width="54" height="121" rx="5" fill="${panel}"/>` +
      `<rect x="23" y="17" width="54" height="121" rx="5" fill="url(#d)"/>` +
      `<g clip-path="url(#c)">${portrait(rank, ink)}</g>` +
      `<rect x="23" y="17" width="54" height="121" rx="5" fill="none" stroke="${ink}" stroke-width="1.3"/>`;
  } else {
    const size = rank === '9' || rank === 'T' ? 16.5 : 18;
    for (const [x, y, flip] of pipLayout(rank)) body += pip(x, y, size, flip);
  }
  return wrap(defs, body + EDGE);
}

function jokerCard(card: Card, layout: CardLayout): string {
  const t = card === BIG_JOKER ? BIG : SMALL;
  const defs = `<clipPath id="c"><rect x="4" y="4" width="${W - 8}" height="${H - 8}" rx="5"/></clipPath>`;
  let body = `<rect ${CARD_SHAPE} fill="${t.base}"/><g clip-path="url(#c)">${sunburst(t)}</g>`;
  body += `<rect x="4" y="4" width="${W - 8}" height="${H - 8}" rx="5" fill="none" stroke="${t.ink}" stroke-opacity=".3" stroke-width=".8"/>`;
  if (layout === 'compact') {
    body += `<g transform="translate(-6 -12) scale(.62)">${jester(t)}</g>`;
    body += jokerLetters(8, 61, 11, t.ink, 3.3);
    body += `<path transform="translate(50 104) scale(.42)" d="${STAR}" fill="${t.hat[0]}" stroke="${t.bell}" stroke-width="5" stroke-linejoin="round"/>`;
    return wrap(defs, body + EDGE);
  }
  const letters = jokerLetters(7, 8, 8.6, t.ink, 3.3);
  body += letters + `<g transform="rotate(180 ${CX} ${CY})">${letters}</g>`;
  body += `<g transform="translate(8 10) scale(.84)">${jester(t)}</g>`;
  body += `<path transform="translate(41.5 115) scale(.17)" d="${STAR}" fill="${t.bell}"/>`;
  return wrap(defs, body + EDGE);
}

const TRACTOR =
  // the favicon's tractor, in a 64 × 64 box
  '<rect x="46.5" y="19" width="4" height="14" rx="1" fill="#1b1b1b"/><rect x="45.5" y="17.5" width="6" height="3" rx="1" fill="#1b1b1b"/>' +
  '<rect x="12" y="13" width="21" height="24" rx="2" fill="#d62828"/><rect x="9.5" y="9.5" width="26" height="5" rx="2" fill="#a61b1b"/>' +
  '<rect x="16" y="17" width="13" height="12" rx="1.5" fill="#e8f4ff"/><rect x="29" y="31" width="27" height="13" rx="3" fill="#d62828"/>' +
  '<rect x="31" y="34" width="18" height="2.2" rx="1" fill="#a61b1b"/><circle cx="20" cy="44" r="14" fill="#1b1b1b"/>' +
  '<circle cx="20" cy="44" r="6.5" fill="#f5c542"/><circle cx="20" cy="44" r="2.5" fill="#1b1b1b"/>' +
  '<circle cx="49" cy="50" r="8" fill="#1b1b1b"/><circle cx="49" cy="50" r="3.6" fill="#f5c542"/>';

function backCard(): string {
  const defs =
    `<pattern id="l" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">` +
    `<rect width="10" height="10" fill="#a3203a"/><path d="M0 5H10M5 0V10" stroke="#c23a52" stroke-width="1.6"/>` +
    `<circle cx="5" cy="5" r="1.4" fill="#f2b928"/></pattern>` +
    `<radialGradient id="g" cx=".5" cy=".5" r=".6"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".28"/></radialGradient>`;
  const body =
    `<rect ${CARD_SHAPE} fill="#fdfaf3"/>` +
    `<rect x="5" y="5" width="${W - 10}" height="${H - 10}" rx="5" fill="url(#l)"/>` +
    `<rect x="5" y="5" width="${W - 10}" height="${H - 10}" rx="5" fill="url(#g)"/>` +
    `<rect x="8" y="8" width="${W - 16}" height="${H - 16}" rx="3.5" fill="none" stroke="#f2b928" stroke-width="1.2"/>` +
    `<circle cx="${CX}" cy="${CY}" r="27" fill="#fff6df" stroke="#f2b928" stroke-width="3"/>` +
    `<circle cx="${CX}" cy="${CY}" r="22.5" fill="none" stroke="#a3203a" stroke-width="1" stroke-dasharray="1.6 2.2"/>` +
    `<g transform="translate(${CX - 17.5} ${CY - 18}) scale(.55)">${TRACTOR}</g>` +
    EDGE;
  return wrap(defs, body);
}

/** The SVG markup for a card (null: the back). */
export function cardSvg(card: Card | null, layout: CardLayout, colors: SuitColors): string {
  if (card === null) return backCard();
  if (card === BIG_JOKER || card === SMALL_JOKER) return jokerCard(card, layout);
  return suitedCard(card, layout, colors);
}

const cache = new Map<string, string>();

/** A data URL for the card's image (memoised). */
export function cardSrc(card: Card | null, layout: CardLayout, colors: SuitColors): string {
  const key = `${card ?? 'back'}|${card === null ? '' : layout}|${colors}`;
  let src = cache.get(key);
  if (!src) {
    src = `data:image/svg+xml,${encodeURIComponent(cardSvg(card, layout, colors))}`;
    cache.set(key, src);
  }
  return src;
}
