// The seat color palette shared by the local and online lobbies: 14
// distinguishable colors. Hex values mirror the Catan and card-collecting projects'
// PLAYER_COLOR_HEX so the same seat color reads the same across both apps.

export const SEAT_COLORS: readonly string[] = [
  '#e64a4a', // red
  '#3a6ec9', // blue
  '#f37025', // orange
  '#f0f0f0', // white
  '#8a47c4', // purple
  '#e85a9c', // pink
  '#1abc9c', // teal
  '#f5b223', // gold
  '#6dc960', // lime
  '#6e4a2c', // brown
  '#4a4a4a', // black
  '#2e7d3a', // forest
  '#c89fe8', // lavender
  '#7e2a2a', // maroon
];

export const SEAT_COLOR_LABELS: Record<string, string> = {
  '#e64a4a': 'Red',     '#3a6ec9': 'Blue',     '#f37025': 'Orange',
  '#f0f0f0': 'White',   '#8a47c4': 'Purple',   '#e85a9c': 'Pink',
  '#1abc9c': 'Teal',    '#f5b223': 'Gold',     '#6dc960': 'Lime',
  '#6e4a2c': 'Brown',   '#4a4a4a': 'Black',    '#2e7d3a': 'Forest',
  '#c89fe8': 'Lavender','#7e2a2a': 'Maroon',
};

/** The first palette color no seat wears yet. */
export function freeSeatColor(taken: readonly { color: string }[]): string {
  return SEAT_COLORS.find((c) => !taken.some((s) => s.color === c)) ?? SEAT_COLORS[taken.length % SEAT_COLORS.length]!;
}
