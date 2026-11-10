// Shared, game-agnostic types: who sits at the table and how a device sees
// them. Everything Tractor-specific lives under src/game/.

export type PlayerId = string;

export type Seat = {
  /** Stable identity. Online human seats use their owner's UUID; AI seats are `ai-N`. */
  id: PlayerId;
  name: string;
  /** Hex color from SEAT_COLORS. */
  color: string;
  isAI: boolean;
  /** A seat on this device's hot-seat table (online seats are never local). */
  isLocal: boolean;
};
