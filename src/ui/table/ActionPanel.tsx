// What the viewer can do right now, stage by stage: draw and declare trump,
// take / turn over the kitty, bury cards and call friends, play. Keyboard:
// Enter does the main thing (play, bury, draw, take the kitty), Esc clears
// the selection, H selects a hint, D or Space draws.

import { useEffect, useMemo, useRef, useState, type MutableRefObject, type ReactNode } from 'react';
import { adviseAction } from '@/ai';
import { RANKS, SUITS, SUIT_SYMBOL, cardLabel, rankLabel, type Card, type Rank, type Suit } from '@/game/cards';
import {
  activeSeat,
  bidText,
  canEndRoundEarly,
  canPickUp,
  canRedeal,
  canRevealKitty,
  canSayDoneBidding,
  drawingSeat,
  friendCallProblems,
  ordinalText,
  playProblemFor,
  tractorRequirements,
  validBidsFor,
} from '@/game/engine';
import { describeShapes, describeUnit, formatSize, possibleFormats, requiredShapes } from '@/game/format';
import type { Action, FriendCall, TractorState } from '@/game/types';
import { usePrefs, type BidWindow, type DealSpeed } from '@/store/prefsStore';
import type { Automation } from './Table';

interface Props {
  state: TractorState;
  viewer: number | null;
  /** The viewer's hand as shown (sorted). */
  hand: Card[];
  selected: ReadonlySet<number>;
  setSelected: (s: Set<number>) => void;
  dispatch: (a: Action) => string | null;
  automation?: Automation | null;
}

interface Hotkeys {
  primary?: () => void;
  hint?: () => void;
  draw?: () => void;
}

function name(state: TractorState, seat: number | null): string {
  return seat === null ? 'someone' : state.seats[seat]?.name ?? '?';
}

/** Indices in `hand` holding `cards` (each card matched once). */
export function indicesOf(hand: readonly Card[], cards: readonly Card[]): Set<number> {
  const used = new Set<number>();
  for (const c of cards) {
    const i = hand.findIndex((h, j) => h === c && !used.has(j));
    if (i >= 0) used.add(i);
  }
  return used;
}

/** The current time, re-rendering every `ms` while `on` (for countdowns). Read at render, so it's never stale. */
function useNow(on: boolean, ms = 250): number {
  const [, tick] = useState(0);
  useEffect(() => {
    if (!on) return;
    const t = setInterval(() => tick((n) => n + 1), ms);
    return () => clearInterval(t);
  }, [on, ms]);
  return Date.now();
}

function useHotkeys(keys: MutableRefObject<Hotkeys>, clear: () => void): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      const k = keys.current;
      if (e.key === 'Enter' && k.primary) {
        e.preventDefault();
        k.primary();
      } else if (e.key === 'Escape') {
        clear();
      } else if ((e.key === 'h' || e.key === 'H') && k.hint) {
        k.hint();
      } else if ((e.key === 'd' || e.key === 'D' || e.key === ' ') && k.draw) {
        e.preventDefault();
        k.draw();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [keys, clear]);
}

export function ActionPanel({ state, viewer, hand, selected, setSelected, dispatch, automation = null }: Props) {
  const r = state.round;
  const [error, setError] = useState<string | null>(null);
  const [reading, setReading] = useState<number | null>(null);
  const picked = useMemo(() => [...selected].sort((a, b) => a - b).map((i) => hand[i]!), [selected, hand]);
  useEffect(() => setError(null), [selected, r.stage, r.trick?.plays.length]);
  useEffect(() => setReading(null), [picked.join(',')]);
  const keys = useRef<Hotkeys>({});
  const clear = useMemo(() => () => setSelected(new Set()), [setSelected]);
  useHotkeys(keys, clear);
  const now = useNow(!!automation);
  keys.current = {};

  function act(a: Action) {
    const err = dispatch(a);
    setError(err);
    if (!err) setSelected(new Set());
  }

  if (state.status === 'gameOver' || r.stage === 'roundOver') return null;
  const active = activeSeat(state);
  const mine = viewer !== null && active === viewer;
  const secondsLeft = automation ? Math.max(0, Math.ceil((automation.firesAt - now) / 1000)) : 0;
  const handoff = automation?.kind === 'handoff' && automation.seat !== null && (
    <div className="lobby-hint">{name(state, automation.seat)} is offline — the AI plays for them in {secondsLeft}s.</div>
  );

  // -------------------------------------------------------- dealing and declaring
  if (r.stage === 'draw' || r.stage === 'bid') {
    const bids = viewer !== null ? validBidsFor(state, viewer) : [];
    const myTurn = r.stage === 'bid' && viewer !== null && r.bidTurn === viewer;
    const drawer = drawingSeat(state);
    const myDraw = viewer !== null && drawer === viewer;
    let prompt: string;
    const buttons: ReactNode[] = [];
    if (r.stage === 'draw' && r.deck.length > 0) {
      const nothing = state.landlord !== null && state.levels[state.landlord] === 'NT';
      const who = myDraw ? 'Your draw' : `${name(state, drawer)} draws`;
      prompt = nothing
        ? `${who} — the landlord plays no-trump, so there is nothing to declare.`
        : `${who} · ${r.deck.length} left. Show a ${trumpNumberText(state, viewer)} to declare its suit as trump.`;
    } else if (viewer !== null && canPickUp(state, viewer)) {
      prompt = "The deal is done. Take the kitty when you're ready (others can still overcall until you do).";
      buttons.push(<button key="pick" onClick={() => act({ type: 'pickUpKitty', seat: viewer })}>Take the kitty</button>);
      keys.current.primary = () => act({ type: 'pickUpKitty', seat: viewer });
    } else if (viewer !== null && canRevealKitty(state, viewer)) {
      prompt = 'Nobody declared. Turn over kitty cards to set trump.';
      buttons.push(<button key="rev" onClick={() => act({ type: 'revealKitty', seat: viewer })}>Turn over a kitty card</button>);
      keys.current.primary = () => act({ type: 'revealKitty', seat: viewer });
    } else if (canRedeal(state) && (viewer === active || state.seats[active ?? 0]?.isAI === false)) {
      prompt = 'Nobody declared trump.';
      buttons.push(<button key="redeal" onClick={() => act({ type: 'redeal' })}>Shuffle and redeal</button>);
    } else if (myTurn) {
      prompt = bids.length ? 'Your turn: declare trump, or pass.' : 'Your turn: you have nothing to declare.';
      if (!bids.length) keys.current.primary = () => act({ type: 'pass', seat: viewer! });
    } else if (r.stage === 'bid') {
      prompt = r.bidTurn === null ? `Waiting for ${name(state, state.landlord)} to turn over the kitty…` : `${name(state, r.bidTurn)} is deciding whether to declare…`;
    } else {
      const what = r.bids.length || r.autobid ? `${name(state, active)} takes the kitty` : canRedeal(state) ? 'Redeal' : `${name(state, active)} turns over the kitty`;
      prompt = automation?.kind === 'bidWindow' ? `Last chance to overcall — ${what} in ${secondsLeft}s.` : `Waiting for ${name(state, active)}…`;
    }
    if (myDraw && r.deck.length > 0) {
      const draw = () => act({ type: 'draw', seat: viewer! });
      keys.current.draw = draw;
      if (!keys.current.primary) keys.current.primary = draw;
    }
    const done = viewer !== null && canSayDoneBidding(state, viewer);
    return (
      <div className="action-panel">
        <div className="action-prompt">{prompt}</div>
        <div className="action-buttons">
          {myDraw && <DrawButton onDraw={() => act({ type: 'draw', seat: viewer! })} />}
          {bids.map((b) => (
            <button key={`${b.card}x${b.count}`} className="bid-button" onClick={() => act({ type: 'bid', seat: viewer!, card: b.card, count: b.count })}>
              Declare {bidText(b)}
            </button>
          ))}
          {myTurn && <button className="secondary" onClick={() => act({ type: 'pass', seat: viewer! })}>Pass</button>}
          {done && (
            <button className="secondary" onClick={() => act({ type: 'doneBidding', seat: viewer! })} title="Let the landlord-to-be take the kitty without waiting">
              No more bids
            </button>
          )}
          {buttons}
        </div>
        {r.stage === 'draw' && <DealControls />}
        {handoff}
        {error && <div className="action-error">{error}</div>}
      </div>
    );
  }

  // -------------------------------------------------------- the kitty
  if (r.stage === 'exchange') {
    if (viewer === null || r.landlord !== viewer) {
      return (
        <div className="action-panel">
          <div className="action-prompt">
            {name(state, r.landlord)} picked up the kitty and is burying {r.kittySize} cards{state.rules.gameMode === 'findingFriends' ? ' and calling friends' : ''}…
          </div>
          {handoff}
        </div>
      );
    }
    return <ExchangePanel state={state} viewer={viewer} hand={hand} picked={picked} setSelected={setSelected} act={act} error={error} keys={keys} />;
  }

  // -------------------------------------------------------- tricks
  const trick = r.trick!;
  const endEarly = viewer !== null && canEndRoundEarly(state) && (
    <div className="end-early">
      <span className="lobby-hint">Nothing left to play can change this round's result.</span>
      <button className="secondary" onClick={() => act({ type: 'endRoundEarly', seat: viewer })}>End round early</button>
    </div>
  );
  if (!mine || viewer === null) {
    return (
      <div className="action-panel">
        <div className="action-prompt">{active === null ? '' : `Waiting for ${name(state, active)}…`}</div>
        {handoff}
        {endEarly}
      </div>
    );
  }
  const format = trick.format;
  const need = format ? formatSize(format) : null;
  const required = format && r.trump ? requiredShapes(r.trump, format, r.hands[viewer]!, state.config.trickDrawPolicy) : null;
  const problem = picked.length ? playProblemFor(state, viewer, picked) : null;
  const readings = !format && picked.length > 1 && !problem && r.trump ? possibleFormats(r.trump, tractorRequirements(state.config), picked) : [];
  const canPlay = picked.length > 0 && !problem && (need === null || picked.length === need);
  const playIt = () => act({ type: 'play', seat: viewer, cards: picked, ...(readings.length > 1 && reading !== null ? { format: reading } : {}) });

  function hint() {
    void adviseAction(state, viewer!).then((a) => {
      if (a?.type === 'play') setSelected(indicesOf(hand, a.cards));
    }, () => undefined);
  }
  keys.current.hint = hint;
  if (canPlay) keys.current.primary = playIt;

  let prompt: string;
  if (!format) prompt = 'Your lead: a single, a pair, a tractor — or throw several that nobody can beat.';
  else {
    const what = format.units.map(describeUnit).join(' + ');
    prompt = `Your turn: play ${need} card${need === 1 ? '' : 's'} (led: ${what}).`;
    if (required && required.length) prompt += ` You must include ${describeShapes(required)} of the led suit.`;
  }

  return (
    <div className="action-panel">
      <div className="action-prompt">{prompt}</div>
      <div className="action-buttons">
        <button onClick={playIt} disabled={!canPlay} title="Enter">
          Play{picked.length ? ` ${picked.length}` : ''}
        </button>
        <button className="secondary" onClick={() => setSelected(new Set())} disabled={picked.length === 0} title="Esc">Clear</button>
        <button className="secondary" onClick={hint} title="Select what the AI would play (H)">Hint</button>
        {readings.length > 1 && (
          <label className="action-reading">
            Read as
            <select value={reading ?? readings.length - 1} onChange={(e) => setReading(Number(e.target.value))}>
              {readings.map((units, i) => (
                <option key={i} value={i}>{units.map(describeUnit).join(' + ')}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      {(error || (problem && need !== null && picked.length === need) || (problem && !format)) && <div className="action-error">{error ?? problem}</div>}
      {readings.length > 1 && <div className="lobby-hint">More than one unit: this is a throw — if anyone can beat part of it, only that part is played.</div>}
      {!format && readings.length === 1 && readings[0]!.length > 1 && (
        <div className="lobby-hint">This is a throw — if anyone can beat part of it, only that part is played.</div>
      )}
      {endEarly}
    </div>
  );
}

function trumpNumberText(state: TractorState, viewer: number | null): string {
  const level = state.levels[state.landlord ?? viewer ?? 0]!;
  return level === 'NT' ? 'pair of jokers' : rankLabel(level);
}

function DrawButton({ onDraw }: { onDraw: () => void }) {
  const autoDraw = usePrefs((s) => s.autoDraw);
  if (autoDraw) return null;
  return (
    <button className="draw-button" onClick={onDraw} title="D or Space">
      Draw a card
    </button>
  );
}

/** This device's deal settings, right where they matter. */
function DealControls() {
  const autoDraw = usePrefs((s) => s.autoDraw);
  const dealSpeed = usePrefs((s) => s.dealSpeed);
  const bidWindow = usePrefs((s) => s.bidWindow);
  const set = usePrefs((s) => s.set);
  return (
    <div className="deal-controls">
      <label className="opt-inline">
        <input type="checkbox" checked={autoDraw} onChange={(e) => set({ autoDraw: e.target.checked })} /> Draw my cards automatically
      </label>
      <label className="opt-inline">
        Deal speed
        <select value={dealSpeed} onChange={(e) => set({ dealSpeed: e.target.value as DealSpeed })}>
          <option value="slow">Slow</option>
          <option value="normal">Normal</option>
          <option value="fast">Fast</option>
        </select>
      </label>
      <label className="opt-inline" title="After the deal, how long an AI waits before taking the kitty, so you can still overcall">
        Time to overcall
        <select value={bidWindow} onChange={(e) => set({ bidWindow: e.target.value as BidWindow })}>
          <option value="short">3 s</option>
          <option value="normal">6 s</option>
          <option value="long">12 s</option>
        </select>
      </label>
    </div>
  );
}

// -------------------------------------------------------- bury + friends

function ExchangePanel({ state, viewer, hand, picked, setSelected, act, error, keys }: {
  state: TractorState;
  viewer: number;
  hand: Card[];
  picked: Card[];
  setSelected: (s: Set<number>) => void;
  act: (a: Action) => void;
  error: string | null;
  keys: MutableRefObject<Hotkeys>;
}) {
  const r = state.round;
  const ff = state.rules.gameMode === 'findingFriends' && state.rules.numFriends > 0;
  const [calls, setCalls] = useState<FriendCall[]>([]);
  // Start the friend calls from the AI's suggestion once its network has loaded.
  useEffect(() => {
    if (!ff) return;
    let live = true;
    void adviseAction(state, viewer).then((a) => {
      if (live && a?.type === 'exchange') setCalls((cur) => (cur.length ? cur : a.friends));
    }, () => undefined);
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const problems = friendCallProblems(state, calls);
  const k = r.kittySize;
  const ready = picked.length === k && !(ff && problems.length > 0);
  const bury = () => act({ type: 'exchange', seat: viewer, bury: picked, friends: ff ? calls : [] });

  function suggest() {
    void adviseAction(state, viewer).then((a) => {
      if (a?.type !== 'exchange') return;
      setSelected(indicesOf(hand, a.bury));
      if (ff) setCalls(a.friends);
    }, () => undefined);
  }
  keys.current.hint = suggest;
  if (ready) keys.current.primary = bury;

  return (
    <div className="action-panel">
      <div className="action-prompt">
        You're the landlord. Pick {k} card{k === 1 ? '' : 's'} to bury ({picked.length}/{k} selected). Points you bury go to the attackers ×2 or more if they win the last trick.
      </div>
      {ff && (
        <div className="friend-picker">
          <span className="friend-picker-label">Call {state.rules.numFriends === 1 ? 'a friend' : `${state.rules.numFriends} friends`}: whoever plays</span>
          {calls.map((c, i) => (
            <FriendCallInput key={i} call={c} numDecks={state.rules.numDecks} onChange={(next) => setCalls(calls.map((x, j) => (j === i ? next : x)))} />
          ))}
          {problems.length > 0 && <div className="action-error">{problems.join(' ')}</div>}
        </div>
      )}
      <div className="action-buttons">
        <button onClick={bury} disabled={!ready} title="Enter">
          Bury {k} and start
        </button>
        <button className="secondary" onClick={() => setSelected(new Set())} disabled={picked.length === 0} title="Esc">Clear</button>
        <button className="secondary" onClick={suggest} title="H">Suggest</button>
      </div>
      {error && <div className="action-error">{error}</div>}
    </div>
  );
}

function FriendCallInput({ call, numDecks, onChange }: { call: FriendCall; numDecks: number; onChange: (c: FriendCall) => void }) {
  const rank = call.card[0] as Rank;
  const suit = call.card[1] as Suit;
  return (
    <span className="friend-call">
      <select value={call.ordinal} onChange={(e) => onChange({ ...call, ordinal: Number(e.target.value) })} aria-label="Which copy">
        {Array.from({ length: numDecks }, (_, i) => (
          <option key={i} value={i + 1}>the {ordinalText(i + 1)}</option>
        ))}
      </select>
      <select value={rank} onChange={(e) => onChange({ ...call, card: e.target.value + suit })} aria-label="Rank">
        {[...RANKS].reverse().map((x) => (
          <option key={x} value={x}>{rankLabel(x)}</option>
        ))}
      </select>
      <select value={suit} onChange={(e) => onChange({ ...call, card: rank + e.target.value })} aria-label="Suit">
        {SUITS.map((x) => (
          <option key={x} value={x}>{SUIT_SYMBOL[x]}</option>
        ))}
      </select>
      <span className="friend-call-label">({cardLabel(call.card)})</span>
    </span>
  );
}
