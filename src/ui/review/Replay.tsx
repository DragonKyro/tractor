// Step through the match move by move, every hand face up. Rebuilt from the
// recorded opening state + actions (gameStore), with a checkpoint every 20
// steps so any step is a few reducer calls away. Only finished rounds can be
// replayed (an unfinished round would show the hands still in play).

import { useEffect, useMemo, useState } from 'react';
import { applyAction } from '@/game/engine';
import { totalPoints } from '@/game/cards';
import type { Action, TractorState } from '@/game/types';
import { useGameStore } from '@/store/gameStore';
import { CardRow } from '../CardView';
import { AllHands } from '../table/Table';
import { Felt } from '../table/Felt';
import { trumpParts } from '../table/labels';

interface Timeline {
  checkpoints: Map<number, TractorState>;
  /** Step at which each round's deal starts, and at which it's scored. */
  roundStarts: number[];
  roundEnds: number[];
  /** The last step that may be shown. */
  maxStep: number;
}

const EVERY = 20;

function buildTimeline(initial: TractorState, actions: readonly Action[]): Timeline {
  const checkpoints = new Map<number, TractorState>([[0, initial]]);
  const roundStarts = [0];
  const roundEnds: number[] = [];
  let s = initial;
  for (let i = 0; i < actions.length; i++) {
    const prev = s;
    try {
      s = applyAction(s, actions[i]!);
    } catch {
      break;
    }
    const step = i + 1;
    if (step % EVERY === 0) checkpoints.set(step, s);
    if (s.round.stage === 'roundOver' && prev.round.stage !== 'roundOver') {
      roundEnds.push(step);
      checkpoints.set(step, s);
    }
    if (prev.round.stage === 'roundOver' && s.round.stage !== 'roundOver') roundStarts.push(step);
  }
  return { checkpoints, roundStarts, roundEnds, maxStep: roundEnds[roundEnds.length - 1] ?? 0 };
}

function stateAt(t: Timeline, actions: readonly Action[], step: number): TractorState {
  let base = step;
  while (base > 0 && !t.checkpoints.has(base)) base--;
  let s = t.checkpoints.get(base)!;
  for (let i = base; i < step; i++) s = applyAction(s, actions[i]!);
  return s;
}

export function Replay({ viewer }: { viewer: number | null }) {
  const initial = useGameStore((s) => s.initial);
  const actions = useGameStore((s) => s.actions);
  const timeline = useMemo(() => (initial ? buildTimeline(initial, actions) : null), [initial, actions]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600);
  const max = timeline?.maxStep ?? 0;
  const state = useMemo(() => (timeline && initial ? stateAt(timeline, actions, Math.min(step, max)) : null), [timeline, initial, actions, step, max]);

  useEffect(() => {
    if (!playing) return;
    if (step >= max) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setStep((s) => Math.min(max, s + 1)), speed);
    return () => clearTimeout(t);
  }, [playing, step, max, speed]);

  if (!timeline || !state || max === 0) {
    return <p className="lobby-hint">Finish a round to replay it.</p>;
  }
  const roundIdx = Math.max(0, timeline.roundStarts.filter((s) => s <= step).length - 1);
  const go = (s: number) => setStep(Math.max(0, Math.min(max, s)));
  const prevRound = () => go(timeline.roundStarts.filter((s) => s < step).pop() ?? 0);
  const nextRound = () => go(timeline.roundStarts.find((s) => s > step) ?? max);
  const r = state.round;
  const last = state.log[state.log.length - 1];
  const t = trumpParts(r.trump);

  return (
    <div className="replay">
      <div className="replay-controls">
        <button className="secondary" onClick={prevRound} title="Previous round start">⏮</button>
        <button className="secondary" onClick={() => go(step - 1)} disabled={step === 0} title="Back one move">◀</button>
        <button onClick={() => setPlaying((p) => !p)} disabled={step >= max}>{playing ? '⏸ Pause' : '▶ Play'}</button>
        <button className="secondary" onClick={() => go(step + 1)} disabled={step >= max} title="Forward one move">▶</button>
        <button className="secondary" onClick={nextRound} title="Next round">⏭</button>
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Replay speed">
          <option value={1200}>Slow</option>
          <option value={600}>Normal</option>
          <option value={200}>Fast</option>
        </select>
        <input
          type="range"
          min={0}
          max={max}
          value={step}
          onChange={(e) => go(Number(e.target.value))}
          aria-label="Replay position"
          className="replay-slider"
        />
        <span className="lobby-hint replay-pos">
          Round {r.number} · move {step}/{max}
        </span>
      </div>
      <div className="replay-info">
        <span>
          Round {roundIdx + 1}
          {r.trump && (
            <>
              {' '}· trump <strong className={t.red ? 'red' : ''}>{t.symbol}{t.number}</strong>
            </>
          )}
          {r.landlord !== null && ` · landlord ${state.seats[r.landlord]!.name}`}
        </span>
        {last && <span className="replay-last">{last.text}</span>}
      </div>
      <Felt state={state} bottom={viewer ?? 0} />
      <AllHands state={state} />
      {(r.stage === 'play' || r.stage === 'roundOver') && r.kitty.length > 0 && (
        <div className="replay-kitty">
          <span>Buried kitty ({totalPoints(r.kitty)} pts):</span> <CardRow cards={r.kitty} size="xs" />
        </div>
      )}
    </div>
  );
}
