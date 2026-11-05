// The in-game view. Gets the state, which seat's hand to show (null for a
// spectator or a pass-the-device screen) and dispatch; never reads the
// stores itself, so hot-seat / online / spectating all render the same way.

import { useEffect, useMemo, useState } from 'react';
import { activeSeat, attackerPointsSoFar, displayTrump, scoringParams, sortedHand } from '@/game/engine';
import { materialize } from '@/game/scoring';
import type { Action, TractorState } from '@/game/types';
import { ActionPanel, indicesOf } from './ActionPanel';
import { Felt } from './Felt';
import { Hand } from './Hand';
import { RoundSummary } from './RoundSummary';
import { SidePanel } from './SidePanel';
import { levelText, trumpParts } from './labels';
import { bidText } from '@/game/engine';

export interface SummaryControls {
  onNextRound?: () => void;
  onPlayAgain?: () => void;
  onLobby?: () => void;
  waitingText?: string;
}

interface Props {
  state: TractorState;
  viewer: number | null;
  dispatch: (a: Action) => string | null;
  summary: SummaryControls;
}

export function Table({ state, viewer, dispatch, summary }: Props) {
  const r = state.round;
  const hand = useMemo(() => (viewer !== null ? sortedHand(state, viewer) : []), [state, viewer]);
  const trump = r.trump ?? displayTrump(state, viewer ?? 0);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const handKey = `${viewer}|${r.number}|${r.stage}|${hand.join(',')}`;
  useEffect(() => setSelected(new Set()), [handKey]);
  const [hideSummary, setHideSummary] = useState(false);
  useEffect(() => setHideSummary(false), [r.number, r.stage]);

  const active = activeSeat(state);
  const canSelect = viewer !== null && ((r.stage === 'play' && active === viewer) || (r.stage === 'exchange' && r.landlord === viewer));
  const marked = useMemo(
    () => (r.stage === 'exchange' && viewer !== null && r.landlord === viewer ? indicesOf(hand, r.dealtKitty) : undefined),
    [r.stage, r.landlord, r.dealtKitty, hand, viewer],
  );

  function toggle(i: number) {
    const next = new Set(selected);
    if (next.has(i)) next.delete(i);
    else next.add(i);
    setSelected(next);
  }

  return (
    <div className="table-layout">
      <div className="table-main">
        <StatusBar state={state} />
        <Felt state={state} bottom={viewer ?? 0} />
        {viewer !== null && (
          <>
            <ActionPanel state={state} viewer={viewer} hand={hand} selected={selected} setSelected={setSelected} dispatch={dispatch} />
            <Hand cards={hand} trump={trump} selected={selected} onToggle={canSelect ? toggle : undefined} marked={marked} markLabel="kitty" />
          </>
        )}
        {viewer === null && r.stage !== 'roundOver' && <div className="action-panel"><div className="action-prompt">Watching.</div></div>}
      </div>
      <SidePanel state={state} />
      {r.stage === 'roundOver' && !hideSummary && <RoundSummary state={state} {...summary} onHide={() => setHideSummary(true)} />}
      {r.stage === 'roundOver' && hideSummary && (
        <button className="summary-reopen" onClick={() => setHideSummary(false)}>Show the round's result</button>
      )}
    </div>
  );
}

function StatusBar({ state }: { state: TractorState }) {
  const r = state.round;
  const t = trumpParts(r.trump);
  const m = materialize(scoringParams(state.config), state.rules.numDecks, state.rules.numDecks * 100);
  const threshold = m.landlordWins[m.landlordWins.length - 1]!.end;
  const landlord = r.landlord ?? state.landlord;
  const lead = r.autobid ?? r.bids[r.bids.length - 1] ?? null;
  const number = landlord !== null ? levelText(state.levels[landlord]!) : null;
  return (
    <div className="status-bar">
      <span className="status-item">Round {r.number}</span>
      <span className="status-item status-trump" title={t.label}>
        Trump{' '}
        {r.trump ? (
          <strong className={t.red ? 'red' : ''}>
            {t.symbol}
            {t.number}
          </strong>
        ) : lead ? (
          <span>
            {bidText(lead)} by {state.seats[lead.seat]!.name}
            {number ? ` (${number}s)` : ''}
          </span>
        ) : (
          <span className="lobby-hint">{number ? `${number}s, suit not declared` : 'not declared'}</span>
        )}
      </span>
      {landlord !== null && (
        <span className="status-item">
          👑 {state.seats[landlord]!.name} <span className="lobby-hint">at {levelText(state.levels[landlord]!)}</span>
        </span>
      )}
      {(r.stage === 'play' || r.stage === 'roundOver') && (
        <span className="status-item status-points" title="Points the attackers have captured / needed to take over">
          Attackers <strong>{r.stage === 'roundOver' && r.result ? r.result.attackerPoints : attackerPointsSoFar(state)}</strong> / {threshold}
        </span>
      )}
      {r.stage === 'play' && (
        <span className="status-item lobby-hint">
          Trick {r.tricks.length + 1} · {r.hands.reduce((n, h) => Math.max(n, h.length), 0)} left
        </span>
      )}
    </div>
  );
}
