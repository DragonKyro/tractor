// The in-game view. Gets the state, which seat's hand to show (null for a
// spectator or a pass-the-device screen) and dispatch; never reads the game
// stores itself, so hot-seat / online / spectating all render the same way.

import { useEffect, useMemo, useState } from 'react';
import { activeSeat, attackerPointsSoFar, bidText, displayTrump, ordinalText, scoringParams, sortedHand } from '@/game/engine';
import { cardLabel, cardRank, cardSuit, isJoker, type Card } from '@/game/cards';
import { effectiveSuit, isTrump } from '@/game/trump';
import { materialize } from '@/game/scoring';
import type { Action, TractorState } from '@/game/types';
import { CardRow } from '../CardView';
import { SuitText } from '../SuitText';
import { ActionPanel, indicesOf } from './ActionPanel';
import { Felt } from './Felt';
import { Hand } from './Hand';
import { RoundSummary } from './RoundSummary';
import { SidePanel } from './SidePanel';
import { captureLaunch } from './flight';
import { levelText, trumpParts } from './labels';

export interface SummaryControls {
  onNextRound?: () => void;
  onPlayAgain?: () => void;
  onLobby?: () => void;
  onReview?: (kind: 'stats' | 'replay') => void;
  waitingText?: string;
}

/** A pending automated step the table can count down to (known on the driving device). */
export interface Automation {
  kind: 'bidWindow' | 'handoff';
  firesAt: number;
  seat: number | null;
}

interface Props {
  state: TractorState;
  viewer: number | null;
  dispatch: (a: Action) => string | null;
  summary: SummaryControls;
  automation?: Automation | null;
  /** Watch mode (all-AI table): every hand face up. */
  showAllHands?: boolean;
  canShowAllHands?: boolean;
}

export function Table({ state, viewer, dispatch, summary, automation = null, showAllHands = false, canShowAllHands = false }: Props) {
  const r = state.round;
  const hand = useMemo(() => (viewer !== null ? sortedHand(state, viewer) : []), [state, viewer]);
  // Trump is settled once the kitty is taken; until then the hand stays in printed suits (so suit lengths show
  // for overcalling) with the trump numbers and jokers marked.
  const trumpNumber = (r.trump ?? displayTrump(state, viewer ?? 0)).number;
  const groupOf = (c: Card) => (r.trump ? effectiveSuit(r.trump, c) : isJoker(c) ? 'J' : cardSuit(c)!);
  const isTrumpCard = (c: Card) => (r.trump ? isTrump(r.trump, c) : isJoker(c) || (trumpNumber !== null && cardRank(c) === trumpNumber));
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const handKey = `${viewer}|${r.number}|${r.stage}|${hand.join(',')}`;
  useEffect(() => setSelected(new Set()), [handKey]);
  const [hideSummary, setHideSummary] = useState(false);
  useEffect(() => setHideSummary(false), [r.number, r.stage]);
  const [sideOpen, setSideOpen] = useState(false);
  // The viewer's plays fly from the hand: note where the cards are just before they leave it.
  const send = (a: Action) => {
    if (a.type === 'play' && a.seat === viewer) captureLaunch(viewer);
    return dispatch(a);
  };

  const active = activeSeat(state);
  const canSelect = viewer !== null && ((r.stage === 'play' && active === viewer) || (r.stage === 'exchange' && r.landlord === viewer));
  const marked = useMemo(
    () => (r.stage === 'exchange' && viewer !== null && r.landlord === viewer ? indicesOf(hand, r.dealtKitty) : undefined),
    [r.stage, r.landlord, r.dealtKitty, hand, viewer],
  );

  return (
    <div className={`table-layout${sideOpen ? ' side-open' : ''}`}>
      <div className="table-main">
        <StatusBar state={state} onToggleSide={() => setSideOpen((v) => !v)} />
        <TeamBanner state={state} viewer={viewer} />
        <Felt state={state} bottom={viewer ?? 0} live />
        {viewer !== null && (
          <>
            <ActionPanel
              state={state}
              viewer={viewer}
              hand={hand}
              selected={selected}
              setSelected={setSelected}
              dispatch={send}
              automation={automation}
            />
            <Hand cards={hand} groupOf={groupOf} isTrumpCard={isTrumpCard} selected={selected} onSelect={canSelect ? setSelected : undefined} marked={marked} markLabel="kitty" />
          </>
        )}
        {viewer === null && r.stage !== 'roundOver' && (
          <div className="action-panel">
            <div className="action-prompt">Watching.{canShowAllHands && !showAllHands ? ' Turn on “Show all hands” in Settings to see every hand.' : ''}</div>
          </div>
        )}
        {showAllHands && <AllHands state={state} />}
      </div>
      <SidePanel state={state} canShowAllHands={canShowAllHands} onClose={() => setSideOpen(false)} />
      {r.stage === 'roundOver' && !hideSummary && <RoundSummary state={state} viewer={viewer} {...summary} onHide={() => setHideSummary(true)} />}
      {r.stage === 'roundOver' && hideSummary && (
        <button className="summary-reopen" onClick={() => setHideSummary(false)}>Show the round's result</button>
      )}
    </div>
  );
}

function StatusBar({ state, onToggleSide }: { state: TractorState; onToggleSide: () => void }) {
  const r = state.round;
  const t = trumpParts(r.trump);
  const m = materialize(scoringParams(state.config), state.rules.numDecks, state.rules.numDecks * 100);
  const threshold = m.landlordWins[m.landlordWins.length - 1]!.end;
  const landlord = r.landlord ?? state.landlord;
  const lead = r.autobid ?? r.bids[r.bids.length - 1] ?? null;
  const number = landlord !== null ? levelText(state.levels[landlord]!) : null;
  const points = r.stage === 'roundOver' && r.result ? r.result.attackerPoints : attackerPointsSoFar(state);
  return (
    <div className="status-bar">
      <span className="status-item status-round">Round {r.number}</span>
      <span className="status-item status-trump" title={t.label}>
        <span className="status-label">Trump</span>
        {r.trump ? (
          <strong key={`${t.symbol}${t.number}`} className="trump-chip on-card">
            <span className={t.suitClass}>{t.symbol}</span>
            {t.number}
          </strong>
        ) : lead ? (
          <span>
            <SuitText text={bidText(lead)} /> by {state.seats[lead.seat]!.name}
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
          Attackers <strong key={points}>{points}</strong> / {threshold}
          <span className={`points-meter${points >= threshold ? ' points-meter-over' : ''}`} aria-hidden>
            <span style={{ width: `${Math.min(100, (points / Math.max(1, threshold)) * 100)}%` }} />
          </span>
        </span>
      )}
      {r.stage === 'play' && (
        <span className="status-item lobby-hint">
          Trick {r.tricks.length + 1} · {r.hands.reduce((n, h) => Math.max(n, h.length), 0)} left
        </span>
      )}
      <button className="secondary tiny side-toggle" onClick={onToggleSide} aria-label="Table, log and settings">☰</button>
    </div>
  );
}

/** Who's with whom, from the viewer's side of the table (and what a held friend card means). */
function TeamBanner({ state, viewer }: { state: TractorState; viewer: number | null }) {
  const r = state.round;
  if (viewer === null || r.landlord === null || r.stage !== 'play') return null;
  const names = (seats: number[]) => seats.map((i) => (i === viewer ? 'you' : state.seats[i]!.name)).join(', ');
  const all = state.seats.map((_, i) => i);
  if (state.rules.gameMode === 'tractor') {
    const defenders = all.filter((i) => i % 2 === r.landlord! % 2);
    const attackers = all.filter((i) => !defenders.includes(i));
    const mine = defenders.includes(viewer);
    return (
      <div className={`team-banner ${mine ? 'team-defend' : 'team-attack'}`}>
        <strong>{mine ? 'Defending' : 'Attacking'}</strong> with {names((mine ? defenders : attackers).filter((i) => i !== viewer)) || 'nobody'} · against{' '}
        {names(mine ? attackers : defenders)}
      </div>
    );
  }
  const hidden = r.friends.filter((f) => f.seat === null);
  const onTeam = r.landlordTeam.includes(viewer);
  const team = r.landlordTeam;
  const knownAttackers = hidden.length === 0 ? all.filter((i) => !team.includes(i)) : [];
  const myCalls = hidden.filter((f) => r.hands[viewer]!.includes(f.card));
  return (
    <div className={`team-banner ${onTeam ? 'team-defend' : myCalls.length ? 'team-maybe' : 'team-attack'}`}>
      {onTeam ? (
        <span>
          <strong>{viewer === r.landlord ? "You're the landlord" : "You're the landlord's friend"}</strong> · team: {names(team)}
        </span>
      ) : (
        <span>
          <strong>Attacking</strong>
          {knownAttackers.length ? ` with ${names(knownAttackers.filter((i) => i !== viewer)) || 'nobody'}` : ''} · landlord's team: {names(team)}
        </span>
      )}
      {hidden.length > 0 && (
        <span className="team-hidden">
          {' '}· {hidden.length} friend{hidden.length === 1 ? '' : 's'} still hidden (<SuitText text={hidden.map((f) => `the ${ordinalText(f.ordinal)} ${cardLabel(f.card)}`).join(', ')} />)
        </span>
      )}
      {!onTeam && myCalls.map((f, i) => (
        <div key={i} className="team-hint">
          You hold the called <SuitText text={cardLabel(f.card)} />:{' '}
          {f.skip === 0 ? 'if the next one played is yours, you join the landlord.' : `${f.skip} more cop${f.skip === 1 ? 'y has' : 'ies have'} to be played before the one that counts.`}
        </div>
      ))}
    </div>
  );
}

/** Watch mode / review: every seat's hand face up. */
export function AllHands({ state }: { state: TractorState }) {
  return (
    <div className="all-hands">
      {state.seats.map((s, i) => (
        <div key={s.id} className="all-hands-row">
          <span className="all-hands-name">
            <span className="seat-dot" style={{ background: s.color }} /> {s.name}
          </span>
          <CardRow cards={sortedHand(state, i)} size="sm" />
        </div>
      ))}
    </div>
  );
}
