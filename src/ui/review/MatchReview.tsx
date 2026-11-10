// The match review: stats (levels over the rounds, attackers' points against
// the threshold, the round-by-round table) and a move-by-move replay of the
// finished rounds. Opens from the round summary, the game-over screen, or the
// top bar's Stats button.

import { useState } from 'react';
import { levelIndex, levelLabel, LEVELS } from '@/game/cards';
import { scoringParams } from '@/game/engine';
import { materialize } from '@/game/scoring';
import type { TractorState } from '@/game/types';
import { trumpParts } from '../table/labels';
import { ChartCard, ColumnsWithReference, EmphasisLines, type LineSeries } from './charts';
import { Replay } from './Replay';

interface Props {
  state: TractorState;
  initialTab: 'stats' | 'replay';
  viewer: number | null;
  onClose(): void;
}

export function MatchReview({ state, initialTab, viewer, onClose }: Props) {
  const [tab, setTab] = useState(initialTab);
  return (
    <div className="rules-modal-backdrop" onClick={onClose}>
      <div className="rules-modal review-modal" onClick={(e) => e.stopPropagation()}>
        <div className="rules-modal-header">
          <div className="side-tabs review-tabs" role="tablist">
            {(['stats', 'replay'] as const).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'side-tab active' : 'side-tab'} onClick={() => setTab(t)}>
                {t === 'stats' ? '📊 Match stats' : '⏪ Replay'}
              </button>
            ))}
          </div>
          <button className="secondary" onClick={onClose}>Close</button>
        </div>
        <div className="rules-modal-body">{tab === 'stats' ? <Stats state={state} viewer={viewer} /> : <Replay viewer={viewer} />}</div>
      </div>
    </div>
  );
}

function Stats({ state, viewer }: { state: TractorState; viewer: number | null }) {
  const h = state.history;
  const n = state.seats.length;
  const tractor = state.rules.gameMode === 'tractor';
  const maxIdx = levelIndex(state.config.maxRank);
  const startIdx = levelIndex(state.config.startingRank);
  const won = maxIdx + 1;
  // Series: teams in classic tractor (partners share levels), players otherwise.
  const groups: Array<{ id: string; name: string; seats: number[] }> = tractor
    ? [0, 1].filter((p) => p < n).map((p) => {
        const seats = state.seats.map((_, i) => i).filter((i) => i % 2 === p);
        return { id: `team${p}`, name: seats.map((i) => state.seats[i]!.name).join(' & '), seats };
      })
    : state.seats.map((s, i) => ({ id: `seat${i}`, name: s.name, seats: [i] }));
  const series: LineSeries[] = groups.map((g) => {
    const seat = g.seats[0]!;
    const values: Array<number | null> = [startIdx];
    for (const r of h) {
      const a = r.advancements[seat]!;
      values.push(a.finished ? won : levelIndex(a.to));
    }
    return { id: g.id, name: g.name, values };
  });
  const mine = groups.find((g) => viewer !== null && g.seats.includes(viewer))?.id ?? groups[0]!.id;
  const [highlight, setHighlight] = useState(mine);
  const anyWon = h.some((r) => r.winners.length > 0);
  const top = anyWon ? won : Math.max(...series.flatMap((s) => s.values.filter((v): v is number => v !== null)), startIdx + 1);
  const yTicks = [];
  for (let v = startIdx; v <= top; v++) yTicks.push({ value: v, label: v === won ? 'Won' : levelLabel(LEVELS[v]!) });
  const yLabel = (v: number) => (v === won ? 'won' : levelLabel(LEVELS[v]!));
  const xLabels = ['Start', ...h.map((r) => `R${r.round}`)];

  const m = materialize(scoringParams(state.config), state.rules.numDecks, state.rules.numDecks * 100);
  const threshold = m.landlordWins[m.landlordWins.length - 1]!.end;
  const held = h.filter((r) => r.landlordWon).length;
  const best = h.reduce((b, r) => Math.max(b, r.attackerPoints), 0);

  if (h.length === 0) return <p className="lobby-hint">Stats appear once a round has been played.</p>;

  return (
    <div className="stats">
      <div className="kpi-row">
        <div className="kpi">
          <div className="kpi-label">Rounds played</div>
          <div className="kpi-value">{h.length}</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Landlord's side held</div>
          <div className="kpi-value">
            {held} <span className="kpi-sub">of {h.length}</span>
          </div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Best attack</div>
          <div className="kpi-value">
            {best} <span className="kpi-sub">pts</span>
          </div>
        </div>
      </div>

      <ChartCard
        title="Levels after each round"
        subtitle={tractor ? 'Partners share a level, so each line is a team.' : 'Pick a player to highlight.'}
        chart={<EmphasisLines series={series} xLabels={xLabels} yTicks={yTicks} yLabel={yLabel} highlight={highlight} onHighlight={setHighlight} xTitle="Round" />}
        table={
          <div className="table-scroll">
            <table className="players-table stats-table">
              <thead>
                <tr>
                  <th>{tractor ? 'Team' : 'Player'}</th>
                  {xLabels.map((l) => (
                    <th key={l}>{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {series.map((s) => (
                  <tr key={s.id}>
                    <td>{s.name}</td>
                    {s.values.map((v, i) => (
                      <td key={i}>{v === null ? '–' : yLabel(v)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        }
      />

      <ChartCard
        title="Attackers' points each round"
        subtitle={`${threshold} or more takes the landlord seat.`}
        chart={
          <ColumnsWithReference
            columns={h.map((r) => ({
              label: `R${r.round}`,
              value: Math.max(0, r.attackerPoints),
              detail: `${state.seats[r.landlord]!.name} landlord · ${r.landlordWon ? `held (+${r.landlordDelta})` : `taken over${r.attackerDelta ? ` (+${r.attackerDelta})` : ''}`}`,
            }))}
            reference={threshold}
            referenceLabel={`take over at ${threshold}`}
            yTitle="Attackers' points"
          />
        }
        table={<RoundsTable state={state} />}
      />
      <h4 className="lobby-section-h">Every round</h4>
      <RoundsTable state={state} />
    </div>
  );
}

function RoundsTable({ state }: { state: TractorState }) {
  return (
    <div className="table-scroll">
      <table className="players-table stats-table">
        <thead>
          <tr>
            <th>Round</th>
            <th>Landlord</th>
            <th>Trump</th>
            <th>Attackers</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {state.history.map((r) => {
            const t = trumpParts(r.trump);
            return (
              <tr key={r.round}>
                <td>{r.round}</td>
                <td>{state.seats[r.landlord]!.name}</td>
                <td className={t.red ? 'red' : ''}>
                  {t.symbol}
                  {t.number}
                </td>
                <td>
                  {r.attackerPoints}
                  {r.endedEarly ? ' (early)' : ''}
                </td>
                <td>
                  {r.landlordWon
                    ? `Held, +${r.landlordDelta}${r.bonusLevel ? ' (bonus)' : ''}`
                    : r.attackerDelta
                      ? `Taken over, +${r.attackerDelta}`
                      : 'Taken over'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
