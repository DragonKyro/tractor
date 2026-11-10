// Dependency-free SVG charts for the match review, built to the dataviz
// skill's specs: an emphasis line chart (one highlighted series in the accent
// blue, the rest in a recessive gray — seat colors are player-picked and
// unbounded in number, so they don't carry the chart) and single-hue columns
// with a labeled reference line. 2px lines, ≥8px end-dots with a surface ring,
// ≤24px columns with 4px rounded caps, hairline grid, a crosshair / per-mark
// tooltip on hover and keyboard focus, and a table view for every chart.
// Validated (scripts/validate_palette.js, dark, surface #232730): accent
// #3987e5 and context gray #8a8c93 both clear 3:1; adjacent CVD ΔE 13.7.

import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';

export const ACCENT = '#3987e5';
export const CONTEXT = '#8a8c93';
const SURFACE = '#232730';
const GRID = 'rgba(255,255,255,0.07)';
const AXIS = 'rgba(255,255,255,0.22)';

function useWidth(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(560);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const m = () => setW(Math.max(260, el.clientWidth));
    m();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(m) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, []);
  return [ref, w];
}

/** A chart card with a Chart / Table switch, so no value is hover-only. */
export function ChartCard({ title, subtitle, chart, table }: { title: string; subtitle?: string; chart: ReactNode; table: ReactNode }) {
  const [view, setView] = useState<'chart' | 'table'>('chart');
  return (
    <section className="chart-card">
      <div className="chart-card-head">
        <div>
          <h4>{title}</h4>
          {subtitle && <div className="lobby-hint">{subtitle}</div>}
        </div>
        <div className="chart-switch" role="tablist">
          {(['chart', 'table'] as const).map((v) => (
            <button key={v} role="tab" aria-selected={view === v} className={`secondary tiny${view === v ? ' active' : ''}`} onClick={() => setView(v)}>
              {v === 'chart' ? 'Chart' : 'Table'}
            </button>
          ))}
        </div>
      </div>
      {view === 'chart' ? chart : table}
    </section>
  );
}

export interface LineSeries {
  id: string;
  name: string;
  /** y per x index (null = no value). */
  values: Array<number | null>;
}

/** Emphasis line chart: `highlight` in the accent, the rest gray; legend chips pick the highlight. */
export function EmphasisLines({
  series,
  xLabels,
  yTicks,
  yLabel,
  highlight,
  onHighlight,
  xTitle,
}: {
  series: LineSeries[];
  xLabels: string[];
  yTicks: Array<{ value: number; label: string }>;
  yLabel: (v: number) => string;
  highlight: string;
  onHighlight: (id: string) => void;
  xTitle: string;
}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const H = 220;
  const pad = { l: 34, r: 92, t: 12, b: 34 };
  const plotW = width - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const ys = yTicks.map((t) => t.value);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const n = xLabels.length;
  const x = (i: number) => pad.l + (n <= 1 ? plotW / 2 : (i * plotW) / (n - 1));
  const y = (v: number) => pad.t + plotH - ((v - yMin) / Math.max(1, yMax - yMin)) * plotH;
  const path = (vals: Array<number | null>) =>
    vals.map((v, i) => (v === null ? '' : `${i === 0 || vals[i - 1] === null ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`)).join(' ');
  const hi = series.find((s) => s.id === highlight) ?? series[0]!;
  const ordered = [...series.filter((s) => s.id !== hi.id), hi];
  const lastIdx = (vals: Array<number | null>) => vals.reduce<number>((acc, v, i) => (v === null ? acc : i), -1);
  const hiLast = lastIdx(hi.values);

  const onMove = (clientX: number, rect: DOMRect) => {
    const px = clientX - rect.left;
    const i = n <= 1 ? 0 : Math.round(((px - pad.l) / plotW) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };

  return (
    <div className="chart" ref={ref}>
      <div className="chart-legend" role="group" aria-label="Highlight a series">
        {series.map((s) => (
          <button
            key={s.id}
            className={`chart-chip${s.id === hi.id ? ' active' : ''}`}
            onClick={() => onHighlight(s.id)}
            aria-pressed={s.id === hi.id}
          >
            <span className="chart-linekey" style={{ background: s.id === hi.id ? ACCENT : CONTEXT }} />
            {s.name}
          </button>
        ))}
      </div>
      <svg
        width={width}
        height={H}
        role="img"
        aria-label={`${hi.name}: ${hi.values.map((v, i) => `${xLabels[i]} ${v === null ? '–' : yLabel(v)}`).join(', ')}`}
        onPointerMove={(e) => onMove(e.clientX, e.currentTarget.getBoundingClientRect())}
        onPointerLeave={() => setHover(null)}
      >
        {yTicks.map((t) => (
          <g key={t.value}>
            <line x1={pad.l} x2={pad.l + plotW} y1={y(t.value)} y2={y(t.value)} stroke={GRID} strokeWidth={1} />
            <text x={pad.l - 6} y={y(t.value) + 4} textAnchor="end" className="chart-tick">{t.label}</text>
          </g>
        ))}
        <line x1={pad.l} x2={pad.l + plotW} y1={pad.t + plotH} y2={pad.t + plotH} stroke={AXIS} strokeWidth={1} />
        {xLabels.map((l, i) => (
          <text key={i} x={x(i)} y={H - pad.b + 16} textAnchor="middle" className="chart-tick">{l}</text>
        ))}
        <text x={pad.l + plotW / 2} y={H - 4} textAnchor="middle" className="chart-axis-title">{xTitle}</text>
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={pad.t + plotH} stroke={AXIS} strokeWidth={1} />}
        {ordered.map((s) => (
          <path
            key={s.id}
            d={path(s.values)}
            fill="none"
            stroke={s.id === hi.id ? ACCENT : CONTEXT}
            strokeOpacity={s.id === hi.id ? 1 : 0.55}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
        {hiLast >= 0 && (
          <>
            <circle cx={x(hiLast)} cy={y(hi.values[hiLast]!)} r={6} fill={SURFACE} />
            <circle cx={x(hiLast)} cy={y(hi.values[hiLast]!)} r={4} fill={ACCENT} />
            <text x={x(hiLast) + 10} y={y(hi.values[hiLast]!) + 4} className="chart-direct-label">
              {hi.name} · {yLabel(hi.values[hiLast]!)}
            </text>
          </>
        )}
        {hover !== null &&
          ordered.map((s) => {
            const v = s.values[hover];
            if (v === null || v === undefined) return null;
            return <circle key={s.id} cx={x(hover)} cy={y(v)} r={4} fill={s.id === hi.id ? ACCENT : CONTEXT} stroke={SURFACE} strokeWidth={2} />;
          })}
      </svg>
      {hover !== null && (
        <div className="chart-tooltip" style={{ left: Math.min(width - 170, Math.max(0, x(hover) + 10)) }}>
          <div className="chart-tooltip-title">{xLabels[hover]}</div>
          {[...series]
            .map((s) => ({ s, v: s.values[hover] }))
            .filter((r) => r.v !== null && r.v !== undefined)
            .sort((a, b) => b.v! - a.v!)
            .map(({ s, v }) => (
              <div key={s.id} className="chart-tooltip-row">
                <span className="chart-linekey" style={{ background: s.id === hi.id ? ACCENT : CONTEXT }} />
                <strong>{yLabel(v!)}</strong> <span className="lobby-hint">{s.name}</span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

export interface Column {
  label: string;
  value: number;
  /** Tooltip lines (after the value). */
  detail: string;
}

/** Single-hue columns from zero with a labeled reference line (e.g. the take-over threshold). */
export function ColumnsWithReference({ columns, reference, referenceLabel, yTitle }: { columns: Column[]; reference: number; referenceLabel: string; yTitle: string }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const H = 220;
  const pad = { l: 40, r: 16, t: 14, b: 34 };
  const plotW = width - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const top = Math.max(reference, ...columns.map((c) => c.value), 10);
  const step = niceStep(top / 4);
  const yMax = Math.ceil(top / step) * step;
  const ticks: number[] = [];
  for (let v = 0; v <= yMax; v += step) ticks.push(v);
  const band = plotW / Math.max(1, columns.length);
  const bw = Math.min(24, band * 0.6);
  const y = (v: number) => pad.t + plotH - (Math.max(0, v) / yMax) * plotH;
  const base = y(0);

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={H} role="img" aria-label={`${yTitle} by round, reference ${referenceLabel}`} onPointerLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={pad.l + plotW} y1={y(t)} y2={y(t)} stroke={GRID} strokeWidth={1} />
            <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" className="chart-tick">{t}</text>
          </g>
        ))}
        {columns.map((c, i) => {
          const cx = pad.l + band * i + band / 2;
          const h = Math.max(0, base - y(c.value));
          const r = Math.min(4, h, bw / 2);
          const x0 = cx - bw / 2;
          const yTop = base - h;
          // Rounded data end (top), square at the baseline.
          const d = h <= 0 ? '' : `M${x0},${base} V${yTop + r} Q${x0},${yTop} ${x0 + r},${yTop} H${x0 + bw - r} Q${x0 + bw},${yTop} ${x0 + bw},${yTop + r} V${base} Z`;
          return (
            <g
              key={i}
              tabIndex={0}
              role="img"
              aria-label={`${c.label}: ${c.value}. ${c.detail}`}
              onPointerEnter={() => setHover(i)}
              onFocus={() => setHover(i)}
              onBlur={() => setHover(null)}
              className="chart-hit"
            >
              <rect x={cx - band / 2} y={pad.t} width={band} height={plotH} fill="transparent" />
              {d && <path d={d} fill={ACCENT} opacity={hover === null || hover === i ? 1 : 0.55} />}
              <text x={cx} y={H - pad.b + 16} textAnchor="middle" className="chart-tick">{c.label}</text>
            </g>
          );
        })}
        <line x1={pad.l} x2={pad.l + plotW} y1={base} y2={base} stroke={AXIS} strokeWidth={1} />
        <line x1={pad.l} x2={pad.l + plotW} y1={y(reference)} y2={y(reference)} stroke="#c3c2b7" strokeWidth={1} strokeDasharray="4 3" />
        <text x={pad.l + plotW} y={y(reference) - 5} textAnchor="end" className="chart-ref-label">{referenceLabel}</text>
        <text x={pad.l + plotW / 2} y={H - 4} textAnchor="middle" className="chart-axis-title">Round</text>
      </svg>
      {hover !== null && columns[hover] && (
        <div className="chart-tooltip" style={{ left: Math.min(width - 190, Math.max(0, pad.l + band * hover + band / 2 + 10)) }}>
          <div className="chart-tooltip-title">{columns[hover]!.label}</div>
          <div className="chart-tooltip-row">
            <strong>{columns[hover]!.value}</strong> <span className="lobby-hint">{yTitle.toLowerCase()}</span>
          </div>
          <div className="lobby-hint">{columns[hover]!.detail}</div>
        </div>
      )}
    </div>
  );
}

function niceStep(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(Math.max(1, raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p;
  return 10 * p;
}
