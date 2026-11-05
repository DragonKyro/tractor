// Pieces shared by the local lobby (LocalLobby) and the online one (OnlineLobby).

import { useEffect, useRef, useState } from 'react';
import { SEAT_COLORS, SEAT_COLOR_LABELS } from '@/core/seatColors';
import type { Seat } from '@/core/types';
import { AI_LEVELS, seatAiLevel } from '@/ai';

/** Clickable seat-color picker. Shows the current swatch; clicking opens a
 *  small popover grid of the palette. Colors already used by OTHER seats are
 *  dimmed + show a tooltip saying which seat took them. */
export function ColorSwatchPicker({ value, takenBy, disabled, onChange }: {
  value: string;
  takenBy: Record<string, string>;
  disabled?: boolean;
  onChange: (c: string) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <span style={{ position: 'relative', display: 'inline-flex' }}>
      <button
        type="button"
        className="swatch swatch-button"
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        title={SEAT_COLOR_LABELS[value] ?? value}
        aria-label={`Seat color: ${SEAT_COLOR_LABELS[value] ?? value}`}
        style={{
          background: value,
          width: 18, height: 18, borderRadius: 3, border: '1px solid rgba(0,0,0,0.25)',
          padding: 0, cursor: disabled ? 'default' : 'pointer',
          opacity: 1,
        }}
      />
      {open && !disabled && (
        <>
          <div
            onClick={() => setOpen(false)}
            style={{ position: 'fixed', inset: 0, zIndex: 10 }}
          />
          <div
            style={{
              position: 'absolute', top: 'calc(100% + 4px)', left: 0, zIndex: 11,
              background: 'var(--bg-elevated, #1e1c2e)',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 6,
              padding: 6,
              boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
              display: 'grid',
              gridTemplateColumns: 'repeat(7, 22px)',
              gap: 4,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {SEAT_COLORS.map((c) => {
              const taken = takenBy[c];
              return (
                <button
                  key={c}
                  type="button"
                  onClick={() => { onChange(c); setOpen(false); }}
                  title={taken ? `${SEAT_COLOR_LABELS[c]} — taken by ${taken}` : SEAT_COLOR_LABELS[c]}
                  disabled={!!taken}
                  style={{
                    width: 22, height: 22, borderRadius: 3, padding: 0,
                    background: c,
                    border: c === value ? '2px solid #fff' : '1px solid rgba(0,0,0,0.25)',
                    cursor: taken ? 'not-allowed' : 'pointer',
                    opacity: taken ? 0.3 : 1,
                  }}
                />
              );
            })}
          </div>
        </>
      )}
    </span>
  );
}

/** A name field that commits after a pause, on blur or on Enter — not on
 *  every keystroke (online, each commit is a write to the room). */
export function NameInput({ value, onCommit, disabled, className, maxLength = 24 }: {
  value: string;
  onCommit: (name: string) => void;
  disabled?: boolean;
  className?: string;
  maxLength?: number;
}) {
  const [draft, setDraft] = useState(value);
  const editing = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Follow the room's value unless the user is mid-edit.
  useEffect(() => {
    if (!editing.current) setDraft(value);
  }, [value]);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  const commit = (next: string) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    const clean = next.trim();
    if (clean && clean !== value) onCommit(clean);
  };
  return (
    <input
      className={className}
      value={draft}
      disabled={disabled}
      maxLength={maxLength}
      onFocus={() => { editing.current = true; }}
      onChange={(e) => {
        const next = e.target.value;
        setDraft(next);
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(() => commit(next), 600);
      }}
      onBlur={() => {
        editing.current = false;
        commit(draft);
        if (!draft.trim()) setDraft(value);
      }}
      onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
    />
  );
}

export function ChatInput({ onSend }: { onSend: (s: string) => void }) {
  const [v, setV] = useState('');
  return (
    <form
      onSubmit={(e) => { e.preventDefault(); if (v.trim()) { onSend(v); setV(''); } }}
      style={{ display: 'flex', gap: 6, marginTop: 4 }}
    >
      <input value={v} onChange={(e) => setV(e.target.value)} placeholder="Chat…" maxLength={200} style={{ flex: 1 }} />
      <button type="submit" disabled={!v.trim()}>Send</button>
    </form>
  );
}

export function RulesModal({ title, Rules, onClose }: { title: string; Rules: React.ComponentType; onClose(): void }) {
  return (
    <div className="rules-modal-backdrop" onClick={onClose}>
      <div className="rules-modal" onClick={(e) => e.stopPropagation()}>
        <div className="rules-modal-header">
          <h3>{title} — Rules</h3>
          <button className="secondary" onClick={onClose}>Close</button>
        </div>
        <div className="rules-modal-body">
          <Rules />
        </div>
      </div>
    </div>
  );
}

/** Per-seat AI difficulty picker; nothing for human seats. */
export function AiLevelSelect({ seat, disabled, onChange }: {
  seat: Seat;
  disabled?: boolean;
  onChange: (level: string) => void;
}) {
  if (!seat.isAI || AI_LEVELS.length < 2) return null;
  const value = seatAiLevel(seat);
  if (disabled) {
    return <span className="online-tag">{AI_LEVELS.find((l) => l.id === value)?.label ?? value}</span>;
  }
  return (
    <select
      className="seat-level"
      aria-label={`${seat.name} difficulty`}
      title="AI difficulty"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {AI_LEVELS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
    </select>
  );
}
