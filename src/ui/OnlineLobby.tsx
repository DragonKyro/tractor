// The lobby of an online room. Everything shown comes from the room
// (networkStore.lobby); edits are lobby transactions. Guests seat themselves
// on joining (while there's room) and edit their own seat's name and color;
// the host sets the options, adds / removes / reorders seats, and starts the
// match. Guests see the options read-only.

import { useEffect, useState } from 'react';
import { MAX_PLAYERS } from '@/game/config';
import { lobbyConfig, lobbyStartProblem, useNetworkStore } from '@/store/networkStore';
import { roomLink } from '@/net/roomLink';
import { AiLevelSelect, ChatInput, ColorSwatchPicker, NameInput, RulesModal } from './lobbyParts';
import { RulesText } from './Rules';
import { TractorOptions } from './TractorOptions';

export function OnlineLobby() {
  const lobby = useNetworkStore((s) => s.lobby);
  const role = useNetworkStore((s) => s.role);
  const myUuid = useNetworkStore((s) => s.myUuid);
  const roomCode = useNetworkStore((s) => s.roomCode);
  const people = useNetworkStore((s) => s.people);
  const onlineUuids = useNetworkStore((s) => s.onlineUuids);
  const chat = useNetworkStore((s) => s.chat);
  const net = useNetworkStore.getState;

  const [showRules, setShowRules] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [copied, setCopied] = useState<'code' | 'link' | null>(null);

  useEffect(() => {
    setStartError(null);
  }, [lobby?.configJson, lobby?.seats.length]);

  // Like the local lobby: an odd table can't have fixed partners, so the host's
  // device moves it to Finding Friends when a seat makes the count odd.
  const oddTractor = !!lobby && lobby.seats.length % 2 === 1 && lobbyConfig(lobby).gameMode === 'tractor';
  useEffect(() => {
    if (!oddTractor || role !== 'host' || !lobby) return;
    net().hostSetConfig({ ...lobbyConfig(lobby), gameMode: 'findingFriends' });
  }, [oddTractor, role, lobby, net]);

  if (!lobby) return <p>Loading the room…</p>;

  const isHost = role === 'host';
  const seats = lobby.seats;
  const config = lobbyConfig(lobby);
  const mySeat = seats.find((s) => s.id === myUuid) ?? null;
  const hostSeat = seats.find((s) => s.id === lobby.hostUuid);
  const hostOnline = onlineUuids.has(lobby.hostUuid);
  const problem = lobbyStartProblem(lobby);
  const roomFull = seats.length >= MAX_PLAYERS;

  async function start() {
    setStarting(true);
    setStartError(null);
    const err = await net().hostStartMatch();
    setStarting(false);
    if (err) setStartError(err);
  }

  function copy(what: 'code' | 'link') {
    if (!roomCode) return;
    void navigator.clipboard?.writeText(what === 'code' ? roomCode : roomLink(roomCode));
    setCopied(what);
    setTimeout(() => setCopied(null), 1500);
  }

  return (
    <div className="lobby-single">
      <div className="lobby-topbar">
        <button className="secondary" onClick={() => net().leaveRoom()}>← Leave room</button>
        <button className="secondary" onClick={() => setShowRules(true)}>📖 Rules</button>
      </div>

      <section className="lobby-panel">
        <div className="online-lobby-head">
          <h3>Tractor — online</h3>
          {roomCode && (
            <div className="online-room">
              <span className="online-label">Room</span>
              <code className="room-code">{roomCode}</code>
              <button className="secondary tiny" onClick={() => copy('code')}>{copied === 'code' ? 'Copied' : 'Copy code'}</button>
              <button className="secondary tiny" onClick={() => copy('link')}>{copied === 'link' ? 'Copied' : 'Copy invite link'}</button>
            </div>
          )}
        </div>

        <h4 className="lobby-section-h">
          Seats · {seats.length} of up to {MAX_PLAYERS} · in play order{config.gameMode === 'tractor' ? ' (partners sit opposite)' : ''}
        </h4>
        <div className="seat-list">
          {seats.map((s, i) => {
            const mine = s.id === myUuid;
            const editable = mine || (isHost && s.isAI);
            const offline = !s.isAI && !onlineUuids.has(s.id);
            return (
              <div key={s.id} className="seat-row">
                <span className="seat-index">{i + 1}</span>
                <ColorSwatchPicker
                  value={s.color}
                  takenBy={Object.fromEntries(seats.filter((x) => x.id !== s.id).map((x) => [x.color, x.name]))}
                  disabled={!editable}
                  onChange={(color) => net().editSeat(s.id, { color })}
                />
                {editable ? (
                  <NameInput className="seat-name" value={s.name} onCommit={(name) => net().editSeat(s.id, { name })} />
                ) : (
                  <span className="seat-name online-seat-name">{s.name}</span>
                )}
                <span className="online-seat-tags">
                  {mine && <span className="online-tag">you</span>}
                  {s.id === lobby.hostUuid && <span className="online-tag">host</span>}
                  {s.isAI && <span className="online-tag">AI</span>}
                  <AiLevelSelect seat={s} disabled={!isHost} onChange={(aiLevel) => net().editSeat(s.id, { aiLevel })} />
                  {offline && <span className="online-tag offline">offline</span>}
                </span>
                {isHost && (
                  <span className="seat-move">
                    <button className="secondary tiny" disabled={i === 0} onClick={() => net().hostMoveSeat(s.id, -1)} aria-label="Move up">↑</button>
                    <button className="secondary tiny" disabled={i === seats.length - 1} onClick={() => net().hostMoveSeat(s.id, 1)} aria-label="Move down">↓</button>
                  </span>
                )}
                {isHost && s.id !== lobby.hostUuid && (
                  <button className="secondary" title="Remove seat" onClick={() => net().hostRemoveSeat(s.id)}>×</button>
                )}
              </div>
            );
          })}
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {isHost && <button className="secondary" onClick={() => net().hostAddAISeat()} disabled={roomFull}>+ Add AI</button>}
          {!mySeat && (
            <button className="secondary" onClick={() => net().takeSeat()} disabled={roomFull}>
              {roomFull ? 'Table full — watching' : 'Take a seat'}
            </button>
          )}
        </div>

        <h4 className="lobby-section-h">Options{isHost ? '' : ' (set by the host)'}</h4>
        <fieldset className="online-options" disabled={!isHost}>
          <TractorOptions config={config} seats={seats} readOnly={!isHost} onChange={(c) => isHost && net().hostSetConfig(c)} />
        </fieldset>

        <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          {isHost ? (
            <button onClick={start} disabled={!!problem || starting}>{starting ? 'Starting…' : 'Start match'}</button>
          ) : (
            <span className="lobby-hint" style={{ margin: 0 }}>
              {hostOnline
                ? `Waiting for ${hostSeat?.name ?? 'the host'} to start the match.`
                : `${hostSeat?.name ?? 'The host'} is offline; the match starts when they're back.`}
            </span>
          )}
        </div>
        {isHost && startError && <div className="online-error">{startError}</div>}

        <div className="online-peers">
          <h4>In the room</h4>
          <ul>
            {people.map((p) => {
              const seated = seats.some((s) => s.id === p.uuid);
              return (
                <li key={p.uuid}>
                  <span className={`peer-dot ${seated ? 'seated' : 'spectator'}`} />
                  {seats.find((s) => s.id === p.uuid)?.name ?? p.name}
                  {p.uuid === myUuid && <em>(you)</em>}
                  {!seated && <span className="online-tag">watching</span>}
                </li>
              );
            })}
          </ul>
        </div>
        <div className="lobby-chat">
          <h4>Chat</h4>
          <div className="lobby-chat-log">
            {chat.slice(-8).map((m) => (
              <div key={m.id} className={m.kind === 'system' ? 'chat-system' : undefined}>
                {m.kind === 'system' ? m.text : (
                  <>
                    <strong>{m.name}:</strong> {m.text}
                  </>
                )}
              </div>
            ))}
          </div>
          <ChatInput onSend={(t) => net().sendChat(t)} />
        </div>
      </section>

      {showRules && <RulesModal title="Tractor" Rules={RulesText} onClose={() => setShowRules(false)} />}
    </div>
  );
}
