// The small online screens around the lobby and the game: the picker's
// "join a room" box, the connecting card and the error card, plus the hook
// that rejoins the room in the URL after a reload or from an invite link.

import { useEffect, useState } from 'react';
import { useNetworkStore, warmUpNetwork } from '@/store/networkStore';
import { normalizeRoomCode, roomCodeFromUrl } from '@/net/roomLink';
import { getSavedDisplayName } from '@/net/identity';
import { isOnlineConfigured } from '@/net/firebaseConfig';

let rejoinTried = false;

/**
 * A room code in the URL (?room=ABCD — a reload, or an invite link) joins it
 * at once when this device has a saved name. Returns the code otherwise, to
 * prefill the join box.
 */
export function useRejoinFromUrl(): string | null {
  const [pending] = useState(() => roomCodeFromUrl());
  useEffect(() => {
    if (rejoinTried || !pending || !isOnlineConfigured()) return;
    rejoinTried = true; // once per page (StrictMode runs effects twice)
    const name = getSavedDisplayName();
    if (name) void useNetworkStore.getState().joinRoom(pending, name);
  }, [pending]);
  return pending;
}

export function JoinRoomBox({ initialCode }: { initialCode?: string | null }) {
  const savedName = useNetworkStore((s) => s.myDisplayName);
  const [name, setName] = useState(savedName);
  const [code, setCode] = useState(initialCode ?? '');
  const clean = normalizeRoomCode(code);
  const canJoin = clean.length >= 4 && name.trim().length > 0;
  return (
    <form
      className="online-join join-room-box"
      onPointerEnter={warmUpNetwork}
      onFocus={warmUpNetwork}
      onSubmit={(e) => {
        e.preventDefault();
        if (canJoin) void useNetworkStore.getState().joinRoom(clean, name);
      }}
    >
      <h4>Join a friend's room</h4>
      <div className="online-join-row">
        <input placeholder="Your name" value={name} maxLength={24} onChange={(e) => setName(e.target.value)} />
        <input
          className="online-join-code"
          placeholder="CODE"
          aria-label="Room code"
          value={code}
          onChange={(e) => setCode(e.target.value.toUpperCase())}
          maxLength={6}
        />
        <button className="secondary" type="submit" disabled={!canJoin}>Join</button>
      </div>
      <p className="lobby-hint" style={{ margin: '8px 0 0' }}>To host, set up a game and choose “Host online”.</p>
    </form>
  );
}

export function OnlineStatusCard() {
  const connection = useNetworkStore((s) => s.connection);
  const roomCode = useNetworkStore((s) => s.roomCode);
  const role = useNetworkStore((s) => s.role);
  const errorMessage = useNetworkStore((s) => s.errorMessage);
  const joinFailure = useNetworkStore((s) => s.joinFailure);
  const net = useNetworkStore.getState;

  if (connection === 'error') {
    return (
      <div className="lobby-single">
        <section className="lobby-panel online-status">
          <h3>Couldn't {joinFailure ? `join room ${joinFailure.code}` : 'open the room'}</h3>
          <p>{errorMessage}</p>
          <div style={{ display: 'flex', gap: 8 }}>
            {joinFailure?.reason === 'unreachable' && <button onClick={() => net().retryJoin()}>Try again</button>}
            <button className="secondary" onClick={() => net().clearError()}>Back</button>
          </div>
        </section>
      </div>
    );
  }
  return (
    <div className="lobby-single">
      <section className="lobby-panel online-status">
        <h3>{role === 'host' && !roomCode ? 'Opening a room…' : `Joining room ${roomCode ?? ''}…`}</h3>
        <p className="lobby-hint">Connecting to the game server.</p>
        <button className="secondary" onClick={() => net().leaveRoom()}>Cancel</button>
      </section>
    </div>
  );
}
