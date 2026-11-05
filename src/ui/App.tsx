// Top-level shell. Offline, three screens — home, lobby, in-game — chosen by
// local route state. While this device is in an online room the room
// decides instead: connecting / error card, the room's lobby, or its match.

import { useEffect, useRef, useState } from 'react';
import { isOnlineConfigured } from '@/net/firebaseConfig';
import type { MatchSetup } from '@/store/gameStore';
import { useNetworkStore } from '@/store/networkStore';
import { CardView } from './CardView';
import { GameHost } from './GameHost';
import { LocalLobby } from './LocalLobby';
import { OnlineLobby } from './OnlineLobby';
import { JoinRoomBox, OnlineStatusCard, useRejoinFromUrl } from './OnlineStatus';
import { RulesModal } from './lobbyParts';
import { RulesText } from './Rules';

type Screen = { kind: 'home' } | { kind: 'lobby'; initial?: MatchSetup } | { kind: 'playing' };

export default function App() {
  const [screen, setScreen] = useState<Screen>({ kind: 'home' });
  const [lobbyVisit, setLobbyVisit] = useState(0);
  const [rules, setRules] = useState(false);
  const connection = useNetworkStore((s) => s.connection);
  const urlCode = useRejoinFromUrl();
  const onlineReady = isOnlineConfigured();

  // Leaving a room lands back home.
  const wasOnline = useRef(false);
  useEffect(() => {
    if (connection !== 'disconnected') wasOnline.current = true;
    else if (wasOnline.current) {
      wasOnline.current = false;
      setScreen({ kind: 'home' });
    }
  }, [connection]);

  const online = connection !== 'disconnected';

  return (
    <div className={`app-shell${(online && connection === 'in-game') || (!online && screen.kind === 'playing') ? ' app-playing' : ''}`}>
      <header>
        <strong
          className="app-title"
          onClick={() => {
            if (!online && screen.kind !== 'playing') setScreen({ kind: 'home' });
          }}
        >
          <span className="app-logo">🚜</span> Tractor <span className="app-sub">升级</span>
        </strong>
      </header>
      <main>
        {online && (connection === 'connecting' || connection === 'error') && <OnlineStatusCard />}
        {online && connection === 'lobby' && <OnlineLobby />}
        {online && connection === 'in-game' && <GameHost onExit={() => {}} onReturnToLobby={() => {}} />}
        {!online && screen.kind === 'home' && (
          <div className="home">
            <div className="home-hero">
              <div className="home-fan">
                {['AH', 'AH', 'BJ', 'LJ', '2S'].map((c, i) => (
                  <span key={i} className="home-fan-card" style={{ transform: `rotate(${(i - 2) * 9}deg) translateY(${Math.abs(i - 2) * 6}px)` }}>
                    <CardView card={c} size="lg" />
                  </span>
                ))}
              </div>
              <h1>Tractor</h1>
              <p className="home-tag">升级 · Shengji · Finding Friends — the climbing trick-taking game, for any number of players and decks.</p>
              <div className="home-actions">
                <button className="home-play" onClick={() => setScreen({ kind: 'lobby' })}>Play — hot-seat or vs AI</button>
                <button className="secondary" onClick={() => setRules(true)}>📖 How to play</button>
              </div>
            </div>
            {onlineReady ? (
              <JoinRoomBox initialCode={urlCode} />
            ) : (
              <p className="lobby-hint home-offline">Online rooms switch on once a Firebase project is configured (src/net/firebaseConfig.ts).</p>
            )}
          </div>
        )}
        {!online && screen.kind === 'lobby' && (
          <LocalLobby
            key={lobbyVisit}
            initial={screen.initial}
            onBack={() => setScreen({ kind: 'home' })}
            onStart={() => setScreen({ kind: 'playing' })}
          />
        )}
        {!online && screen.kind === 'playing' && (
          <GameHost
            onExit={() => setScreen({ kind: 'home' })}
            onReturnToLobby={(initial) => {
              setLobbyVisit((n) => n + 1);
              setScreen({ kind: 'lobby', initial });
            }}
          />
        )}
      </main>
      {rules && <RulesModal title="Tractor" Rules={RulesText} onClose={() => setRules(false)} />}
    </div>
  );
}
