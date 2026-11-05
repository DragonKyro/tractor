// In-room chat. Messages live in the room (so a rejoining device sees them)
// and in networkStore.chat; system lines (joins, leaves, match start) are made
// on each device. Not part of GameState.

import { useEffect, useRef, useState } from 'react';
import { useNetworkStore } from '@/store/networkStore';

export function ChatPanel() {
  const chat = useNetworkStore((s) => s.chat);
  const sendChat = useNetworkStore((s) => s.sendChat);
  const [draft, setDraft] = useState('');
  const [open, setOpen] = useState(false);
  const [seen, setSeen] = useState(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const userCount = chat.filter((m) => m.kind === 'user').length;

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    if (open) setSeen(userCount);
  }, [chat.length, open, userCount]);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!draft.trim()) return;
    sendChat(draft);
    setDraft('');
  }

  const unread = Math.max(0, userCount - seen);

  return (
    <div className={`chat-panel ${open ? 'open' : 'closed'}`}>
      <button className="chat-toggle" onClick={() => setOpen(!open)} type="button" aria-label={open ? 'Close chat' : 'Open chat'}>
        {open ? '✕' : '💬'} {unread > 0 && !open ? `(${unread})` : ''}
      </button>
      {open && (
        <div className="chat-body">
          <div className="chat-log" ref={scrollRef}>
            {chat.length === 0 && <div className="chat-empty">No messages yet.</div>}
            {chat.map((m) => (
              <div key={m.id} className={m.kind === 'system' ? 'chat-row chat-system' : 'chat-row'}>
                {m.kind === 'system' ? m.text : <><strong>{m.name}:</strong> <span>{m.text}</span></>}
              </div>
            ))}
          </div>
          <form onSubmit={submit} className="chat-form">
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Say something…"
              maxLength={200}
            />
            <button type="submit" disabled={!draft.trim()}>Send</button>
          </form>
        </div>
      )}
    </div>
  );
}
