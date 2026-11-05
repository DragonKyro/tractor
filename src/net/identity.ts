// Persistent player identity. The UUID lives in localStorage and is reused
// forever for this browser profile, so a player who reloads or loses signal
// rejoins a room in the same seat. The UUID, not the Firebase sign-in, is what
// owns a seat (an online human seat's id is its owner's UUID).
//
// For local testing with two windows that share localStorage (two tabs of one
// incognito session), add `?fresh` to the URL: the UUID (and the anonymous
// Firebase sign-in) then live in sessionStorage, one per tab.

const UUID_KEY = 'tractor.uuid';
const NAME_KEY = 'tractor.displayName';

export function freshTabMode(): boolean {
  try {
    if (typeof window === 'undefined') return false;
    return new URLSearchParams(window.location.search).has('fresh');
  } catch {
    return false;
  }
}

export function getOrCreateUuid(): string {
  try {
    const storage = freshTabMode() ? sessionStorage : localStorage;
    const existing = storage.getItem(UUID_KEY);
    if (existing) return existing;
    const next = typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : fallbackUuid();
    storage.setItem(UUID_KEY, next);
    return next;
  } catch {
    return fallbackUuid();
  }
}

export function getSavedDisplayName(): string {
  try {
    return localStorage.getItem(NAME_KEY) ?? '';
  } catch {
    return '';
  }
}

export function saveDisplayName(name: string): void {
  try {
    localStorage.setItem(NAME_KEY, name);
  } catch {
    /* ignore */
  }
}

function fallbackUuid(): string {
  const rand = () =>
    Math.floor(Math.random() * 0xffffffff)
      .toString(16)
      .padStart(8, '0');
  return `${rand()}-${rand()}-${rand()}-${rand()}`;
}

/** A short, friendly room code: 4 characters, A–Z and 2–9 without the look-alikes (I, O, 0, 1). */
export function generateRoomCode(): string {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let code = '';
  for (let i = 0; i < 4; i++) code += alphabet[Math.floor(Math.random() * alphabet.length)];
  return code;
}
