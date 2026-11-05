// The room code rides in the page URL (?room=ABCD) while a device is in a
// room, so reloading the page (phones do this to background tabs) or opening
// a shared invite link lands back in the same room.

const PARAM = 'room';

export function normalizeRoomCode(code: string): string {
  return code.trim().toUpperCase().replace(/[^A-Z0-9]/g, '');
}

/** The room code in this page's URL, if any. */
export function roomCodeFromUrl(): string | null {
  try {
    if (typeof window === 'undefined') return null;
    const code = new URLSearchParams(window.location.search).get(PARAM);
    const clean = code ? normalizeRoomCode(code) : '';
    return clean.length >= 4 ? clean : null;
  } catch {
    return null;
  }
}

/** Put `code` in the URL (or take it out with null) without reloading. Keeps other params like ?fresh. */
export function setRoomCodeInUrl(code: string | null): void {
  try {
    if (typeof window === 'undefined' || !window.history?.replaceState) return;
    const url = new URL(window.location.href);
    if (code) url.searchParams.set(PARAM, code);
    else url.searchParams.delete(PARAM);
    // URLSearchParams writes a bare flag as "fresh="; keep it bare.
    const href = url.toString().replace(/([?&]fresh)=(?=&|$)/, '$1');
    if (href !== window.location.href) window.history.replaceState(window.history.state, '', href);
  } catch {
    /* ignore */
  }
}

/** A link that opens this app straight into room `code`. */
export function roomLink(code: string): string {
  if (typeof window === 'undefined') return code;
  const url = new URL(window.location.origin + window.location.pathname);
  url.searchParams.set(PARAM, code);
  return url.toString();
}
