// Small synthesized sounds (Web Audio, no files): card snaps, trick wins,
// your turn, bids, friends revealed, the round / game ending. Browsers only
// allow audio after a user gesture, so the context is created on the first
// pointer press.

export type Sound = 'card' | 'fail' | 'trick' | 'turn' | 'bid' | 'friend' | 'round' | 'win';

let ctx: AudioContext | null = null;

function audio(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  if (!ctx) {
    const C = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!C) return null;
    try {
      ctx = new C();
    } catch {
      return null;
    }
  }
  if (ctx.state === 'suspended') void ctx.resume().catch(() => {});
  return ctx;
}

if (typeof window !== 'undefined') {
  window.addEventListener('pointerdown', () => audio(), { once: true, capture: true });
}

function tone(a: AudioContext, freq: number, start: number, dur: number, type: OscillatorType = 'sine', gain = 0.07): void {
  const osc = a.createOscillator();
  const g = a.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, start);
  g.gain.setValueAtTime(0, start);
  g.gain.linearRampToValueAtTime(gain, start + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  osc.connect(g).connect(a.destination);
  osc.start(start);
  osc.stop(start + dur + 0.02);
}

function snap(a: AudioContext, start: number, gain = 0.22): void {
  const len = Math.floor(a.sampleRate * 0.05);
  const buf = a.createBuffer(1, len, a.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len) ** 3;
  const src = a.createBufferSource();
  src.buffer = buf;
  const filter = a.createBiquadFilter();
  filter.type = 'bandpass';
  filter.frequency.value = 2400;
  filter.Q.value = 0.8;
  const g = a.createGain();
  g.gain.value = gain;
  src.connect(filter).connect(g).connect(a.destination);
  src.start(start);
}

export function playSound(kind: Sound): void {
  const a = audio();
  if (!a || a.state !== 'running') return;
  const t = a.currentTime + 0.01;
  switch (kind) {
    case 'card':
      snap(a, t);
      break;
    case 'fail':
      snap(a, t);
      tone(a, 220, t + 0.03, 0.22, 'sawtooth', 0.04);
      break;
    case 'trick':
      tone(a, 660, t, 0.12, 'triangle', 0.05);
      tone(a, 880, t + 0.09, 0.16, 'triangle', 0.05);
      break;
    case 'turn':
      tone(a, 988, t, 0.18, 'sine', 0.06);
      break;
    case 'bid':
      tone(a, 523, t, 0.08, 'square', 0.03);
      tone(a, 784, t + 0.07, 0.1, 'square', 0.03);
      break;
    case 'friend':
      [523, 659, 784].forEach((f, i) => tone(a, f, t + i * 0.08, 0.18, 'triangle', 0.06));
      break;
    case 'round':
      tone(a, 392, t, 0.25, 'triangle', 0.06);
      tone(a, 523, t + 0.18, 0.35, 'triangle', 0.06);
      break;
    case 'win':
      [523, 659, 784, 1047].forEach((f, i) => tone(a, f, t + i * 0.12, 0.3, 'triangle', 0.07));
      break;
  }
}
