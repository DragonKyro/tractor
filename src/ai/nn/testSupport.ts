// Test helpers: a random-weight policy file in the exported format, so the
// runtime and agent can be exercised without a trained network.

import { CAND_DIM, ACTION_SPACE_VERSION } from './decision';
import { ENCODER_VERSION, GLOBAL_DIM, MAX_SEATS, SEAT_DIM } from './encode';
import { DEFAULT_ARCH, POLICY_FORMAT, type TractorArch, type TractorPolicyFile } from './runtime';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function b64(a: Float32Array): string {
  return Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64');
}

/** float to IEEE half, round to nearest even (as numpy's astype(float16)); tiny values flush to zero (enough for tests). */
export function floatToHalf(x: number): number {
  const u = new Uint32Array(new Float32Array([x]).buffer)[0]!;
  const sign = (u >>> 16) & 0x8000;
  const exp = ((u >>> 23) & 0xff) - 127 + 15;
  const mant = u & 0x7fffff;
  if (exp <= 0) return sign;
  if (exp >= 0x1f) return sign | 0x7c00;
  let h = sign | (exp << 10) | (mant >>> 13);
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && h & 1)) h++;
  return h;
}

/** The same policy file with every tensor stored as float16. */
export function toHalfPolicyFile(file: TractorPolicyFile): TractorPolicyFile {
  const tensors: TractorPolicyFile['tensors'] = {};
  for (const [k, t] of Object.entries(file.tensors)) {
    const raw = Buffer.from(t.data, 'base64');
    const f = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    const h = new Uint16Array(f.length);
    for (let i = 0; i < f.length; i++) h[i] = floatToHalf(f[i]!);
    tensors[k] = { shape: t.shape, dtype: 'f16', data: Buffer.from(h.buffer).toString('base64') };
  }
  return { ...file, tensors };
}

export function randomPolicyFile(seed = 1, arch: Partial<TractorArch> = {}): TractorPolicyFile {
  const a: TractorArch = { globalDim: GLOBAL_DIM, seatDim: SEAT_DIM, maxSeats: MAX_SEATS, candDim: CAND_DIM, ...DEFAULT_ARCH, ...arch };
  const r = rng(seed);
  const tensors: TractorPolicyFile['tensors'] = {};
  const linear = (name: string, nIn: number, nOut: number, bias = true) => {
    const scale = Math.sqrt(2 / nIn);
    const w = new Float32Array(nIn * nOut).map(() => (r() * 2 - 1) * scale);
    tensors[`${name}.weight`] = { shape: [nOut, nIn], data: b64(w) };
    if (bias) tensors[`${name}.bias`] = { shape: [nOut], data: b64(new Float32Array(nOut).map(() => (r() * 2 - 1) * 0.05)) };
  };
  const tower = (prefix: string, nIn: number, widths: number[]) => {
    let w = nIn;
    widths.forEach((h, i) => {
      linear(`${prefix}.${2 * i}`, w, h);
      w = h;
    });
    return w;
  };
  const e = tower('seat', a.seatDim, a.seat);
  const hs = tower('trunk', a.globalDim + a.maxSeats * e, a.trunk);
  const hc = tower('cand', a.candDim, a.cand);
  linear('joint_state', hs, a.joint);
  linear('joint_cand', hc, a.joint, false);
  linear('score', a.joint, 1);
  return { format: POLICY_FORMAT, game: 'tractor', encoderVersion: ENCODER_VERSION, actionSpaceVersion: ACTION_SPACE_VERSION, arch: a, tensors };
}
