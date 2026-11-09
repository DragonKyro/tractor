// FROZEN COPY of src/ai/nn (encoder v2, action space v1, tractor/cand-mlp@1) for benchmarking
// v2 policies (training/benchmarks/) against newer networks. Do not edit: a v2
// policy only plays as trained if this code stays exactly as it was.

// Tractor learned agent — the policy network in plain TypeScript (float32,
// no ML library). Runs a `tractor/cand-mlp@1` file exported by
// training/tractor/model.py; the same layers in the same order.
//
//   state tower   obs (OBS_DIM) → Linear+ReLU per width in arch.state → h
//   cand tower    each candidate (CAND_DIM) → Linear+ReLU per width in arch.cand → c_k
//   scorer        logit_k = w · ReLU(J_s·h + J_c·c_k + b) + b'      (one per candidate)
//   value         V = Linear(ReLU(Linear(h)))                        (training only)
//
// Weights are PyTorch's nn.Linear layout ([out][in] row-major), base64
// little-endian, keyed by state_dict name. Each tensor's `dtype` is 'f16'
// (what the trainer exports: half the download) or 'f32' (the default when
// absent); either way the network computes in float32.

import { ACTION_SPACE_VERSION, CAND_DIM } from './decision';
import { ENCODER_VERSION, OBS_DIM } from './encode';

export const POLICY_FORMAT = 'tractor/cand-mlp@1';

export interface TractorArch {
  obsDim: number;
  candDim: number;
  state: number[];
  cand: number[];
  joint: number;
  value: number;
}

export const DEFAULT_ARCH: Omit<TractorArch, 'obsDim' | 'candDim'> = { state: [512, 512, 256], cand: [128, 128], joint: 256, value: 128 };

interface TensorJson {
  shape: number[];
  data: string;
  dtype?: 'f16' | 'f32';
}

export interface TractorPolicyFile {
  format: string;
  game: string;
  encoderVersion: number;
  actionSpaceVersion: number;
  arch: TractorArch;
  tensors: Record<string, TensorJson>;
  meta?: Record<string, unknown>;
}

function decodeBytes(b64: string): Uint8Array {
  const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/** IEEE 754 half to float (exact: every half is a float). */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24; // zero / subnormal
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/** A tensor's values as float32 (little-endian f16 or f32 in the file). */
export function decodeTensor(x: TensorJson): Float32Array {
  const bytes = decodeBytes(x.data);
  const size = x.shape.reduce((a, b) => a * b, 1);
  const width = x.dtype === 'f16' ? 2 : 4;
  if (bytes.length !== size * width) throw new Error(`tensor ${JSON.stringify(x.shape)} needs ${size * width} bytes, got ${bytes.length}`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(size);
  if (width === 2) for (let i = 0; i < size; i++) out[i] = halfToFloat(view.getUint16(2 * i, true));
  else for (let i = 0; i < size; i++) out[i] = view.getFloat32(4 * i, true);
  return out;
}

class Linear {
  constructor(readonly w: Float32Array, readonly b: Float32Array | null, readonly nIn: number, readonly nOut: number) {}

  apply(x: Float32Array, relu: boolean): Float32Array {
    const { w, b, nIn, nOut } = this;
    if (x.length !== nIn) throw new Error(`layer expects ${nIn} inputs, got ${x.length}`);
    const y = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = b ? b[o]! : 0;
      const wo = o * nIn;
      for (let i = 0; i < nIn; i++) s += w[wo + i]! * x[i]!;
      y[o] = relu && s < 0 ? 0 : s;
    }
    return y;
  }
}

export class TractorPolicyNet {
  readonly arch: TractorArch;
  private readonly stateLayers: Linear[];
  private readonly candLayers: Linear[];
  private readonly jointState: Linear;
  private readonly jointCand: Linear;
  private readonly score: Linear;
  private readonly value1: Linear;
  private readonly value2: Linear;

  constructor(readonly file: TractorPolicyFile) {
    if (file.format !== POLICY_FORMAT || file.game !== 'tractor') throw new Error(`not a tractor policy (${file.format}, ${file.game})`);
    if (file.encoderVersion !== ENCODER_VERSION || file.actionSpaceVersion !== ACTION_SPACE_VERSION) {
      throw new Error(`policy trained on encoder v${file.encoderVersion} / actions v${file.actionSpaceVersion}; this build is v${ENCODER_VERSION} / v${ACTION_SPACE_VERSION}`);
    }
    if (file.arch.obsDim !== OBS_DIM || file.arch.candDim !== CAND_DIM) {
      throw new Error(`policy expects ${file.arch.obsDim} / ${file.arch.candDim} inputs; the encoder makes ${OBS_DIM} / ${CAND_DIM}`);
    }
    this.arch = file.arch;
    const t = (name: string) => {
      const x = file.tensors[name];
      if (!x) throw new Error(`policy file has no tensor ${name}`);
      return { data: decodeTensor(x), shape: x.shape };
    };
    const linear = (name: string, bias = true) => {
      const w = t(`${name}.weight`);
      const [nOut, nIn] = w.shape as [number, number];
      return new Linear(w.data, bias ? t(`${name}.bias`).data : null, nIn, nOut);
    };
    this.stateLayers = this.arch.state.map((_, i) => linear(`state.${2 * i}`));
    this.candLayers = this.arch.cand.map((_, i) => linear(`cand.${2 * i}`));
    this.jointState = linear('joint_state');
    this.jointCand = linear('joint_cand', false);
    this.score = linear('score');
    this.value1 = linear('value.0');
    this.value2 = linear('value.2');
  }

  embed(obs: Float32Array): Float32Array {
    let h = obs;
    for (const l of this.stateLayers) h = l.apply(h, true);
    return h;
  }

  /** One logit per candidate. */
  logits(obs: Float32Array, candidates: readonly Float32Array[]): Float32Array {
    const h = this.embed(obs);
    const js = this.jointState.apply(h, false);
    const out = new Float32Array(candidates.length);
    candidates.forEach((cf, k) => {
      let c = cf;
      for (const l of this.candLayers) c = l.apply(c, true);
      const jc = this.jointCand.apply(c, false);
      for (let i = 0; i < jc.length; i++) jc[i] = Math.max(0, jc[i]! + js[i]!);
      out[k] = this.score.apply(jc, false)[0]!;
    });
    return out;
  }

  value(obs: Float32Array): number {
    return this.value2.apply(this.value1.apply(this.embed(obs), true), false)[0]!;
  }
}
