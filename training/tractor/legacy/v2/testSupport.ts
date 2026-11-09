// Test helper: a random-weight v2 policy file (tractor/cand-mlp@1, encoder v2),
// small, so benchmark seats can be tested without the real benchmark file.

import { ACTION_SPACE_VERSION, CAND_DIM } from './decision';
import { ENCODER_VERSION, OBS_DIM } from './encode';
import { POLICY_FORMAT, type TractorPolicyFile } from './runtime';

export function randomLegacyPolicyFile(seed = 1): TractorPolicyFile {
  let s = seed >>> 0;
  const r = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const arch = { obsDim: OBS_DIM, candDim: CAND_DIM, state: [32], cand: [16], joint: 16, value: 8 };
  const tensors: TractorPolicyFile['tensors'] = {};
  const b64 = (a: Float32Array) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64');
  const linear = (name: string, nIn: number, nOut: number, bias = true) => {
    const scale = Math.sqrt(2 / nIn);
    tensors[`${name}.weight`] = { shape: [nOut, nIn], data: b64(new Float32Array(nIn * nOut).map(() => (r() * 2 - 1) * scale)) };
    if (bias) tensors[`${name}.bias`] = { shape: [nOut], data: b64(new Float32Array(nOut).map(() => (r() * 2 - 1) * 0.05)) };
  };
  linear('state.0', arch.obsDim, 32);
  linear('cand.0', arch.candDim, 16);
  linear('joint_state', 32, 16);
  linear('joint_cand', 16, 16, false);
  linear('score', 16, 1);
  linear('value.0', 32, 8);
  linear('value.2', 8, 1);
  return { format: POLICY_FORMAT, game: 'tractor', encoderVersion: ENCODER_VERSION, actionSpaceVersion: ACTION_SPACE_VERSION, arch, tensors };
}
