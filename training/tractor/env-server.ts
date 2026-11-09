// stdio bridge from TractorVecEnv to an out-of-process learner (Python).
// One JSON object per line in, one per line out, strictly request/response.
//
//   → {"cmd":"spec"}
//   ← {"ok":true,"dims":{global,seat,maxSeats,cand,maxCandidates,cardSlots,privGlobal,privSeat},
//      "layouts":{global,seat,cand},"kinds":{bid:0,…},
//      "encoderVersion":…,"actionSpaceVersion":…,"policyFormat":…}
//   → {"cmd":"reset","options":{numEnvs, seed, lineups, scenario, reward, quota,
//      privileged, pointsShaping, duplicate, maxSteps, matches, maxRounds}}
//   → {"cmd":"step","actions":[candidate index, …]}     (one per env)
//   ← {"ok":true,"n":…,"seatRows":…,"candRows":…,"seat":[…],"role":[…],"kind":[…],
//      "players":[…],"reward":[…],"done":[…],"episodes":[…],
//      "binary":[["glob",bytes],["seatCount",bytes],["seatSlot",bytes],["seats",bytes],
//                ["privGlob",bytes],["privSeats",bytes],["count",bytes],["cand",bytes]]}
//     followed by those raw bytes, in that order: f32 little-endian global blocks
//     (n × global), i16 seat blocks per row, i8 their slots, f32 seat blocks packed
//     in row order (seatRows × seat), f32 privileged features (n × privGlobal,
//     seatRows × privSeat; empty unless privileged), i16 candidate count per row,
//     f32 candidates packed in row order (candRows × cand)
//   → {"cmd":"configure","options":{lineups?, scenario?}}  ← {"ok":true}
//   → {"cmd":"policyLogits","path":"<policy.json>"}
//   ← {"ok":true,"logits":[[…], …]}   (the TS runtime on every env's pending row — parity checks)
//   → {"cmd":"close"}
//   ← {"ok":false,"error":"…"}                       (on any failure)
//
// Nothing else is written to stdout; diagnostics go to stderr.

import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { ACTION_SPACE_VERSION, CAND_DIM, CAND_LAYOUT, MAX_CANDIDATES } from '@/ai/nn/decision';
import { ENCODER_VERSION, GLOBAL_DIM, GLOBAL_LAYOUT, MAX_SEATS, SEAT_DIM, SEAT_LAYOUT } from '@/ai/nn/encode';
import { POLICY_FORMAT, TractorPolicyNet, type TractorPolicyFile } from '@/ai/nn/runtime';
import { CARD_SLOTS } from '@/ai/nn/slots';
import { PRIV_GLOBAL_DIM, PRIV_SEAT_DIM } from './privileged';
import { KIND_CODE, TractorVecEnv, type VecBatch, type VecEnvOptions } from './vec-env';

const bytes = (a: ArrayBufferView) => Buffer.from(a.buffer, a.byteOffset, a.byteLength);

class BatchReply {
  constructor(readonly b: VecBatch) {}

  write(): void {
    const b = this.b;
    const arrays: Array<[string, ArrayBufferView]> = [
      ['glob', b.glob], ['seatCount', b.seatCount], ['seatSlot', b.seatSlot], ['seats', b.seats],
      ['privGlob', b.privGlob], ['privSeats', b.privSeats], ['count', b.count], ['cand', b.cand],
    ];
    const header = {
      ok: true,
      n: b.seat.length,
      seatRows: b.seatRows,
      candRows: b.candRows,
      seat: Array.from(b.seat),
      role: Array.from(b.role),
      kind: Array.from(b.kind),
      players: Array.from(b.players),
      reward: Array.from(b.reward),
      done: Array.from(b.done),
      episodes: b.episodes,
      binary: arrays.map(([name, a]) => [name, a.byteLength]),
    };
    process.stdout.write(Buffer.concat([Buffer.from(JSON.stringify(header) + '\n'), ...arrays.map(([, a]) => bytes(a))]));
  }
}

function spec() {
  return {
    ok: true,
    dims: {
      global: GLOBAL_DIM, seat: SEAT_DIM, maxSeats: MAX_SEATS, cand: CAND_DIM, maxCandidates: MAX_CANDIDATES,
      cardSlots: CARD_SLOTS, privGlobal: PRIV_GLOBAL_DIM, privSeat: PRIV_SEAT_DIM,
    },
    layouts: { global: GLOBAL_LAYOUT.segments, seat: SEAT_LAYOUT.segments, cand: CAND_LAYOUT.segments },
    kinds: KIND_CODE,
    encoderVersion: ENCODER_VERSION,
    actionSpaceVersion: ACTION_SPACE_VERSION,
    policyFormat: POLICY_FORMAT,
  };
}

let env: TractorVecEnv | null = null;

interface Request {
  cmd?: string;
  options?: Partial<VecEnvOptions>;
  actions?: number[];
  path?: string;
}

function handle(msg: Request): object | BatchReply {
  switch (msg.cmd) {
    case 'spec':
      return spec();
    case 'reset':
      env = new TractorVecEnv(msg.options ?? {});
      return new BatchReply(env.reset());
    case 'step':
      if (!env) throw new Error('step before reset');
      if (!Array.isArray(msg.actions)) throw new Error('step needs "actions": number[]');
      return new BatchReply(env.step(msg.actions));
    case 'configure':
      if (!env) throw new Error('configure before reset');
      env.configure(msg.options ?? {});
      return { ok: true };
    case 'policyLogits': {
      if (!env) throw new Error('policyLogits before reset');
      if (!msg.path) throw new Error('policyLogits needs "path"');
      const net = new TractorPolicyNet(JSON.parse(readFileSync(msg.path, 'utf8')) as TractorPolicyFile);
      return { ok: true, logits: env.policyLogits(net) };
    }
    case 'close':
      process.exit(0);
    default:
      throw new Error(`unknown cmd ${JSON.stringify(msg.cmd)}`);
  }
}

createInterface({ input: process.stdin, crlfDelay: Infinity }).on('line', (line) => {
  if (!line.trim()) return;
  let reply: object | BatchReply;
  try {
    reply = handle(JSON.parse(line));
  } catch (e) {
    reply = { ok: false, error: (e as Error).stack ?? (e as Error).message };
  }
  if (reply instanceof BatchReply) reply.write();
  else process.stdout.write(JSON.stringify(reply) + '\n');
});
