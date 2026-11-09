// The AI difficulty levels: three checkpoints of one training run
// (training/tractor/train.py, run v3), exported as float16 policy files in
// ./policies and fetched on first use (hashed assets).
//
//   easy    update 100   straight after distillation: sound, but still rough
//   normal  update 500   past the quick early gains
//   hard    update 2000  where the run's gains levelled off
//
// A network plays every table (2-12 players, both modes); how well it plays
// a table depends on what the curriculum covered (training/tractor/scenario.ts).

import easyUrl from './policies/easy.json?url';
import normalUrl from './policies/normal.json?url';
import hardUrl from './policies/hard.json?url';
import { TractorPolicyNet, type TractorPolicyFile } from './runtime';

export interface NnLevel {
  id: string;
  label: string;
  url: string;
}

export const NN_LEVELS: NnLevel[] = [
  { id: 'easy', label: 'Easy', url: easyUrl },
  { id: 'normal', label: 'Normal', url: normalUrl },
  { id: 'hard', label: 'Hard', url: hardUrl },
];

const loading = new Map<string, Promise<TractorPolicyNet>>();
const loaded = new Map<string, TractorPolicyNet>();

export function isNnLevel(id: string | undefined): boolean {
  return !!id && NN_LEVELS.some((l) => l.id === id);
}

/** Fetch and build a level's network (once per page; a failed fetch can be retried). */
export function loadNnLevel(id: string): Promise<TractorPolicyNet> {
  const level = NN_LEVELS.find((l) => l.id === id);
  if (!level) return Promise.reject(new Error(`no AI level ${id}`));
  let p = loading.get(id);
  if (!p) {
    p = fetch(level.url)
      .then((res) => {
        if (!res.ok) throw new Error(`AI ${id}: HTTP ${res.status}`);
        return res.json() as Promise<TractorPolicyFile>;
      })
      .then((file) => {
        const net = new TractorPolicyNet(file);
        loaded.set(id, net);
        return net;
      });
    p.catch(() => loading.delete(id));
    loading.set(id, p);
  }
  return p;
}

export function loadedNnLevel(id: string): TractorPolicyNet | null {
  return loaded.get(id) ?? null;
}

/** Install a network built elsewhere (tests, tools) under a level id. */
export function setNnLevel(level: NnLevel, net: TractorPolicyNet): void {
  if (!NN_LEVELS.some((l) => l.id === level.id)) NN_LEVELS.push(level);
  loaded.set(level.id, net);
  loading.set(level.id, Promise.resolve(net));
}
