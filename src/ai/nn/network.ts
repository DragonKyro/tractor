// The AI's network: one trained policy (training/tractor/train.py, run v4,
// update 2000), a float16 file in ./policy.json fetched on first use (a hashed
// asset). It plays every table (2-12 players, both modes); how well it plays a
// table depends on what the curriculum covered (training/tractor/scenario.ts).

import policyUrl from './policy.json?url';
import { TractorPolicyNet, type TractorPolicyFile } from './runtime';

let loading: Promise<TractorPolicyNet> | null = null;
let loaded: TractorPolicyNet | null = null;

/** Fetch and build the network (once per page; a failed fetch can be retried). */
export function loadNetwork(): Promise<TractorPolicyNet> {
  if (!loading) {
    const p = fetch(policyUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`AI: HTTP ${res.status}`);
        return res.json() as Promise<TractorPolicyFile>;
      })
      .then((file) => (loaded = new TractorPolicyNet(file)));
    p.catch(() => {
      if (loading === p) loading = null;
    });
    loading = p;
  }
  return loading;
}

/** The network if it has loaded, else null. */
export function loadedNetwork(): TractorPolicyNet | null {
  return loaded;
}

/** Use a network built elsewhere (tests, tools) instead of fetching the shipped file. */
export function setNetwork(net: TractorPolicyNet): void {
  loaded = net;
  loading = Promise.resolve(net);
}
