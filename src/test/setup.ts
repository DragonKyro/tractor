// Vitest setup: the AI as a tiny random-weight network, so tests that play
// whole matches (engine, driver, stores) don't fetch the real policy file.
// It plays badly but legally. Tests of the real file read it from disk
// (src/ai/nn/nn.test.ts).

import { setNetwork } from '@/ai/nn/network';
import { TractorPolicyNet } from '@/ai/nn/runtime';
import { randomPolicyFile } from '@/ai/nn/testSupport';

setNetwork(new TractorPolicyNet(randomPolicyFile(4, { seat: [8], trunk: [16], cand: [8], joint: 8 })));
