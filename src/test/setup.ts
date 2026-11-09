// Vitest setup: the three AI levels, as tiny random-weight networks, so tests
// that play whole matches (engine, driver, stores) don't fetch the real policy
// files. They play badly but legally. Tests of the real files read them from
// disk (src/ai/nn/nn.test.ts).

import { NN_LEVELS, setNnLevel } from '@/ai/nn/levels';
import { TractorPolicyNet } from '@/ai/nn/runtime';
import { randomPolicyFile } from '@/ai/nn/testSupport';

NN_LEVELS.forEach((level, i) => {
  setNnLevel(level, new TractorPolicyNet(randomPolicyFile(11 + i, { seat: [8], trunk: [16], cand: [8], joint: 8 })));
});
