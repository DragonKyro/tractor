// Headless matches of the shipped AI against itself, through the real driver
// and reducer, with summary stats: a quick check of the engine, the network
// file and its speed.
// Run: npm run sim -- [players] [matches]    e.g. npm run sim -- 6 4
// (To compare networks use training/tractor/evaluate.py: duplicate deals.)

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { aiSeats, simulateMatch } from '../src/ai/simulate';
import { setNetwork } from '../src/ai/nn/network';
import { TractorPolicyNet, type TractorPolicyFile } from '../src/ai/nn/runtime';

const [players = '4', matches = '6'] = process.argv.slice(2);
const n = Number(players);
const file = JSON.parse(readFileSync(join('src', 'ai', 'nn', 'policy.json'), 'utf8')) as TractorPolicyFile;
setNetwork(new TractorPolicyNet(file));

let rounds = 0;
let steps = 0;
let unfinished = 0;
let landlordWins = 0;
const started = Date.now();
for (let m = 0; m < Number(matches); m++) {
  const res = simulateMatch({ seats: aiSeats(n), seed: 1000 + m, maxRounds: 80 });
  rounds += res.rounds;
  steps += res.steps;
  landlordWins += res.state.history.filter((r) => r.landlordWon).length;
  if (!res.finished) unfinished++;
}
const secs = (Date.now() - started) / 1000;
console.log(`${matches} matches · ${n} players · update ${String(file.meta?.update ?? '?')}`);
console.log(`avg ${(rounds / Number(matches)).toFixed(1)} rounds, ${(steps / Number(matches)).toFixed(0)} actions per match · ${secs.toFixed(1)}s`);
console.log(`  the landlord's side kept the seat in ${((100 * landlordWins) / Math.max(1, rounds)).toFixed(0)}% of rounds`);
if (unfinished) console.log(`  ${unfinished} hit the round cap`);
