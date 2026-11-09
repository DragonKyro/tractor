// Headless AI-vs-AI matches between the shipped levels, with summary stats — a
// quick check of the engine and of how the levels compare.
// Run: npm run sim -- [players] [matches] [levels]
//   e.g. npm run sim -- 4 6 hard,easy   (seats alternate hard / easy)
// (For rigorous comparisons use training/tractor/evaluate.py: duplicate deals.)

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { aiSeats, simulateMatch } from '../src/ai/simulate';
import { NN_LEVELS, setNnLevel } from '../src/ai/nn/levels';
import { TractorPolicyNet, type TractorPolicyFile } from '../src/ai/nn/runtime';

const [players = '4', matches = '6', levelsArg = 'hard,easy'] = process.argv.slice(2);
const n = Number(players);
const levels = levelsArg.split(',');
for (const id of levels) {
  const level = NN_LEVELS.find((l) => l.id === id);
  if (!level) throw new Error(`no AI level "${id}" (have ${NN_LEVELS.map((l) => l.id).join(', ')})`);
  const file = JSON.parse(readFileSync(join('src', 'ai', 'nn', 'policies', `${id}.json`), 'utf8')) as TractorPolicyFile;
  setNnLevel(level, new TractorPolicyNet(file));
}

const wins = new Map<string, number>();
let rounds = 0;
let steps = 0;
let unfinished = 0;
const started = Date.now();
for (let m = 0; m < Number(matches); m++) {
  // Rotate the lineup so no level always sits in the same seats.
  const rotated = levels.map((_, i) => levels[(i + m) % levels.length]!);
  const seats = aiSeats(n, rotated);
  const res = simulateMatch({ seats, seed: 1000 + m, maxRounds: 80 });
  rounds += res.rounds;
  steps += res.steps;
  if (!res.finished) unfinished++;
  for (const w of new Set((res.state.winners ?? []).map((i) => seats[i]!.aiLevel ?? 'normal'))) {
    wins.set(w, (wins.get(w) ?? 0) + 1);
  }
}
const secs = (Date.now() - started) / 1000;
console.log(`${matches} matches · ${n} players · levels ${levels.join('/')}`);
console.log(`avg ${(rounds / Number(matches)).toFixed(1)} rounds, ${(steps / Number(matches)).toFixed(0)} actions per match · ${secs.toFixed(1)}s`);
for (const [level, w] of wins) console.log(`  ${level}: won ${w}/${matches}`);
if (unfinished) console.log(`  ${unfinished} hit the round cap`);
