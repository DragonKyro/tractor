// Merge this app's Realtime Database rules into the live rules of the shared
// database, so deploying them leaves the other apps' rules alone.
//
// Several web apps share one Realtime Database (project github-games-e7c7d),
// and a database has a single rules set: deploying a rules file replaces
// every app's rules. So each app keeps its data under one top-level key (this
// one: "tractor", DB_ROOT in src/net/firebaseRoom.ts), its database.rules.json
// defines only that key, and this script:
//
//   1. reads the live rules (`firebase database:get /.settings/rules`),
//   2. replaces the keys database.rules.json defines (every key under "rules"
//      that doesn't start with "."), keeping every other key as it is live,
//   3. writes the result to .firebase/database.rules.deploy.json, the file
//      firebase.json points the deploy at.
//
// firebase.json runs it as the database predeploy hook, so a plain
// `firebase deploy --only database` (or `npm run deploy:rules`) is safe. If the
// live rules can't be read it fails, and the deploy stops. Run on its own
// (`node scripts/merge-db-rules.mjs`) it writes the merged file and deploys
// nothing.
//
// Other apps on the database can copy this script as is: it reads the keys
// they own from their own database.rules.json.

import { execSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = process.env.PROJECT_DIR ?? resolve(dirname(fileURLToPath(import.meta.url)), '..');
const LOCAL = join(root, 'database.rules.json');
const OUT = join(root, '.firebase', 'database.rules.deploy.json');

function fail(message) {
  console.error(`merge-db-rules: ${message}`);
  process.exit(1);
}

/** Parse rules text: JSON plus the // and /* *\/ comments the rules language allows. */
function parseRules(text, what) {
  let json = '';
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      let j = i + 1;
      while (j < text.length && text[j] !== '"') j += text[j] === '\\' ? 2 : 1;
      json += text.slice(i, j + 1);
      i = j;
    } else if (c === '/' && text[i + 1] === '/') {
      while (i < text.length && text[i] !== '\n') i++;
      json += '\n';
    } else if (c === '/' && text[i + 1] === '*') {
      const end = text.indexOf('*/', i + 2);
      i = end < 0 ? text.length : end + 1;
    } else {
      json += c;
    }
  }
  let parsed;
  try {
    parsed = JSON.parse(json);
  } catch (e) {
    fail(`${what} isn't valid JSON (${e.message}).`);
  }
  if (!parsed || typeof parsed.rules !== 'object' || parsed.rules === null) fail(`${what} has no "rules" object.`);
  return parsed.rules;
}

function projectId() {
  if (process.env.GCLOUD_PROJECT) return process.env.GCLOUD_PROJECT;
  try {
    const id = JSON.parse(readFileSync(join(root, '.firebaserc'), 'utf8')).projects?.default;
    if (id) return id;
  } catch {
    // fall through
  }
  fail('no Firebase project (set GCLOUD_PROJECT or a default in .firebaserc).');
}

function liveRules(project) {
  const dir = mkdtempSync(join(tmpdir(), 'db-rules-'));
  const file = join(dir, 'live.json');
  try {
    execSync(`npx -y firebase-tools@latest database:get /.settings/rules --project ${project} --output "${file}"`, {
      cwd: root,
      stdio: ['ignore', 'ignore', 'pipe'],
      // Git Bash would otherwise rewrite the /.settings path.
      env: { ...process.env, MSYS_NO_PATHCONV: '1' },
    });
    return parseRules(readFileSync(file, 'utf8'), `The live rules of ${project}`);
  } catch (e) {
    const stderr = e.stderr ? String(e.stderr).trim().split('\n').slice(-5).join('\n') : e.message;
    fail(`couldn't read the live rules of ${project}, so nothing was deployed.\n${stderr}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const local = parseRules(readFileSync(LOCAL, 'utf8'), 'database.rules.json');
const owned = Object.keys(local).filter((k) => !k.startsWith('.'));
if (owned.length === 0) fail('database.rules.json defines no top-level key of its own.');
for (const k of Object.keys(local).filter((k) => k.startsWith('.'))) {
  if (local[k] !== false) fail(`database.rules.json sets "${k}" at the database root; root rules are shared by every app, keep them false.`);
}

const project = projectId();
const live = liveRules(project);
for (const k of ['.read', '.write']) {
  if (live[k] !== undefined && live[k] !== false) {
    fail(`the live rules of ${project} grant "${k}" at the database root (${JSON.stringify(live[k])}), which opens every app's data. Set it to false in the Firebase console first.`);
  }
}

// Live order kept; this app's keys replaced in place (or added at the end).
const merged = { ...live, '.read': false, '.write': false };
for (const k of owned) merged[k] = local[k];

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, `${JSON.stringify({ rules: merged }, null, 2)}\n`);

const others = Object.keys(live).filter((k) => !k.startsWith('.') && !owned.includes(k));
console.log(
  `merge-db-rules: ${project}: ${owned.join(', ')} from database.rules.json; ` +
    (others.length ? `kept as live: ${others.join(', ')}.` : 'no other app has rules yet.'),
);
