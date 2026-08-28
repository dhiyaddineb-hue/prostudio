/**
 * Tests for the browser-to-GitHub uploader.
 *
 * These pages are static, so committing straight from the browser is the only
 * way a large file reaches the repository without a server of our own. The
 * network calls are exercised end to end separately; what is pinned here is
 * the arithmetic and the naming, because those are what corrupt a file
 * silently rather than failing loudly.
 *
 * Run: node tests/browser/github-upload.test.mjs
 */
import {
  partCount, partName, safeName, formatMB, BRANCH, OWNER, REPO,
} from '../../docs/github-upload.js';

let pass = 0, fail = 0;
const ok = (c, m) => { c ? pass++ : (fail++, console.log('FAIL:', m)); };
const MB = 1048576;

// Parts must stay under the measured blob ceiling: 40 MB was accepted by the
// API, 45 MB refused with "your input was too large to process".
ok(partCount(18 * MB) === 1, 'exactly one part-size stays one part');
ok(partCount(18 * MB + 1) === 2, 'one byte over rolls to a second part');
ok(partCount(0) === 1, 'an empty file is one part, never zero');
ok(partCount(74.5 * MB) === 5, '74.5 MB becomes 5 parts');

for (const mb of [1, 30, 74.5, 200, 500]) {
  const size = mb * MB;
  const largest = Math.ceil(size / partCount(size));
  ok(largest <= 40 * MB,
     `${mb} MB file: largest part ${(largest / MB).toFixed(1)} MB is under the 40 MB ceiling`);
}

// The joiner reads a sorted listing, so unpadded numbers would place part 10
// between 1 and 2 and reassemble the file in the wrong order.
const many = Array.from({ length: 12 }, (_, i) => partName('v.mp4', i + 1, 12));
ok(JSON.stringify([...many].sort()) === JSON.stringify(many),
   'part names sort into numeric order past ten');
ok(partName('v.mp4', 1, 5) === 'v.mp4.part01of05', 'name states position and total');

// scripts/join_parts.py matches ^(.+)\.part(\d+)of(\d+)$ — the two must agree,
// or uploads arrive and are never recognised as parts.
const JOINER = /^(?<name>.+)\.part(?<index>\d+)of(?<total>\d+)$/;
const m = JOINER.exec(partName('clip.mp4', 3, 12));
ok(m && m.groups.name === 'clip.mp4' && +m.groups.index === 3 && +m.groups.total === 12,
   'part names parse with the joiner regex');

// A filename crosses the network, so it cannot be trusted to stay in inbox/.
ok(safeName('clip.mp4') === 'clip.mp4', 'an ordinary name passes through');
ok(safeName('../escape.mp4') === 'escape.mp4', 'posix traversal is stripped');
ok(safeName('..\\windows.mp4') === 'windows.mp4', 'backslash traversal is stripped');
ok(safeName('/etc/passwd') === 'passwd', 'absolute paths are stripped');
for (const bad of ['', '.', '..', '...', '.hidden', '   ']) {
  let threw = false;
  try { safeName(bad); } catch { threw = true; }
  ok(threw, `refuses ${JSON.stringify(bad)}`);
}

ok(BRANCH === 'arena/01a03969-prostudio', 'commits land on the session branch');
ok(OWNER === 'dhiyaddineb-hue' && REPO === 'prostudio', 'points at the right repo');
ok(formatMB(74.5 * MB) === '74.5', 'formats sizes for display');

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
