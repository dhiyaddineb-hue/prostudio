/**
 * Tests for the upload shrinker.
 *
 * GitHub's browser upload form refuses files over 25 MB and these pages are
 * static, so there is no server that could take one instead: docs/shrink.js
 * has to make the file smaller before it is sent.
 *
 * The regression these guard against is the one that already shipped. Writing
 * WAV capped uploads at 13 minutes, so a 23 minute clip was bounced back with
 * instructions to trim it by hand -- the tool declining to do its job. The
 * capacity assertions below fail if anything pushes the ceiling back down.
 *
 * Run: node tests/browser/shrink.test.mjs
 */
import {
  needsShrink, formatMB, maxSeconds, costPerMinute, LIMIT_MB,
} from '../../docs/shrink.js';

let pass = 0, fail = 0;
const ok = (c, m) => { c ? pass++ : (fail++, console.log('FAIL:', m)); };

const CAP = 25 * 1024 * 1024;

ok(LIMIT_MB === 25, 'the advertised limit is the real one');

// Threshold sits under the cap, not at it: a file landing at exactly 25 MB is
// still refused once the multipart envelope is added.
ok(needsShrink({ size: CAP }), 'a file at exactly the cap is shrunk');
ok(needsShrink({ size: CAP + 1 }), 'a file over the cap is shrunk');
ok(!needsShrink({ size: 10 * 1024 * 1024 }), 'a 10 MB file is left alone');
ok(!needsShrink({ size: 0 }), 'an empty file is left alone');

let margin = 0;
for (let mb = 20; mb <= 25; mb += 0.25) {
  if (needsShrink({ size: mb * 1024 * 1024 })) { margin = 25 - mb; break; }
}
ok(margin > 0 && margin <= 2, `safety margin is ${margin.toFixed(2)} MB (want >0, <=2)`);

// Capacity: the whole point of the MP3 rewrite.
const secs = maxSeconds();
const mins = secs / 60;
ok(mins >= 45, `capacity is ${mins.toFixed(0)} min (want >=45; WAV gave 13)`);
ok(secs > 23 * 60 + 7,
   `the 23:07 clip that was rejected now fits (capacity ${mins.toFixed(0)} min)`);

// The quoted cost must match the real capacity, or the page lies.
const impliedBytes = costPerMinute() * 1048576 * mins;
ok(Math.abs(impliedBytes - secs * (64000 / 8)) < 1024,
   'quoted MB/minute agrees with actual capacity');
ok(costPerMinute() < 0.5, `cost is ${costPerMinute().toFixed(2)} MB/min`);

ok(formatMB(1048576) === '1.0', 'formats 1 MiB as 1.0');
ok(formatMB(6279466) === '6.0', 'formats the Vikings clip as 6.0');

console.log(`\ncapacity ${mins.toFixed(0)} min @ ${costPerMinute().toFixed(2)} MB/min`);
console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
