/**
 * Tests for the upload shrinker.
 *
 * GitHub's browser upload form refuses files over 25 MB, and these pages are
 * static so there is no server that could accept one instead. docs/shrink.js
 * therefore has to make the file smaller before it is ever sent.
 *
 * The properties worth pinning are the ones that would fail silently: the
 * threshold must sit below the real cap rather than at it, and a clip whose
 * audio alone is still too big must be refused rather than truncated.
 *
 * Run: node tests/browser/shrink.test.mjs
 */
import { needsShrink, formatMB, maxSeconds, LIMIT_MB } from '../../docs/shrink.js';

let pass = 0, fail = 0;
const ok = (c, m) => { c ? pass++ : (fail++, console.log('FAIL:', m)); };

const CAP = 25 * 1024 * 1024;

// The advertised limit must be the real one.
ok(LIMIT_MB === 25, 'limit is stated as 25 MB');

// Threshold sits under the cap, not at it: a file that lands at exactly 25 MB
// is still refused once the multipart envelope is added.
ok(needsShrink({ size: CAP }), 'a file at exactly the cap is shrunk');
ok(needsShrink({ size: CAP + 1 }), 'a file over the cap is shrunk');
ok(!needsShrink({ size: 10 * 1024 * 1024 }), 'a 10 MB file is left alone');
ok(!needsShrink({ size: 0 }), 'an empty file is left alone');

// The safety margin should be real but not wasteful.
let margin = 0;
for (let mb = 20; mb <= 25; mb += 0.25) {
  if (needsShrink({ size: mb * 1024 * 1024 })) { margin = 25 - mb; break; }
}
ok(margin > 0 && margin <= 2,
   `safety margin is ${margin.toFixed(2)} MB (want >0 and <=2)`);

// Capacity must be honest: 16 kHz mono 16-bit is 32000 bytes/sec.
const secs = maxSeconds();
const impliedBytes = secs * TARGET_BYTES_PER_SEC();
function TARGET_BYTES_PER_SEC() { return 16000 * 2; }
ok(impliedBytes <= CAP, `capacity ${secs}s implies ${(impliedBytes/1048576).toFixed(1)} MB, within cap`);
ok(secs > 600, `capacity is ${Math.floor(secs/60)} min (want over 10)`);

ok(formatMB(1048576) === '1.0', 'formats 1 MiB as 1.0');
ok(formatMB(6279466) === '6.0', 'formats the Vikings clip as 6.0');

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
