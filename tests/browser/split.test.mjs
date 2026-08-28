/**
 * Tests for the browser file splitter.
 *
 * GitHub's upload form refuses files over 25 MB and these pages are static, so
 * a large video has to be cut up client-side and rejoined by the pipeline.
 *
 * The property that matters is that splitting is lossless. The earlier answer
 * -- extract the audio and drop the picture -- was lossy in a way that only
 * showed up when the finished dub arrived with no video. A byte split has no
 * such trap, and these tests pin the invariants that keep it that way: parts
 * fit under the cap, cover the file exactly, sort correctly, and rejoin to the
 * identical bytes.
 *
 * Run: node tests/browser/split.test.mjs
 */
import { webcrypto } from 'crypto';
if (!globalThis.crypto) globalThis.crypto = webcrypto;

import {
  splitFile, needsSplit, partCount, partName, formatMB, PART_MB, LIMIT_MB,
} from '../../docs/split.js';

let pass = 0, fail = 0;
const ok = (c, m) => { c ? pass++ : (fail++, console.log('FAIL:', m)); };
const MB = 1048576;

ok(LIMIT_MB === 25, 'the advertised GitHub limit is the real one');
ok(PART_MB < LIMIT_MB, `part size ${PART_MB} MB is under the ${LIMIT_MB} MB cap`);

// Threshold: a file that fits should not be cut up needlessly.
ok(!needsSplit({ size: 10 * MB }), 'a 10 MB file is left whole');
ok(!needsSplit({ size: 0 }), 'an empty file is left whole');
ok(needsSplit({ size: 25 * MB }), 'a file at the cap is split');
ok(needsSplit({ size: 100 * MB }), 'a 100 MB file is split');

// Counting.
ok(partCount(0) === 1, 'an empty file is one part, not zero');
ok(partCount(1) === 1, 'a tiny file is one part');
ok(partCount(20 * MB) === 1, 'exactly one part-size is one part');
ok(partCount(20 * MB + 1) === 2, 'one byte over rolls to two parts');
ok(partCount(74.5 * MB) === 4, '74.5 MB becomes 4 parts');

// Names must sort in order: a plain alphabetical listing is what the joiner
// relies on, and unpadded numbers put part 10 between 1 and 2.
const many = Array.from({ length: 12 }, (_, i) => partName('v.mp4', i + 1, 12));
ok(JSON.stringify([...many].sort()) === JSON.stringify(many),
   'part names sort into numeric order');
ok(partName('v.mp4', 1, 4) === 'v.mp4.part01of04', 'name states position and total');
ok(partName('v.mp4', 3, 3).endsWith('of03'), 'total is padded too');

// Round trip: split, concatenate, compare.
const size = 47 * MB + 12345;
const source = new Uint8Array(size);
// A pattern that must not repeat at the 20 MB part boundary. The obvious
// i*31+(i>>13) generator does: 31 * 20 MB is a multiple of 256, so parts 1 and
// 2 came out byte-identical and the "each part hashes differently" assertion
// failed against correct code. An xorshift has no such short period.
let seed = 0x2545f491;
for (let i = 0; i < size; i++) {
  seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
  source[i] = seed & 0xff;
}
const file = new File([source], 'clip.mp4', { type: 'video/mp4' });

const r = await splitFile(file);
ok(r.total === 3, `47 MB split into ${r.total} parts`);
ok(r.parts.every(p => p.size <= LIMIT_MB * MB),
   'every part is under the upload cap');
ok(r.parts.reduce((n, p) => n + p.size, 0) === size,
   'parts sum to the original size exactly');

const joined = new Uint8Array(await new Blob(r.parts).arrayBuffer());
ok(joined.length === source.length, 'rejoined length matches');
let identical = true;
for (let i = 0; i < source.length; i++) {
  if (joined[i] !== source[i]) { identical = false; break; }
}
ok(identical, 'rejoined bytes are identical to the original');

// The manifest is what lets the joiner refuse corrupt data.
const manifest = JSON.parse(await r.manifest.text());
ok(manifest.original === 'clip.mp4', 'manifest names the original file');
ok(manifest.size === size, 'manifest records the true size');
ok(manifest.parts === r.total, 'manifest records the part count');
ok(manifest.sha256.length === r.total, 'manifest carries a hash per part');
ok(manifest.sha256.every(h => /^[0-9a-f]{64}$/.test(h)), 'hashes are SHA-256 hex');
ok(new Set(manifest.sha256).size === r.total, 'each part hashes differently');

ok(formatMB(20 * MB) === '20.0', 'formats 20 MiB as 20.0');

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
