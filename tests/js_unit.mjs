// Unit tests for the browser-side helpers that don't need a GPU. Run by tests/run_tests.py --cpu:
//   node tests/js_unit.mjs build/tiny/tokenizer.json
import { readFileSync } from "node:fs";

import { turnIds } from "../web/src/chat.js";
import { Sampler } from "../web/src/sampler.js";
import { Tokenizer, chatml } from "../web/src/tokenizer.js";

let failures = 0;
const check = (ok, what) => {
  console.log(`  ${what.padEnd(66)} ${ok ? "ok" : "FAIL"}`);
  if (!ok) failures++;
};
const throws = (fn) => {
  try {
    fn();
  } catch {
    return true;
  }
  return false;
};
const sampler = (temperature, topP) => new Sampler({ temperature, topP, seed: 7 });

console.log("[sampler.js]");
const n = 559;
const peaked = new Float32Array(n);
peaked[123] = 10;
const pinf = new Float32Array(n);
pinf[5] = Infinity;
check(throws(() => sampler(0.6, 0.9).sample(new Float32Array(n).fill(NaN))), "NaN logits are an error, not undefined");
check(throws(() => sampler(0.6, 0.9).sample(pinf)), "+inf logit is an error");
check(throws(() => sampler(0.6, 0.9).sample(new Float32Array(n).fill(-Infinity))), "all -inf logits are an error");
const tinyP = sampler(1e9, 1e-5);
check(
  Array.from({ length: 50 }, () => tinyP.sample(new Float32Array(n))).every((t) => Number.isInteger(t) && t >= 0 && t < n),
  "tiny top-p on a flat distribution still returns a valid id",
);
const sp = sampler(0.6, 0.9);
check(Array.from({ length: 50 }, () => sp.sample(peaked)).every((t) => t === 123), "nucleus of a peaked distribution is just the top token");
const st = sampler(1, 0.3);
check(Array.from({ length: 20 }, () => st.sample(new Float32Array([0, 5, 5, 0]))).every((t) => t === 1), "a tie at the top of the nucleus goes to the lower id");
check(sampler(0.6, 0.9).sample(new Float32Array([3])) === 0, "single-entry vocabulary");

console.log("[chat.js]");
const tokPath = process.argv[2];
if (tokPath) {
  const tok = new Tokenizer(JSON.parse(readFileSync(tokPath, "utf8")));
  const opts = { maxNew: 3, maxSeq: 512, system: "be brief" };
  const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  const firstIds = tok.encode(chatml([{ role: "user", content: "hi" }], "be brief"));
  const later = tok.encode("<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n");
  const t1 = turnIds(tok, { pos: 0, firstTurn: true, carry: -1 }, "hi", opts);
  check(same(t1.ids, firstIds) && !t1.reset, "first turn: system + user message in ChatML");
  const t2 = turnIds(tok, { pos: 40, firstTurn: false, carry: 42 }, "more", opts);
  check(t2.ids[0] === 42 && same(t2.ids.slice(1), later), "cut-off reply: its last token goes first");
  const t3 = turnIds(tok, { pos: 40, firstTurn: false, carry: -1 }, "more", opts);
  check(same(t3.ids, later), "reply that ended on a stop token: nothing carried");
  const t4 = turnIds(tok, { pos: 505, firstTurn: false, carry: 42 }, "hi", opts);
  check(t4.reset && same(t4.ids, firstIds), "context full: starts over without the carried token");
} else {
  console.log("  no tokenizer.json given, skipping");
}

console.log(failures ? `${failures} test(s) FAILED` : "all tests passed");
process.exit(failures ? 1 : 0);
