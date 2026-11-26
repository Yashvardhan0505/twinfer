// Unit tests for the browser-side helpers that don't need a GPU. Run by tests/run_tests.py --cpu:
//   node tests/js_unit.mjs
import { Sampler } from "../web/src/sampler.js";

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

console.log(failures ? `${failures} test(s) FAILED` : "all tests passed");
process.exit(failures ? 1 : 0);
