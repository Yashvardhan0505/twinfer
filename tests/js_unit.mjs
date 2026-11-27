// Unit tests for the browser-side helpers that don't need a GPU. Run by tests/run_tests.py --cpu:
//   node tests/js_unit.mjs build/tiny/tokenizer.json
import { readFileSync } from "node:fs";
import { createServer } from "node:http";

import { defaultSystemPrompt, shownBytes, templateSystemPrompt, turnIds } from "../web/src/chat.js";
import { Sampler } from "../web/src/sampler.js";
import { loadModelFromUrl } from "../web/src/safetensors.js";
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
  const imStart = tok.tokenId("<|im_start|>"), word = tok.encode("hi")[0];
  check(shownBytes(tok, imStart).length === 0 && shownBytes(tok, word).length > 0, "special tokens aren't shown");

  console.log("[tokenizer.js normalizers]");
  const json = JSON.parse(readFileSync(tokPath, "utf8"));
  const nfc = new Tokenizer({ ...json, normalizer: { type: "NFC" } });
  const decomposed = "café", composed = "café";
  check(!same(tok.encode(decomposed), tok.encode(composed)), "without a normalizer, e + U+0301 and é differ");
  check(same(nfc.encode(decomposed), nfc.encode(composed)), "NFC normalizer: e + U+0301 tokenizes like é");
  check(same(nfc.encode("<|im_start|>x"), tok.encode("<|im_start|>x")), "special tokens are still matched as typed");
  check(throws(() => new Tokenizer({ ...json, normalizer: { type: "Lowercase" } })), "unsupported normalizer is an error");
} else {
  console.log("  no tokenizer.json given, skipping");
}

console.log("[chat.js: default system prompt]");
{
  const smol = "{% if loop.first %}{{ '<|im_start|>system\nYou are SmolLM, a test.<|im_end|>\n' }}{% endif %}";
  const qwen =
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}{%- else %}" +
    "{{- '<|im_start|>system\\nYou are Qwen, a test.<|im_end|>\\n' }}";
  check(templateSystemPrompt(smol) === "You are SmolLM, a test.", "SmolLM2-style template (newline in the string)");
  check(templateSystemPrompt(qwen) === "You are Qwen, a test.", "Qwen2.5-style template (\\n escapes, code first)");
  check(templateSystemPrompt("{{ '<|im_start|>' + m.role + '\\n' }}") === "", "template without a default: none");
  const named = { chat_template: [{ name: "tool_use", template: qwen }, { name: "default", template: smol }] };
  check(defaultSystemPrompt(named, null) === "You are SmolLM, a test.", "list of named templates: the default one");
  check(defaultSystemPrompt({ chat_template: smol }, qwen) === "You are Qwen, a test.", "chat_template.jinja wins");
  check(defaultSystemPrompt(null, null) === "", "no metadata: none");
}

console.log("[safetensors.js: loadModelFromUrl]");
{
  // a: model.safetensors fails with a 500. b: it's missing (404), the weights are sharded.
  const header = new TextEncoder().encode(JSON.stringify({ t: { dtype: "F32", shape: [1], data_offsets: [0, 4] } }));
  const shard = new Uint8Array(8 + header.length + 4);
  new DataView(shard.buffer).setUint32(0, header.length, true);
  shard.set(header, 8);
  const files = {
    "/a/config.json": "{}", "/a/tokenizer.json": "{}",
    "/b/config.json": "{}", "/b/tokenizer.json": "{}",
    "/b/model.safetensors.index.json": JSON.stringify({ weight_map: { t: "model-00001-of-00001.safetensors" } }),
    "/b/model-00001-of-00001.safetensors": shard,
  };
  const server = createServer((req, res) => {
    if (req.url === "/a/model.safetensors") return res.writeHead(500).end();
    if (!(req.url in files)) return res.writeHead(404).end();
    res.writeHead(200).end(files[req.url]);
  });
  await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
  const base = `http://127.0.0.1:${server.address().port}`;
  const err = await loadModelFromUrl(`${base}/a/`).then(() => null, (e) => e);
  check(err !== null && err.status === 500, "a server error is reported as is, not as a missing shard index");
  const b = await loadModelFromUrl(`${base}/b/`).catch(() => null);
  check(b !== null && b.weights.tensors.has("t"), "a missing model.safetensors falls back to the shards");
  server.close();
}

console.log(failures ? `${failures} test(s) FAILED` : "all tests passed");
process.exit(failures ? 1 : 0);
