// Chat UI for the WebGPU engine.
//   ?model=org/name   load any Hugging Face repo with a supported Llama-architecture model
//   ?base=./models/x/ load config.json / tokenizer.json / model.safetensors from a URL prefix
import { Engine, initDevice } from "./engine.js";
import { fetchBytes, fetchJson, mergeShards, parseSafetensors } from "./safetensors.js";
import { Sampler } from "./sampler.js";
import { SMOLLM_SYSTEM, Tokenizer, chatml } from "./tokenizer.js";

const MODELS = [
  { id: "HuggingFaceTB/SmolLM2-135M-Instruct", label: "SmolLM2-135M-Instruct (270 MB)" },
  { id: "HuggingFaceTB/SmolLM2-360M-Instruct", label: "SmolLM2-360M-Instruct (724 MB)" },
];
const MAX_SEQ = 2048;

const $ = (id) => document.getElementById(id);
const ui = {
  select: $("model-select"), load: $("load-btn"), folder: $("folder-input"), progress: $("progress"),
  bar: $("bar"), progressText: $("progress-text"), info: $("info"), chat: $("chat"), empty: $("empty"),
  input: $("input"), send: $("send-btn"), stop: $("stop-btn"), reset: $("reset-btn"), temp: $("temp"),
  topP: $("top-p"), maxNew: $("max-new"), stats: $("stats"),
};

const state = { gpu: null, engine: null, tok: null, stopIds: [], pos: 0, firstTurn: true, busy: false, stop: false };

const qs = new URLSearchParams(location.search);
if (qs.get("model") && !MODELS.some((m) => m.id === qs.get("model")))
  MODELS.unshift({ id: qs.get("model"), label: qs.get("model") });
for (const m of MODELS) ui.select.add(new Option(m.label, m.id));
if (qs.get("model")) ui.select.value = qs.get("model");

const mb = (n) => (n / 1e6).toFixed(n > 1e8 ? 0 : 1);

function setInfo(text, error = false) {
  ui.info.textContent = text;
  ui.info.classList.toggle("error", error);
}

function setProgress(frac, text) {
  ui.progress.hidden = frac === null;
  if (frac !== null) {
    ui.bar.style.width = `${Math.round(frac * 100)}%`;
    ui.progressText.textContent = text;
  }
}

function addMessage(role, text = "") {
  ui.empty.hidden = true;
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  ui.chat.appendChild(el);
  ui.chat.scrollTop = ui.chat.scrollHeight;
  return el;
}

function setBusy(busy) {
  state.busy = busy;
  ui.send.disabled = busy || !state.engine;
  ui.input.disabled = !state.engine;
  ui.stop.disabled = !busy;
  ui.load.disabled = busy;
}

// Loading

async function loadFromUrlBase(base) {
  const config = await fetchJson(base + "config.json");
  const tokenizer = await fetchJson(base + "tokenizer.json");
  const progress = (label) => (got, total, cached) =>
    setProgress(total ? got / total : 0, `${label}${cached ? " (cache)" : ""}: ${mb(got)} / ${total ? mb(total) : "?"} MB`);
  let weights;
  try {
    weights = parseSafetensors(await fetchBytes(base + "model.safetensors", progress("model.safetensors")));
  } catch (e) {
    const index = await fetchJson(base + "model.safetensors.index.json"); // sharded checkpoint
    const files = [...new Set(Object.values(index.weight_map))];
    const shards = [];
    for (const f of files) shards.push(parseSafetensors(await fetchBytes(base + f, progress(f))));
    weights = mergeShards(shards);
  }
  return { config, tokenizer, weights };
}

async function loadFromFiles(fileList) {
  const files = new Map([...fileList].map((f) => [f.name, f]));
  const need = (n) => {
    if (!files.has(n)) throw new Error(`folder is missing ${n}`);
    return files.get(n);
  };
  const config = JSON.parse(await need("config.json").text());
  const tokenizer = JSON.parse(await need("tokenizer.json").text());
  const shards = [];
  for (const [name, f] of files)
    if (name.endsWith(".safetensors")) shards.push(parseSafetensors(await f.arrayBuffer()));
  if (!shards.length) throw new Error("no .safetensors file in folder");
  return { config, tokenizer, weights: mergeShards(shards) };
}

async function load(getFiles, label) {
  setBusy(true);
  state.engine = null;
  try {
    if (!state.gpu) state.gpu = await initDevice();
    const t0 = performance.now();
    setProgress(0, "downloading...");
    let { config, tokenizer, weights } = await getFiles();
    state.tok = new Tokenizer(tokenizer);
    setProgress(0, "uploading weights to the GPU...");
    state.engine = await Engine.create({
      device: state.gpu.device, config, weights, maxSeq: MAX_SEQ,
      onProgress: (f) => setProgress(f, `uploading + converting weights: ${Math.round(f * 100)}%`),
    });
    weights = null;
    state.stopIds = [state.tok.tokenId("<|im_end|>"), state.tok.tokenId("<|endoftext|>")]
      .concat([].concat(config.eos_token_id ?? []))
      .filter((x) => x >= 0);
    const { vendor = "", architecture = "", description = "" } = state.gpu.info;
    const c = state.engine.c;
    setProgress(null);
    const adapter = [vendor, architecture, description].filter(Boolean).join(" ") || "unknown";
    setInfo(`${label}: ${c.nLayers} layers, d=${c.dim}, heads ${c.nHeads}/${c.nKvHeads}, ` +
      `${mb(state.engine.weightBytes)} MB on the GPU | adapter: ${adapter} | ` +
      `loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    resetChat();
  } catch (e) {
    console.error(e);
    setProgress(null);
    setInfo(String(e.message || e), true);
  } finally {
    setBusy(false);
    if (state.engine) ui.input.focus();
  }
}

ui.load.onclick = () => {
  const id = ui.select.value;
  const base = qs.get("base") || `https://huggingface.co/${id}/resolve/main/`;
  load(() => loadFromUrlBase(base), qs.get("base") ? base : id);
};
ui.folder.onchange = () => {
  if (ui.folder.files.length) load(() => loadFromFiles(ui.folder.files), "local folder");
};

// Chat

function resetChat() {
  state.pos = 0;
  state.firstTurn = true;
  ui.chat.querySelectorAll(".msg").forEach((m) => m.remove());
  ui.empty.hidden = false;
}

async function send() {
  const text = ui.input.value.trim();
  if (!text || state.busy || !state.engine) return;
  ui.input.value = "";
  setBusy(true);
  state.stop = false;
  const eng = state.engine, tok = state.tok;
  addMessage("user", text);
  const out = addMessage("assistant");
  out.classList.add("streaming");
  const maxNew = Math.max(1, Number(ui.maxNew.value) || 256);
  const sampler = new Sampler({
    temperature: Number(ui.temp.value),
    topP: Number(ui.topP.value),
    seed: (Math.random() * 2 ** 31) | 0,
  });
  try {
    let prompt = state.firstTurn
      ? chatml([{ role: "user", content: text }], SMOLLM_SYSTEM)
      : `<|im_end|>\n<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
    let ids = tok.encode(prompt);
    if (state.pos + ids.length + maxNew >= eng.maxSeq) {
      state.pos = 0; // context full: start over, keeping only this message
      ids = tok.encode(chatml([{ role: "user", content: text }], SMOLLM_SYSTEM));
    }
    const step = async (toks, pos) =>
      sampler.greedy ? eng.forward(toks, pos, { greedy: true }) : sampler.sample(await eng.forward(toks, pos));

    const t0 = performance.now();
    let next = await step(ids, state.pos);
    state.pos += ids.length;
    const tPrefill = performance.now() - t0;

    const decoder = new TextDecoder();
    let generated = 0;
    const t1 = performance.now();
    while (generated < maxNew && !state.stop && !state.stopIds.includes(next)) {
      out.textContent += decoder.decode(tok.tokenBytes(next), { stream: true });
      ui.chat.scrollTop = ui.chat.scrollHeight;
      generated++;
      if (generated === maxNew || state.pos + 1 >= eng.maxSeq) break;
      next = await step([next], state.pos);
      state.pos++;
    }
    out.textContent += decoder.decode();
    const tDecode = performance.now() - t1;
    const msPerTok = generated > 1 ? tDecode / (generated - 1) : 0;
    const gbps = msPerTok ? eng.weightBytes / (msPerTok / 1000) / 1e9 : 0;
    const bw = gbps >= 1 ? `${gbps.toFixed(1)} GB/s` : `${(gbps * 1000).toFixed(0)} MB/s`;
    ui.stats.innerHTML =
      `prefill <b>${ids.length}</b> tokens in ${tPrefill.toFixed(0)} ms | ` +
      `decode <b>${(generated > 1 ? (1000 / msPerTok) : 0).toFixed(1)}</b> tok/s (${msPerTok.toFixed(1)} ms/token) | ` +
      `weights read at <b>${bw}</b> | context ${state.pos}/${eng.maxSeq}`;
    state.firstTurn = false;
  } catch (e) {
    console.error(e);
    out.classList.add("error");
    out.textContent += `\n[error: ${e.message || e}]`;
  } finally {
    out.classList.remove("streaming");
    setBusy(false);
    ui.input.focus();
  }
}

ui.send.onclick = send;
ui.stop.onclick = () => (state.stop = true);
ui.reset.onclick = () => !state.busy && resetChat();
ui.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

if (!navigator.gpu) {
  setInfo("WebGPU is not available in this browser. Try a recent Chrome or Edge (desktop or Android).", true);
  ui.load.disabled = true;
}
