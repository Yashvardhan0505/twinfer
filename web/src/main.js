// Chat UI for the WebGPU engine.
//   ?model=org/name   load any Hugging Face repo with a supported Llama-architecture model
//   ?base=./models/x/ load config.json / tokenizer.json / model.safetensors from a URL prefix
import { defaultSystemPrompt, shownBytes, turnIds } from "./chat.js";
import { Engine, initDevice } from "./engine.js";
import { loadModelFromUrl, mergeShards, parseSafetensors } from "./safetensors.js";
import { Sampler } from "./sampler.js";
import { Tokenizer } from "./tokenizer.js";

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

const state = {
  gpu: null, engine: null, tok: null, system: "", stopIds: [], pos: 0, firstTurn: true, carry: -1, busy: false,
  stop: false,
};

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

function loadFromUrlBase(base) {
  return loadModelFromUrl(base, (label, got, total, cached) =>
    setProgress(total ? got / total : 0, `${label}${cached ? " (cache)" : ""}: ${mb(got)} / ${total ? mb(total) : "?"} MB`));
}

async function loadFromFiles(fileList) {
  const files = new Map([...fileList].map((f) => [f.name, f]));
  const need = (n) => {
    if (!files.has(n)) throw new Error(`folder is missing ${n}`);
    return files.get(n);
  };
  const config = JSON.parse(await need("config.json").text());
  const tokenizer = JSON.parse(await need("tokenizer.json").text());
  const optional = async (n) => (files.has(n) ? files.get(n).text() : null);
  const tokenizerConfig = JSON.parse((await optional("tokenizer_config.json")) ?? "null");
  const chatTemplate = await optional("chat_template.jinja");
  const shards = [];
  for (const [name, f] of files)
    if (name.endsWith(".safetensors")) shards.push(parseSafetensors(await f.arrayBuffer()));
  if (!shards.length) throw new Error("no .safetensors file in folder");
  return { config, tokenizer, tokenizerConfig, chatTemplate, weights: mergeShards(shards) };
}

async function load(getFiles, label) {
  setBusy(true);
  if (state.engine) state.engine.destroy(); // free the previous model's GPU memory first
  state.engine = null;
  try {
    if (!state.gpu) state.gpu = await initDevice();
    const t0 = performance.now();
    setProgress(0, "downloading...");
    let { config, tokenizer, tokenizerConfig, chatTemplate, weights } = await getFiles();
    state.tok = new Tokenizer(tokenizer);
    state.system = defaultSystemPrompt(tokenizerConfig, chatTemplate); // the model's own, like HF
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
  state.carry = -1;
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
    const { ids, reset } = turnIds(tok, state, text, { maxNew, maxSeq: eng.maxSeq, system: state.system });
    if (reset) state.pos = 0; // context full: start over, keeping only this message
    state.carry = -1;
    const step = async (toks, pos) =>
      sampler.greedy ? eng.forward(toks, pos, { greedy: true }) : sampler.sample(await eng.forward(toks, pos));

    const t0 = performance.now();
    let next = await step(ids, state.pos);
    state.pos += ids.length;
    const tPrefill = performance.now() - t0;

    const decoder = new TextDecoder();
    let generated = 0, steps = 0, cut = false; // steps = single-token forward passes timed below
    const t1 = performance.now();
    while (generated < maxNew && !state.stop && !state.stopIds.includes(next)) {
      out.textContent += decoder.decode(shownBytes(tok, next), { stream: true });
      ui.chat.scrollTop = ui.chat.scrollHeight;
      generated++;
      if (generated === maxNew || state.pos + 1 >= eng.maxSeq) { cut = true; break; }
      next = await step([next], state.pos);
      state.pos++;
      steps++;
    }
    if (cut) state.carry = next; // shown but not fed yet: it goes in front of the next turn
    out.textContent += decoder.decode();
    const tDecode = performance.now() - t1;
    const msPerTok = steps ? tDecode / steps : 0;
    const gbps = msPerTok ? eng.weightBytes / (msPerTok / 1000) / 1e9 : 0;
    const bw = gbps >= 1 ? `${gbps.toFixed(1)} GB/s` : `${(gbps * 1000).toFixed(0)} MB/s`;
    ui.stats.innerHTML =
      `prefill <b>${ids.length}</b> tokens in ${tPrefill.toFixed(0)} ms | ` +
      `decode <b>${(msPerTok ? 1000 / msPerTok : 0).toFixed(1)}</b> tok/s (${msPerTok.toFixed(1)} ms/token) | ` +
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
