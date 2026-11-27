// Fetch Hugging Face model files (with progress + Cache Storage) and parse safetensors.

const CACHE_NAME = "twinfer-models-v1";

export async function fetchBytes(url, onProgress = () => {}) {
  let resp = null;
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    resp = await cache.match(url);
  } catch (_) {
    cache = null; // Cache Storage unavailable (e.g. insecure context)
  }
  const cached = !!resp;
  if (!resp) {
    resp = await fetch(url);
    if (!resp.ok) {
      const err = new Error(`HTTP ${resp.status} fetching ${url}`);
      err.status = resp.status;
      throw err;
    }
    if (cache && url.startsWith("http")) {
      cache.put(url, resp.clone()).catch(() => {}); // best effort (quota)
    }
  }
  const total = Number(resp.headers.get("Content-Length")) || 0;
  const reader = resp.body.getReader();
  let buf = new Uint8Array(total || 1 << 20);
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (size + value.length > buf.length) {
      const grown = new Uint8Array(Math.max(buf.length * 2, size + value.length));
      grown.set(buf.subarray(0, size));
      buf = grown;
    }
    buf.set(value, size);
    size += value.length;
    onProgress(size, total, cached);
  }
  return size === buf.length ? buf.buffer : buf.slice(0, size).buffer;
}

export async function fetchJson(url) {
  const bytes = await fetchBytes(url);
  return JSON.parse(new TextDecoder().decode(bytes));
}

// A file that may not exist: its text, or null on a 404.
async function fetchOptionalText(url) {
  try {
    return new TextDecoder().decode(await fetchBytes(url));
  } catch (e) {
    if (e.status === 404) return null;
    throw e;
  }
}

// config.json, tokenizer.json and the weights of a model folder at a URL prefix (a Hugging Face
// repo's resolve/main/, say), plus the chat template if there is one (tokenizer_config.json and/or
// chat_template.jinja, which may be null). onProgress(file, bytesSoFar, totalBytes, fromCache)
// reports downloads. Sharded checkpoints are only tried when model.safetensors doesn't exist (404):
// any other failure, such as a dropped connection or a gated repo, is reported as it is.
export async function loadModelFromUrl(base, onProgress = () => {}) {
  const config = await fetchJson(base + "config.json");
  const tokenizer = await fetchJson(base + "tokenizer.json");
  const tokenizerConfig = JSON.parse((await fetchOptionalText(base + "tokenizer_config.json")) ?? "null");
  const chatTemplate = await fetchOptionalText(base + "chat_template.jinja");
  const progress = (file) => (got, total, cached) => onProgress(file, got, total, cached);
  let weights;
  try {
    weights = parseSafetensors(await fetchBytes(base + "model.safetensors", progress("model.safetensors")));
  } catch (e) {
    if (e.status !== 404) throw e;
    const index = await fetchJson(base + "model.safetensors.index.json");
    const files = [...new Set(Object.values(index.weight_map))];
    const shards = [];
    for (const f of files) shards.push(parseSafetensors(await fetchBytes(base + f, progress(f))));
    weights = mergeShards(shards);
  }
  return { config, tokenizer, tokenizerConfig, chatTemplate, weights };
}

// Returns { tensors: Map<name, {dtype, shape, bytes: Uint8Array}> }
export function parseSafetensors(buffer) {
  const view = new DataView(buffer);
  const lo = view.getUint32(0, true);
  const hi = view.getUint32(4, true);
  const headerLen = hi * 2 ** 32 + lo;
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen)));
  const dataStart = 8 + headerLen;
  const tensors = new Map();
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [b, e] = info.data_offsets;
    tensors.set(name, {
      dtype: info.dtype,
      shape: info.shape,
      bytes: new Uint8Array(buffer, dataStart + b, e - b),
    });
  }
  return { tensors };
}

// Merge several parsed shards into one tensor map.
export function mergeShards(shards) {
  const tensors = new Map();
  for (const s of shards) for (const [k, v] of s.tensors) tensors.set(k, v);
  return { tensors };
}

// fp32 copy of a (small) tensor, for norms and biases.
export function toFloat32(t) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(n);
  if (t.dtype === "F32") {
    out.set(new Float32Array(t.bytes.slice().buffer));
  } else if (t.dtype === "BF16") {
    const u16 = new Uint16Array(t.bytes.slice().buffer);
    const u32 = new Uint32Array(out.buffer);
    for (let i = 0; i < n; i++) u32[i] = u16[i] << 16;
  } else if (t.dtype === "F16") {
    const u16 = new Uint16Array(t.bytes.slice().buffer);
    for (let i = 0; i < n; i++) out[i] = f16ToF32(u16[i]);
  } else {
    throw new Error(`unsupported dtype ${t.dtype}`);
  }
  return out;
}

export function f16ToF32(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
