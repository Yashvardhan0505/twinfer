// WebGPU version of the decode pipeline, same steps as the CUDA engine:
// embed, then per layer: qkv gemv (+norm), rope/kv append, attention, o gemv (+residual),
// gate/up gemv (+norm, swiglu), down gemv (+residual). Then lm_head.
// Everything for one token goes into a single compute pass.
import { toFloat32 } from "./safetensors.js";

const SHADERS = ["gemv", "rope_kv", "attention", "misc"];
const SUPPORTED = ["LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM"];
const STORAGE = 0x0080, COPY_SRC = 0x0004, COPY_DST = 0x0008, UNIFORM = 0x0040, MAP_READ = 0x0001;

export async function initDevice() {
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser (try Chrome or Edge 113+).");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found.");
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
    },
  });
  const info = adapter.info || {};
  return { adapter, device, info };
}

export function parseConfig(cfg) {
  const arch = (cfg.architectures || [])[0] || "";
  if (!SUPPORTED.includes(arch)) throw new Error(`unsupported architecture ${arch}`);
  const rs = cfg.rope_scaling;
  if (rs && (rs.rope_type || rs.type) !== "default") throw new Error("rope_scaling is not supported");
  if ((cfg.hidden_act || "silu") !== "silu") throw new Error("only SiLU MLPs are supported");
  const c = {
    arch,
    dim: cfg.hidden_size,
    nLayers: cfg.num_hidden_layers,
    nHeads: cfg.num_attention_heads,
    nKvHeads: cfg.num_key_value_heads ?? cfg.num_attention_heads,
    headDim: cfg.head_dim ?? cfg.hidden_size / cfg.num_attention_heads,
    hidden: cfg.intermediate_size,
    vocab: cfg.vocab_size,
    maxPos: cfg.max_position_embeddings ?? 2048,
    ropeTheta: cfg.rope_theta ?? 10000,
    eps: cfg.rms_norm_eps ?? 1e-6,
    tied: !!cfg.tie_word_embeddings,
  };
  c.qDim = c.nHeads * c.headDim;
  c.kvDim = c.nKvHeads * c.headDim;
  c.qkvDim = c.qDim + 2 * c.kvDim;
  return c;
}

// Same table as common/engine.h rope_table(): float32 inv_freq and angle, like Hugging Face.
export function ropeTable(c, maxSeq) {
  const half = c.headDim / 2;
  const cos = new Float32Array(maxSeq * half), sin = new Float32Array(maxSeq * half);
  for (let i = 0; i < half; i++) {
    const inv = Math.fround(1.0 / Math.pow(c.ropeTheta, (2 * i) / c.headDim));
    for (let p = 0; p < maxSeq; p++) {
      const ang = Math.fround(p * inv);
      cos[p * half + i] = Math.cos(ang);
      sin[p * half + i] = Math.sin(ang);
    }
  }
  return { cos, sin };
}

const align = (n, a) => Math.ceil(n / a) * a;

export class Engine {
  static async create({ device, config, weights, maxSeq = 2048, onProgress = () => {} }) {
    const sources = {};
    await Promise.all(
      SHADERS.map(async (n) => {
        const r = await fetch(new URL(`./shaders/${n}.wgsl`, import.meta.url));
        if (!r.ok) throw new Error(`cannot load shader ${n}.wgsl`);
        sources[n] = await r.text();
      }),
    );
    const c = parseConfig(config);
    const e = new Engine(device, c, weights, Math.min(maxSeq, c.maxPos), sources);
    await e.init(onProgress);
    return e;
  }

  constructor(device, c, weights, maxSeq, sources) {
    this.device = device;
    this.c = c;
    this.weights = weights;
    this.maxSeq = maxSeq;
    this.sources = sources;
    this.pipelines = new Map();
    this.weightBytes = 0;
  }

  async module(name) {
    const m = this.device.createShaderModule({ code: this.sources[name], label: name });
    const info = await m.getCompilationInfo();
    const errs = info.messages.filter((x) => x.type === "error");
    if (errs.length) throw new Error(`${name}.wgsl: ` + errs.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("; "));
    return m;
  }

  pipeline(key, module, entryPoint, constants) {
    let p = this.pipelines.get(key);
    if (!p) {
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint, constants }, label: key });
      this.pipelines.set(key, p);
    }
    return p;
  }

  buffer(size, usage, label) {
    return this.device.createBuffer({ size: align(Math.max(size, 16), 16), usage, label });
  }

  f32Buffer(arr, label) {
    const b = this.buffer(arr.byteLength, STORAGE | COPY_DST, label);
    this.device.queue.writeBuffer(b, 0, arr);
    return b;
  }

  tensor(name, shape) {
    const t = this.weights.tensors.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    if (shape && (t.shape.length !== shape.length || t.shape.some((d, i) => d !== shape[i])))
      throw new Error(`shape mismatch for ${name}: [${t.shape}] vs [${shape}]`);
    return t;
  }

  vec(name, n) {
    return this.f32Buffer(toFloat32(this.tensor(name, [n])), name);
  }

  // Upload a row-major matrix assembled from row blocks ([{name, rows}]) as packed fp16.
  // bf16 data is converted in place on the GPU; f32 data goes through a staging buffer.
  matrix(parts, cols, label) {
    const rows = parts.reduce((a, p) => a + p.rows, 0);
    const words = (rows * cols) / 2;
    const dst = this.buffer(words * 4, STORAGE | COPY_DST, label);
    const dtype = this.tensor(parts[0].name).dtype;
    if (!["F16", "BF16", "F32"].includes(dtype)) throw new Error(`unsupported dtype ${dtype} in ${label}`);
    const q = this.device.queue;
    let staging = null;
    if (dtype === "F32") staging = this.buffer(rows * cols * 4, STORAGE | COPY_DST, label + ".f32");
    let row = 0;
    for (const p of parts) {
      const t = this.tensor(p.name, [p.rows, cols]);
      if (t.dtype !== dtype) throw new Error(`mixed dtypes in ${label}`);
      const elemBytes = dtype === "F32" ? 4 : 2;
      q.writeBuffer(staging || dst, row * cols * elemBytes, t.bytes);
      row += p.rows;
    }
    if (dtype !== "F16") {
      const entry = dtype === "BF16" ? "bf16_to_f16" : "f32_to_f16";
      const pipe = this.pipeline(`${entry}:${words}`, this.miscModule, entry, { COUNT: words });
      const entries =
        dtype === "BF16"
          ? [{ binding: 0, resource: { buffer: dst } }]
          : [{ binding: 0, resource: { buffer: staging } }, { binding: 1, resource: { buffer: dst } }];
      const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      const groups = Math.ceil(words / 256);
      const gx = Math.min(groups, 65535);
      pass.dispatchWorkgroups(gx, Math.ceil(groups / gx));
      pass.end();
      q.submit([enc.finish()]);
      if (staging) staging.destroy();
    }
    this.weightBytes += words * 4;
    return dst;
  }

  async init(onProgress) {
    const c = this.c, dev = this.device, T = this.weights.tensors;
    const D = c.dim, Q = c.qDim, KV = c.kvDim, F = c.hidden, V = c.vocab, hd = c.headDim;
    for (const k of [D, Q, F])
      if (k % 8) throw new Error("dim, n_heads*head_dim and ffn dim must be multiples of 8");
    if (hd % 2 || hd > 128) throw new Error("head_dim must be even and <= 128");
    c.qkvBias = T.has("model.layers.0.self_attn.q_proj.bias");
    if (!c.tied && !T.has("lm_head.weight")) c.tied = true;

    const [gemvM, ropeM, attnM, miscM] = await Promise.all(SHADERS.map((n) => this.module(n)));
    this.miscModule = miscM;

    // Weights.
    this.embed = this.matrix([{ name: "model.embed_tokens.weight", rows: V }], D, "embed");
    this.lmHead = c.tied ? this.embed : this.matrix([{ name: "lm_head.weight", rows: V }], D, "lm_head");
    this.finalNorm = this.vec("model.norm.weight", D);
    this.dummy = this.buffer(16, STORAGE, "dummy");
    this.layers = [];
    for (let l = 0; l < c.nLayers; l++) {
      const p = `model.layers.${l}.`;
      const L = {
        wqkv: this.matrix(
          [
            { name: p + "self_attn.q_proj.weight", rows: Q },
            { name: p + "self_attn.k_proj.weight", rows: KV },
            { name: p + "self_attn.v_proj.weight", rows: KV },
          ],
          D,
          `L${l}.qkv`,
        ),
        wo: this.matrix([{ name: p + "self_attn.o_proj.weight", rows: D }], Q, `L${l}.o`),
        wgu: this.matrix(
          [
            { name: p + "mlp.gate_proj.weight", rows: F },
            { name: p + "mlp.up_proj.weight", rows: F },
          ],
          D,
          `L${l}.gate_up`,
        ),
        wdown: this.matrix([{ name: p + "mlp.down_proj.weight", rows: D }], F, `L${l}.down`),
        attnNorm: this.vec(p + "input_layernorm.weight", D),
        ffnNorm: this.vec(p + "post_attention_layernorm.weight", D),
        bias: null,
        kc: this.buffer(c.nKvHeads * this.maxSeq * (hd / 2) * 4, STORAGE, `L${l}.kcache`),
        vc: this.buffer(c.nKvHeads * this.maxSeq * (hd / 2) * 4, STORAGE, `L${l}.vcache`),
      };
      if (c.qkvBias) {
        const b = new Float32Array(c.qkvDim);
        b.set(toFloat32(this.tensor(p + "self_attn.q_proj.bias", [Q])), 0);
        b.set(toFloat32(this.tensor(p + "self_attn.k_proj.bias", [KV])), Q);
        b.set(toFloat32(this.tensor(p + "self_attn.v_proj.bias", [KV])), Q + KV);
        L.bias = this.f32Buffer(b, `L${l}.bias`);
      }
      this.layers.push(L);
      onProgress((l + 1) / c.nLayers);
      if (l % 4 === 3) await dev.queue.onSubmittedWorkDone(); // keep staging memory bounded
    }
    const rope = ropeTable(c, this.maxSeq);
    this.ropeCos = this.f32Buffer(rope.cos, "rope_cos");
    this.ropeSin = this.f32Buffer(rope.sin, "rope_sin");

    // Activations.
    this.params = this.buffer(16, UNIFORM | COPY_DST, "params");
    this.x = this.buffer(D * 4, STORAGE | COPY_DST, "x");
    this.qkv = this.buffer(c.qkvDim * 4, STORAGE, "qkv");
    this.q = this.buffer(Q * 4, STORAGE, "q");
    this.attn = this.buffer(Q * 4, STORAGE, "attn");
    this.h = this.buffer(F * 4, STORAGE, "h");
    this.logits = this.buffer(V * 4, STORAGE | COPY_SRC, "logits");
    this.scores = this.buffer(c.nHeads * this.maxSeq * 4, STORAGE, "scores");
    this.argmaxOut = this.buffer(16, STORAGE | COPY_SRC, "argmax");
    this.logitsRead = this.buffer(V * 4, MAP_READ | COPY_DST, "logits_read");
    this.argmaxRead = this.buffer(16, MAP_READ | COPY_DST, "argmax_read");

    // Pipelines.
    const shape = { H: c.nHeads, HKV: c.nKvHeads, HD: hd, MAXT: this.maxSeq };
    const gemv = (K, N, norm, mode, bias) =>
      this.pipeline(`gemv:${K}:${N}:${norm}:${mode}:${bias}`, gemvM, "main", {
        K,
        N,
        NORM: norm ? 1 : 0,
        MODE: mode,
        HAS_BIAS: bias ? 1 : 0,
        EPS: c.eps,
      });
    this.pEmbed = this.pipeline("embed", miscM, "embed", { D });
    this.pArgmax = this.pipeline("argmax", miscM, "argmax", { COUNT: V });
    this.pRope = this.pipeline("rope", ropeM, "main", shape);
    this.pAttn = this.pipeline("attn", attnM, "main", shape);
    this.pQKV = gemv(D, c.qkvDim, true, 0, c.qkvBias);
    this.pO = gemv(Q, D, false, 1, false);
    this.pGU = gemv(D, F, true, 2, false);
    this.pDown = gemv(F, D, false, 1, false);
    this.pHead = gemv(D, V, true, 0, false);

    // Bind groups (static: every per-token value lives in the `params` uniform).
    const bg = (pipe, buffers) =>
      dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
      });
    const gemvBG = (pipe, W, x, gamma, bias, out) => bg(pipe, [W, x, gamma || this.dummy, bias || this.dummy, out]);
    this.bgEmbed = bg(this.pEmbed, [this.embed, this.params, this.x]);
    this.bgArgmax = bg(this.pArgmax, [this.logits, this.argmaxOut]);
    this.bgHead = gemvBG(this.pHead, this.lmHead, this.x, this.finalNorm, null, this.logits);
    for (const L of this.layers) {
      L.bgQKV = gemvBG(this.pQKV, L.wqkv, this.x, L.attnNorm, L.bias, this.qkv);
      L.bgRope = bg(this.pRope, [this.qkv, this.ropeCos, this.ropeSin, this.params, this.q, L.kc, L.vc]);
      L.bgAttn = bg(this.pAttn, [this.q, L.kc, L.vc, this.params, this.scores, this.attn]);
      L.bgO = gemvBG(this.pO, L.wo, this.attn, null, null, this.x);
      L.bgGU = gemvBG(this.pGU, L.wgu, this.x, L.ffnNorm, null, this.h);
      L.bgDown = gemvBG(this.pDown, L.wdown, this.h, null, null, this.x);
    }
    await dev.queue.onSubmittedWorkDone();
    this.weights = null; // release the CPU copy
  }

  dispatchRows(pass, pipe, group, rows) {
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    const wgs = Math.ceil(rows / 8);
    const gx = Math.min(wgs, 65535);
    pass.dispatchWorkgroups(gx, Math.ceil(wgs / gx));
  }

  encodeToken(pass) {
    const c = this.c;
    pass.setPipeline(this.pEmbed);
    pass.setBindGroup(0, this.bgEmbed);
    pass.dispatchWorkgroups(Math.ceil(c.dim / 2 / 256));
    for (const L of this.layers) {
      this.dispatchRows(pass, this.pQKV, L.bgQKV, c.qkvDim);
      pass.setPipeline(this.pRope);
      pass.setBindGroup(0, L.bgRope);
      pass.dispatchWorkgroups(c.nHeads + 2 * c.nKvHeads);
      pass.setPipeline(this.pAttn);
      pass.setBindGroup(0, L.bgAttn);
      pass.dispatchWorkgroups(c.nHeads);
      this.dispatchRows(pass, this.pO, L.bgO, c.dim);
      this.dispatchRows(pass, this.pGU, L.bgGU, c.hidden);
      this.dispatchRows(pass, this.pDown, L.bgDown, c.dim);
    }
    this.dispatchRows(pass, this.pHead, this.bgHead, c.vocab);
  }

  // Run tokens at positions [pos0, pos0 + n). Returns the last token's logits (Float32Array),
  // or its argmax when `greedy` (only 4 bytes are read back).
  async forward(tokens, pos0, { greedy = false } = {}) {
    if (pos0 + tokens.length > this.maxSeq) throw new Error("context length exceeded");
    const q = this.device.queue;
    for (let i = 0; i < tokens.length; i++) {
      const last = i === tokens.length - 1;
      q.writeBuffer(this.params, 0, new Uint32Array([tokens[i], pos0 + i, 0, 0]));
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      this.encodeToken(pass);
      if (last && greedy) {
        pass.setPipeline(this.pArgmax);
        pass.setBindGroup(0, this.bgArgmax);
        pass.dispatchWorkgroups(1);
      }
      pass.end();
      if (last) {
        if (greedy) enc.copyBufferToBuffer(this.argmaxOut, 0, this.argmaxRead, 0, 4);
        else enc.copyBufferToBuffer(this.logits, 0, this.logitsRead, 0, this.c.vocab * 4);
      }
      q.submit([enc.finish()]);
    }
    const rb = greedy ? this.argmaxRead : this.logitsRead;
    await rb.mapAsync(GPUMapMode.READ);
    let result;
    if (greedy) result = new Uint32Array(rb.getMappedRange(0, 4))[0];
    else result = new Float32Array(rb.getMappedRange(0, this.c.vocab * 4).slice(0));
    rb.unmap();
    return result;
  }
}
