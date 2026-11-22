// JS version of common/tokenizer.cpp (byte-level BPE from an HF tokenizer.json).
// Here the pre-tokenizer regexes just run on the JS regex engine with \p{L} / \p{N},
// the C++ one hand-implements them. tests/run_tests.py checks that both agree.

// \s in the HF (Oniguruma) regexes = Unicode White_Space. JS's \s differs (it includes U+FEFF and
// excludes U+0085), so spell the class out.
const WS = "\\t\\n\\v\\f\\r \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000";
const GPT2_RE = `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^${WS}\\p{L}\\p{N}]+|[${WS}]+(?![^${WS}])|[${WS}]+`;
const QWEN2_RE =
  `'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|` +
  ` ?[^${WS}\\p{L}\\p{N}]+[\\r\\n]*|[${WS}]*[\\r\\n]+|[${WS}]+(?![^${WS}])|[${WS}]+`;
const HF_GPT2 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
const HF_QWEN2 =
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

function bytesToUnicode() {
  const table = new Array(256);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    table[b] = String.fromCodePoint(printable ? b : 256 + n++);
  }
  return table;
}

export class Tokenizer {
  constructor(json) {
    const model = json.model;
    if (model.type !== "BPE") throw new Error("only BPE tokenizers are supported");
    this.individualDigits = false;
    this.addPrefixSpace = false;
    this.regex = null;
    const visit = (pt) => {
      if (pt.type === "Digits") this.individualDigits = !!pt.individual_digits;
      else if (pt.type === "ByteLevel") {
        this.addPrefixSpace = !!pt.add_prefix_space;
        if (pt.use_regex !== false) this.regex = GPT2_RE;
      } else if (pt.type === "Split") {
        const re = pt.pattern && pt.pattern.Regex;
        if (re === HF_GPT2) this.regex = GPT2_RE;
        else if (re === HF_QWEN2) this.regex = QWEN2_RE;
        else throw new Error("unsupported Split regex: " + re);
      } else throw new Error("unsupported pre-tokenizer " + pt.type);
    };
    const pre = json.pre_tokenizer;
    if (!pre) throw new Error("tokenizer has no pre_tokenizer");
    if (pre.type === "Sequence") pre.pretokenizers.forEach(visit);
    else visit(pre);
    if (!this.regex) throw new Error("expected a ByteLevel or Split pre-tokenizer");
    this.re = new RegExp(this.regex, "gu");

    const b2u = bytesToUnicode();
    const u2b = new Map(b2u.map((c, b) => [c, b]));
    this.vocab = new Map(Object.entries(model.vocab));
    let maxId = 0;
    for (const id of this.vocab.values()) maxId = Math.max(maxId, id);
    for (const a of json.added_tokens || []) maxId = Math.max(maxId, a.id);
    this.idToBytes = new Array(maxId + 1).fill(null);
    this.special = new Uint8Array(maxId + 1);
    const enc = new TextEncoder();
    for (const [tok, id] of this.vocab) {
      const bytes = [];
      let ok = true;
      for (const ch of tok) {
        const b = u2b.get(ch);
        if (b === undefined) { ok = false; break; }
        bytes.push(b);
      }
      this.idToBytes[id] = ok ? Uint8Array.from(bytes) : enc.encode(tok);
    }
    this.byteToId = b2u.map((c) => (this.vocab.has(c) ? this.vocab.get(c) : -1));
    this.added = [];
    for (const a of json.added_tokens || []) {
      this.added.push([a.content, a.id]);
      this.idToBytes[a.id] = enc.encode(a.content);
      if (a.special) this.special[a.id] = 1;
    }
    this.merges = new Map(); // (left * 2^20 + right) -> [rank, merged id]
    model.merges.forEach((m, rank) => {
      let a, b;
      if (typeof m === "string") {
        const sp = m.indexOf(" ");
        if (sp < 0) return;
        a = m.slice(0, sp);
        b = m.slice(sp + 1);
      } else {
        [a, b] = m;
      }
      const ia = this.vocab.get(a), ib = this.vocab.get(b), iab = this.vocab.get(a + b);
      if (ia === undefined || ib === undefined || iab === undefined) return;
      const key = ia * 1048576 + ib;
      if (!this.merges.has(key)) this.merges.set(key, [rank, iab]);
    });
    this.cache = new Map();
    this.encoder = new TextEncoder();
  }

  get vocabSize() {
    return this.idToBytes.length;
  }

  tokenId(content) {
    for (const [c, id] of this.added) if (c === content) return id;
    return this.vocab.has(content) ? this.vocab.get(content) : -1;
  }

  isSpecial(id) {
    return this.special[id] === 1;
  }

  pretokenize(text) {
    if (this.addPrefixSpace && text.length && text[0] !== " ") text = " " + text;
    const chunks = [];
    if (this.individualDigits) {
      let cur = "";
      for (const ch of text) {
        if (/\p{N}/u.test(ch)) {
          if (cur) chunks.push(cur);
          chunks.push(ch);
          cur = "";
        } else cur += ch;
      }
      if (cur) chunks.push(cur);
    } else chunks.push(text);
    const pieces = [];
    for (const c of chunks) for (const m of c.matchAll(this.re)) pieces.push(m[0]);
    return pieces;
  }

  bpe(piece, out) {
    const hit = this.cache.get(piece);
    if (hit) { out.push(...hit); return; }
    const sym = [];
    for (const b of this.encoder.encode(piece)) if (this.byteToId[b] >= 0) sym.push(this.byteToId[b]);
    while (sym.length > 1) {
      let best = Infinity, bi = -1, bid = -1;
      for (let i = 0; i + 1 < sym.length; i++) {
        const m = this.merges.get(sym[i] * 1048576 + sym[i + 1]);
        if (m && m[0] < best) { best = m[0]; bi = i; bid = m[1]; }
      }
      if (bi < 0) break;
      sym.splice(bi, 2, bid);
    }
    if (this.cache.size < 100000) this.cache.set(piece, sym);
    out.push(...sym);
  }

  encode(text) {
    const ids = [];
    let pos = 0;
    while (pos < text.length) {
      let at = -1, len = 0, id = -1;
      for (const [c, cid] of this.added) {
        if (!c) continue;
        const f = text.indexOf(c, pos);
        if (f < 0) continue;
        if (at < 0 || f < at || (f === at && c.length > len)) { at = f; len = c.length; id = cid; }
      }
      const end = at < 0 ? text.length : at;
      if (end > pos) for (const p of this.pretokenize(text.slice(pos, end))) this.bpe(p, ids);
      if (at < 0) break;
      ids.push(id);
      pos = at + len;
    }
    return ids;
  }

  tokenBytes(id) {
    return this.idToBytes[id] || new Uint8Array(0);
  }

  decode(ids, skipSpecial = true) {
    const parts = ids.filter((id) => !(skipSpecial && this.isSpecial(id))).map((id) => this.tokenBytes(id));
    const total = parts.reduce((a, p) => a + p.length, 0);
    const buf = new Uint8Array(total);
    let o = 0;
    for (const p of parts) { buf.set(p, o); o += p.length; }
    return new TextDecoder().decode(buf);
  }
}

export const SMOLLM_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

export function chatml(messages, defaultSystem = SMOLLM_SYSTEM, addGenerationPrompt = true) {
  let s = "";
  if (defaultSystem && (!messages.length || messages[0].role !== "system"))
    s += `<|im_start|>system\n${defaultSystem}<|im_end|>\n`;
  for (const m of messages) s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  if (addGenerationPrompt) s += "<|im_start|>assistant\n";
  return s;
}
