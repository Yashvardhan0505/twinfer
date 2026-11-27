#!/usr/bin/env python3
"""Create a tiny random Llama/Qwen2-style model in Hugging Face format for tests.

Writes config.json, tokenizer.json (byte-level BPE trained on a small built-in corpus,
same pre-tokenizer as SmolLM2) and model.safetensors. Only numpy is needed.

    python3 tools/make_tiny_model.py --out build/tiny
"""
import argparse
import collections
import json
import os
import struct
import unicodedata

import numpy as np

CORPUS = """
The quick brown fox jumps over the lazy dog. GPUs run thousands of threads in parallel.
A kernel is a function that runs on the GPU. Shared memory is fast; global memory is slow.
def matmul(a, b): return [[sum(x * y for x, y in zip(r, c)) for c in zip(*b)] for r in a]
In 2024 there were 1234 tokens per second. The answer is 42! Isn't it? We'll see, they're here.
नमस्ते दुनिया। यह एक परीक्षण है। 你好，世界！ Ünïcödé façade naïve café — résumé.
for (int i = 0; i < n; ++i) { y[i] += a * x[i]; }  // saxpy
Attention is all you need. Flash attention tiles the softmax so it never materialises the matrix.
""" * 3


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def rough_words(text):
    """Approximate GPT-2 pre-tokenization, good enough to train merges."""
    words, cur = [], ""
    for ch in text:
        cat = unicodedata.category(ch)
        if ch.isspace() or cat.startswith("N"):
            if cur:
                words.append(cur)
            cur = "" if cat.startswith("N") else ch
            if cat.startswith("N"):
                words.append(ch)
        else:
            cur += ch
    if cur:
        words.append(cur)
    return words


def train_bpe(text, n_merges):
    b2u = bytes_to_unicode()
    freqs = collections.Counter(rough_words(text))
    words = {w: [b2u[b] for b in w.encode("utf-8")] for w in freqs}
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, syms in words.items():
            for a, b in zip(syms, syms[1:]):
                pairs[(a, b)] += freqs[w]
        if not pairs:
            break
        (a, b), cnt = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        if cnt < 2:
            break
        merges.append((a, b))
        for w, syms in words.items():
            out, i = [], 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            words[w] = out
    return merges


QWEN2_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
               r"\s*[\r\n]+|\s+(?!\S)|\s+")

# Chat templates (tokenizer_config.json) shaped like the real ones: SmolLM2 writes the default system
# prompt with real newlines inside the Jinja string, Qwen2.5 with \n escapes.
SMOLLM_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a tiny test model.<|im_end|>\n' }}{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
QWEN2_TEMPLATE = (
    r"{%- if messages[0]['role'] == 'system' %}"
    r"{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}"
    r"{%- else %}{{- '<|im_start|>system\nYou are a tiny Qwen-style test model.<|im_end|>\n' }}{%- endif %}"
    r"{%- for message in messages %}{%- if message.role != 'system' %}"
    r"{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}{%- endif %}{%- endfor %}"
    r"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)


def make_tokenizer(n_merges, merges_as_arrays, pretok="smollm"):
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    b2u = bytes_to_unicode()
    vocab = {}
    for s in specials:
        vocab[s] = len(vocab)
    for b in range(256):
        vocab[b2u[b]] = len(vocab)
    merges = train_bpe(CORPUS, n_merges)
    for a, b in merges:
        if a + b not in vocab:
            vocab[a + b] = len(vocab)
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": True}
            for i, s in enumerate(specials)
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": True},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
            ],
        } if pretok == "smollm" else {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": QWEN2_REGEX}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
            "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": vocab,
            "merges": [[a, b] for a, b in merges] if merges_as_arrays else [f"{a} {b}" for a, b in merges],
        },
    }
    return tok, len(vocab)


def to_bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16  # round to nearest even
    return u.astype(np.uint16)


def write_safetensors(path, tensors, dtype):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        if dtype == "BF16":
            raw = to_bf16_bits(arr).tobytes()
        elif dtype == "F16":
            raw = arr.astype(np.float16).tobytes()
        else:
            raw = arr.astype(np.float32).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=688)
    ap.add_argument("--max-pos", type=int, default=512)
    ap.add_argument("--merges", type=int, default=300)
    ap.add_argument("--dtype", choices=["BF16", "F16", "F32"], default="BF16")
    ap.add_argument("--qkv-bias", action="store_true", help="Qwen2-style q/k/v bias")
    ap.add_argument("--untied", action="store_true", help="separate lm_head")
    ap.add_argument("--merges-as-arrays", action="store_true")
    ap.add_argument("--pretok", choices=["smollm", "qwen2"], default="smollm", help="pre-tokenizer style")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok, vocab_size = make_tokenizer(args.merges, args.merges_as_arrays, args.pretok)
    with open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)
    with open(os.path.join(args.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"chat_template": QWEN2_TEMPLATE if args.pretok == "qwen2" else SMOLLM_TEMPLATE,
                   "eos_token": "<|im_end|>"}, f, indent=2)

    D, H, KV, F = args.dim, args.heads, args.kv_heads, args.hidden
    hd = D // H
    cfg = {
        "architectures": ["Qwen2ForCausalLM" if args.qkv_bias else "LlamaForCausalLM"],
        "model_type": "qwen2" if args.qkv_bias else "llama",
        "hidden_size": D, "num_hidden_layers": args.layers, "num_attention_heads": H,
        "num_key_value_heads": KV, "intermediate_size": F, "vocab_size": vocab_size,
        "max_position_embeddings": args.max_pos, "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": not args.untied, "hidden_act": "silu", "rope_scaling": None,
        "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "bfloat16",
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    rng = np.random.default_rng(args.seed)

    def lin(o, i):
        return (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)

    t = {"model.embed_tokens.weight": (0.5 * rng.standard_normal((vocab_size, D))).astype(np.float32)}
    for l in range(args.layers):
        p = f"model.layers.{l}."
        t[p + "input_layernorm.weight"] = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)
        t[p + "self_attn.q_proj.weight"] = lin(H * hd, D)
        t[p + "self_attn.k_proj.weight"] = lin(KV * hd, D)
        t[p + "self_attn.v_proj.weight"] = lin(KV * hd, D)
        if args.qkv_bias:
            t[p + "self_attn.q_proj.bias"] = (0.1 * rng.standard_normal(H * hd)).astype(np.float32)
            t[p + "self_attn.k_proj.bias"] = (0.1 * rng.standard_normal(KV * hd)).astype(np.float32)
            t[p + "self_attn.v_proj.bias"] = (0.1 * rng.standard_normal(KV * hd)).astype(np.float32)
        t[p + "self_attn.o_proj.weight"] = lin(D, H * hd)
        t[p + "post_attention_layernorm.weight"] = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)
        t[p + "mlp.gate_proj.weight"] = lin(F, D)
        t[p + "mlp.up_proj.weight"] = lin(F, D)
        t[p + "mlp.down_proj.weight"] = lin(D, F)
    t["model.norm.weight"] = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    if args.untied:
        t["lm_head.weight"] = lin(vocab_size, D) * 4
    write_safetensors(os.path.join(args.out, "model.safetensors"), t, args.dtype)
    print(f"wrote {args.out}: vocab {vocab_size}, dim {D}, layers {args.layers}, heads {H}/{KV}, ffn {F}, {args.dtype}")


if __name__ == "__main__":
    main()
