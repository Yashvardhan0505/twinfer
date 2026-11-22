#!/usr/bin/env python3
"""Independent numpy implementation of the Llama forward pass (full causal attention,
no KV cache). Used as the golden reference for the C++ CPU engine.

    python3 tools/reference.py MODEL_DIR --tokens 1,2,3 --out ref.bin
"""
import argparse
import json
import os
import struct

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = np.frombuffer(data[a:b], dtype=np.uint8)
        if info["dtype"] == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif info["dtype"] == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        elif info["dtype"] == "F32":
            arr = raw.view(np.float32).copy()
        else:
            raise ValueError(info["dtype"])
        out[name] = arr.reshape(info["shape"])
    return out


def rope_tables(cfg, n):
    hd = cfg["hidden_size"] // cfg["num_attention_heads"] if "head_dim" not in cfg else cfg["head_dim"]
    half = hd // 2
    theta = cfg.get("rope_theta", 10000.0)
    inv = np.array([np.float32(1.0 / theta ** (2 * i / hd)) for i in range(half)], dtype=np.float32)
    ang = np.arange(n, dtype=np.float32)[:, None] * inv[None, :]  # float32 product, like HF
    return np.cos(ang.astype(np.float64)).astype(np.float32), np.sin(ang.astype(np.float64)).astype(np.float32)


def rmsnorm(x, w, eps):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)).astype(np.float32) * w


def apply_rope(x, cos, sin):  # x: [T, heads, hd]
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)


def forward(model_dir, tokens):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    W = load_safetensors(os.path.join(model_dir, "model.safetensors"))
    D, H = cfg["hidden_size"], cfg["num_attention_heads"]
    KV = cfg.get("num_key_value_heads", H)
    hd = cfg.get("head_dim", D // H)
    eps = cfg.get("rms_norm_eps", 1e-6)
    T = len(tokens)
    cos, sin = rope_tables(cfg, T)
    x = W["model.embed_tokens.weight"][np.array(tokens)]
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        h = rmsnorm(x, W[p + "input_layernorm.weight"], eps)
        q = h @ W[p + "self_attn.q_proj.weight"].T
        k = h @ W[p + "self_attn.k_proj.weight"].T
        v = h @ W[p + "self_attn.v_proj.weight"].T
        if p + "self_attn.q_proj.bias" in W:
            q = q + W[p + "self_attn.q_proj.bias"]
            k = k + W[p + "self_attn.k_proj.bias"]
            v = v + W[p + "self_attn.v_proj.bias"]
        q = apply_rope(q.reshape(T, H, hd), cos, sin)
        k = apply_rope(k.reshape(T, KV, hd), cos, sin)
        v = v.reshape(T, KV, hd)
        rep = H // KV
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd) + mask[None]
        att = att - att.max(axis=-1, keepdims=True)
        att = np.exp(att)
        att = att / att.sum(axis=-1, keepdims=True)
        o = np.einsum("hts,shd->thd", att, v).reshape(T, H * hd)
        x = x + o @ W[p + "self_attn.o_proj.weight"].T
        h = rmsnorm(x, W[p + "post_attention_layernorm.weight"], eps)
        g = h @ W[p + "mlp.gate_proj.weight"].T
        u = h @ W[p + "mlp.up_proj.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ W[p + "mlp.down_proj.weight"].T
    x = rmsnorm(x, W["model.norm.weight"], eps)
    head = W.get("lm_head.weight", W["model.embed_tokens.weight"])
    return (x @ head.T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    toks = [int(t) for t in a.tokens.split(",")]
    forward(a.model, toks).tofile(a.out)


if __name__ == "__main__":
    main()
