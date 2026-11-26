#!/usr/bin/env python3
"""Correctness tests on tiny random models (no downloads needed).

  --cpu              C++ CPU engine vs the numpy reference; C++ tokenizer vs the JS tokenizer (Node)
  --cuda             CUDA engine vs the CPU engine (decode/graph, chunked prefill, int8, bias/untied)
  --build-dir DIR    where the binaries are and the test models go (default: build/)
  --cuda-bin PATH    CUDA binary to test (default build/twinfer-cuda; the emulator passes its own)
  --quick            fewer tokens (for the slow CPU emulator)
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = ".exe" if os.name == "nt" else ""
BUILD = os.path.join(ROOT, "build")
CPU = os.path.join(BUILD, "twinfer-cpu" + EXE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

FIXED = [
    "Hello world", "  leading spaces", "trailing   ", "tabs\tand\nnewlines\n\n", "it's they're we'll I'M 'S",
    "numbers 12345 and 3.14", "emoji 😀🚀 mixed", "नमस्ते दुनिया", "你好，世界！", "Ünïcödé façade",
    "<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n", "a b", "x　y", "\r\n windows\r\n",
    "''quoted'' 's", " ", "", "multiple\n\n\n\nnewlines   \n  x", "日本語123", "Ⅻ ½ ²", "é accent",
    "Привет мир", "مرحبا ١٢٣", "zero​width", "BOM﻿here", "NEL\u0085line", "#$%&*!!?? ... ---",
]
POOL = list("abcXYZ019 \t\n\r.,;:'!?-_()[]{}\"") + ["é", "ß", "न", "ि", "ं", "你", "。", "😀", " ", " ",
                                                     "½", "²", "٣", "́", "—", "’", "　", "Ω"]
TOKENS = [342, 531, 500, 492, 543, 454, 35, 52, 53, 535, 554, 118, 49]
VOCAB = 559


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def ensure_models():
    tiny, qwen = os.path.join(BUILD, "tiny"), os.path.join(BUILD, "tiny_qwen")
    mk = [sys.executable, os.path.join(ROOT, "tools", "make_tiny_model.py")]
    if not os.path.exists(os.path.join(tiny, "model.safetensors")):
        run(mk + ["--out", tiny])
    if not os.path.exists(os.path.join(qwen, "model.safetensors")):
        run(mk + ["--out", qwen, "--qkv-bias", "--untied", "--dtype", "F32", "--merges-as-arrays", "--seed", "1",
                  "--pretok", "qwen2"])
    return tiny, qwen


def dump(binary, model, tokens, chunk, extra=(), env=None):
    out = os.path.join(BUILD, "test_logits.bin")
    run([binary, "-m", model, "--tokens", ",".join(map(str, tokens)), "--dump-logits", out, "--prefill-chunk",
         str(chunk), *extra], stderr=subprocess.DEVNULL, env=env)
    return np.fromfile(out, np.float32).reshape(-1, VOCAB)


class Report:
    def __init__(self):
        self.failed = 0

    def logits(self, name, ref, got, rel_tol):
        rel = float((np.abs(ref - got).max(axis=1) / np.abs(ref).max(axis=1)).max())
        agree = float((ref.argmax(1) == got.argmax(1)).mean())
        ok = rel <= rel_tol and agree == 1.0
        self.line(name, f"rel_err={rel:.2e} (tol {rel_tol:.0e}) argmax_agree={agree:.2f}", ok)

    def line(self, name, detail, ok):
        print(f"  {name:<50} {detail}  {'ok' if ok else 'FAIL'}", flush=True)
        self.failed += not ok


def test_cpu(r, tiny, qwen):
    import reference

    unit = os.path.join(BUILD, "test-common" + EXE)
    if os.path.exists(unit):
        print("[C++ unit tests: generation loop, sampler, chat turns]")
        res = subprocess.run([unit, tiny], capture_output=True, text=True)
        print(res.stdout + res.stderr, end="", flush=True)
        r.line("test-common", f"exit code {res.returncode}", res.returncode == 0)
    else:
        print(f"[C++ unit tests] {unit} not built, skipping")
    print("[cpu engine vs numpy reference]")
    for name, model in (("llama-style, bf16, tied", tiny), ("qwen-style, f32, bias, untied", qwen)):
        ref = reference.forward(model, TOKENS)
        r.logits(name, ref, dump(CPU, model, TOKENS, 1), 1e-5)
    print("[tokenizer: C++ (hand-written scanners) vs JS (regex engine)]")
    node = shutil.which("node")
    if not node:
        print("  node not found, skipping")
        return
    rng = random.Random(7)
    strings = FIXED + ["".join(rng.choice(POOL) for _ in range(rng.randint(1, 30))) for _ in range(150)]
    path = os.path.join(BUILD, "tok_strings.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(strings, f, ensure_ascii=False)
    for name, model in (("SmolLM2-style (Digits + ByteLevel)", tiny), ("Qwen2-style (Split regex)", qwen)):
        js = json.loads(run([node, os.path.join(ROOT, "tests", "js_tokenize.mjs"), os.path.join(model, "tokenizer.json"),
                             path], capture_output=True).stdout)
        bad = 0
        for s, ids in zip(strings, js):
            out = run([CPU, "-m", model, "--tokenize", s], capture_output=True, text=True).stdout.strip()
            cpp = [int(x) for x in out.split(",")] if out else []
            if cpp != ids:
                bad += 1
                if bad <= 3:
                    print(f"    mismatch {s!r}: c++ {cpp} js {ids}")
        r.line(name, f"{len(strings) - bad}/{len(strings)} identical", bad == 0)


def test_cuda(r, tiny, qwen, binary, quick):
    print(f"[cuda engine vs cpu engine] ({binary})")
    n_dec = 6 if quick else len(TOKENS)
    r.logits("fp16 decode (CUDA graph)", dump(CPU, tiny, TOKENS[:n_dec], 1), dump(binary, tiny, TOKENS[:n_dec], 1), 2e-3)
    r.logits("fp16 decode (no graph)", dump(CPU, tiny, TOKENS[:4], 1), dump(binary, tiny, TOKENS[:4], 1, ["--no-graph"]), 2e-3)
    r.logits("fp16 prefill, chunks of 5", dump(CPU, tiny, TOKENS, 5), dump(binary, tiny, TOKENS, 5), 2e-3)
    env = dict(os.environ, TWINFER_MAX_PREFILL="4")
    r.logits("fp16 prefill split internally into 4s", dump(CPU, tiny, TOKENS, 13), dump(binary, tiny, TOKENS, 13, env=env), 2e-3)
    r.logits("int8 prefill, chunks of 5", dump(CPU, tiny, TOKENS, 5), dump(binary, tiny, TOKENS, 5, ["--int8"]), 3e-2)
    r.logits("int8 decode", dump(CPU, tiny, TOKENS[:4], 1), dump(binary, tiny, TOKENS[:4], 1, ["--int8"]), 3e-2)
    r.logits("qwen-style bias + untied head, prefill 7", dump(CPU, qwen, TOKENS, 7), dump(binary, qwen, TOKENS, 7), 2e-3)
    args = ["-m", tiny, "--raw", "-p", "The quick brown fox", "-n", "8", "--temp", "0", "--quiet"]
    a = run([CPU, *args], capture_output=True).stdout
    b = run([binary, *args], capture_output=True).stdout
    r.line("greedy generation identical", repr(b.decode("utf-8", "replace").strip()), a == b)


def main():
    global BUILD, CPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--build-dir", default=BUILD)
    ap.add_argument("--cuda-bin")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not (a.cpu or a.cuda):
        a.cpu = True
    BUILD = os.path.abspath(a.build_dir)
    CPU = os.path.join(BUILD, "twinfer-cpu" + EXE)
    if not os.path.exists(CPU):
        raise SystemExit(f"build the CPU engine first (make cpu, or cmake --build): {CPU} not found")
    tiny, qwen = ensure_models()
    r = Report()
    if a.cpu:
        test_cpu(r, tiny, qwen)
    if a.cuda:
        test_cuda(r, tiny, qwen, a.cuda_bin or os.path.join(BUILD, "twinfer-cuda" + EXE), a.quick)
    if r.failed:
        raise SystemExit(f"{r.failed} test(s) FAILED")
    print("all tests passed")


if __name__ == "__main__":
    main()
