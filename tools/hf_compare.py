#!/usr/bin/env python3
"""End-to-end check against Hugging Face transformers on a real model (needs torch +
transformers, e.g. on Colab):

    python3 tools/hf_compare.py models/smollm2-135m [--cuda]

1. tokenizer   HF AutoTokenizer vs the C++ tokenizer vs the JS tokenizer (Node), many strings
2. template    HF apply_chat_template vs twinfer's ChatML string
3. logits      HF fp32 vs twinfer CPU (fp32) [and CUDA fp16 / int8], every position of a prompt
4. generation  greedy text, HF generate() vs twinfer
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
BUILD = os.path.join(ROOT, "build")
sys.path.insert(0, os.path.join(ROOT, "tests"))
from run_tests import FIXED, POOL  # noqa: E402

SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
LONG = ("Graphics processing units execute thousands of threads in parallel. A CUDA kernel is launched as a grid "
        "of thread blocks; each block runs on one streaming multiprocessor and can share data through fast "
        "on-chip shared memory. Memory coalescing, occupancy and arithmetic intensity decide how fast it runs. ") * 3


def cpp_ids(binary, model, s):
    out = subprocess.run([binary, "-m", model, "--tokenize", s], capture_output=True, text=True, check=True).stdout.strip()
    return [int(x) for x in out.split(",")] if out else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cuda", action="store_true", help="also check build/twinfer-cuda")
    ap.add_argument("--gen", type=int, default=48)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    exe = ".exe" if os.name == "nt" else ""
    cpu = os.path.join(BUILD, "twinfer-cpu" + exe)
    cuda = os.path.join(BUILD, "twinfer-cuda" + exe)
    tok = AutoTokenizer.from_pretrained(a.model)
    failures = 0

    # 1. tokenizer
    rng = random.Random(3)
    strings = FIXED + [LONG] + ["".join(rng.choice(POOL) for _ in range(rng.randint(1, 40))) for _ in range(200)]
    hf = [tok(s, add_special_tokens=False)["input_ids"] for s in strings]
    bad_cpp = sum(cpp_ids(cpu, a.model, s) != ids for s, ids in zip(strings, hf))
    print(f"tokenizer  C++ vs HF: {len(strings) - bad_cpp}/{len(strings)} identical")
    failures += bad_cpp > 0
    node = shutil.which("node")
    if node:
        path = os.path.join(BUILD, "hf_strings.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(strings, f, ensure_ascii=False)
        js = json.loads(subprocess.run([node, os.path.join(ROOT, "tests", "js_tokenize.mjs"),
                                        os.path.join(a.model, "tokenizer.json"), path],
                                       capture_output=True, check=True).stdout)
        bad_js = sum(x != y for x, y in zip(js, hf))
        print(f"tokenizer  JS  vs HF: {len(strings) - bad_js}/{len(strings)} identical")
        failures += bad_js > 0

    # 2. chat template
    msgs = [{"role": "user", "content": "What is shared memory?"}]
    ours = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nWhat is shared memory?<|im_end|>\n<|im_start|>assistant\n"
    theirs = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"chat template identical: {ours == theirs}")
    if ours != theirs:
        print("  HF  :", repr(theirs))
        print("  ours:", repr(ours))
        failures += 1

    # 3. logits
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()
    ids = tok(theirs + "Shared memory is", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0].float().numpy()

    def check(binary, extra, label, tol, chunk=1):
        out = os.path.join(BUILD, "hf_logits.bin")
        subprocess.run([binary, "-m", a.model, "--tokens", ",".join(map(str, ids)), "--dump-logits", out,
                        "--prefill-chunk", str(chunk), *extra], check=True, stderr=subprocess.DEVNULL)
        got = np.fromfile(out, np.float32).reshape(-1, ref.shape[1])
        # one logits row per chunk: the chunk's last position
        r = ref[[min(i + chunk, len(ids)) - 1 for i in range(0, len(ids), chunk)]]
        rel = float((np.abs(r - got).max(axis=1) / np.abs(r).max(axis=1)).max())
        top1 = float((r.argmax(1) == got.argmax(1)).mean())
        ok = rel < tol and top1 >= 0.95
        print(f"logits     {label:<28} max rel err {rel:.2e}  top-1 agreement {top1:.3f}  {'ok' if ok else 'CHECK'}")
        return ok

    failures += not check(cpu, [], "CPU fp32 (decode)", 1e-3)
    if a.cuda:
        failures += not check(cuda, [], "CUDA fp16 (decode + graph)", 2e-2)
        failures += not check(cuda, [], "CUDA fp16 (prefill chunks)", 2e-2, chunk=len(ids))
        failures += not check(cuda, ["--int8"], "CUDA int8 (decode)", 8e-2)

    # 4. greedy generation
    with torch.no_grad():
        g = model.generate(torch.tensor([tok(theirs, add_special_tokens=False)["input_ids"]]), max_new_tokens=a.gen,
                           do_sample=False, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    hf_text = tok.decode(g[0, len(tok(theirs, add_special_tokens=False)["input_ids"]):], skip_special_tokens=True)
    print(f"\nHF greedy:      {hf_text!r}")
    for binary, label in [(cpu, "twinfer CPU")] + ([(cuda, "twinfer CUDA")] if a.cuda else []):
        txt = subprocess.run([binary, "-m", a.model, "-p", "What is shared memory?", "--temp", "0", "-n", str(a.gen),
                              "--quiet"], capture_output=True, text=True, check=True).stdout.strip()
        print(f"{label + ':':<15} {txt!r}  {'(identical)' if txt == hf_text.strip() else '(differs; fp16 may diverge late)'}")

    if failures:
        raise SystemExit(f"{failures} check(s) failed")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
