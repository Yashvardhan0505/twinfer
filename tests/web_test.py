#!/usr/bin/env python3
"""WebGPU backend + JS tokenizer vs the C++ engine, in headless Chrome (Playwright).

Works without a GPU: Chrome falls back to its SwiftShader software WebGPU adapter.
  python3 tests/web_test.py            (needs build/twinfer-cpu and `pip install playwright`)
Set CHROME_PATH to use a specific Chrome/Chromium binary.
"""
import functools
import http.server
import json
import os
import random
import shutil
import subprocess
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
CPU = os.path.join(BUILD, "twinfer-cpu" + (".exe" if os.name == "nt" else ""))

FIXED = [
    "Hello world", "  leading spaces", "trailing   ", "tabs\tand\nnewlines\n\n", "it's they're we'll I'M 'S",
    "numbers 12345 and 3.14", "emoji 😀🚀 mixed", "नमस्ते दुनिया", "你好，世界！", "Ünïcödé façade",
    "code: for (int i=0;i<n;++i) { y[i] += a*x[i]; }", "<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n",
    "a b", "x　y", "\r\n windows line\r\n", "''quoted'' 's", " ", "", "multiple\n\n\n\nnewlines   \n  x",
    "日本語のテキスト123", "Ⅻ roman ½ fraction ² super", "combining é accent", "ﬁ ligature", "Привет мир",
    "مرحبا بالعالم ١٢٣", "zero​width", "BOM﻿here", "NEL\u0085line", "#$%&*!!?? ... ---", "   \n\n   ",
]
POOL = list("abcXYZ019 \t\n\r.,;:'!?-_()[]{}\"") + ["é", "ß", "ñ", "न", "म", "ि", "ं", "你", "。", "😀", " ",
                                                     " ", "½", "²", "٣", "́", "—", "’", "　", "Ω"]


def random_strings(n, seed=7):
    rng = random.Random(seed)
    return ["".join(rng.choice(POOL) for _ in range(rng.randint(1, 30))) for _ in range(n)]


def cpp_tokenize(model, s):
    out = subprocess.run([CPU, "-m", model, "--tokenize", s], capture_output=True, text=True, check=True).stdout.strip()
    return [int(x) for x in out.split(",")] if out else []


CHAT = ["GPUs run thousands of threads", "def matmul(a, b):", "Shared memory is"]
CHAT_MAX_NEW = 4  # small, so every reply gets cut off and the next turn has to carry its last token


def cpp_chat(model):
    """Replies of `twinfer-cpu -i`, greedy, one string per turn (decoded like the browser does)."""
    out = subprocess.run([CPU, "-m", model, "-i", "-n", str(CHAT_MAX_NEW), "--temp", "0", "--quiet"],
                         input=("\n".join(CHAT) + "\n").encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         check=True).stdout
    # stderr prints "\n> " before each turn; stdout and stderr share the pipe, so that splits the replies
    return [r.decode("utf-8", "replace") for r in out.split(b"\n> ")[1:-1]]


def ui_chat(page, port, model_url):
    """The same conversation typed into the real chat page (web/index.html)."""
    page.goto(f"http://127.0.0.1:{port}/web/index.html?base={model_url}")
    page.click("#load-btn")
    idle = "!document.querySelector('#send-btn').disabled && !document.querySelector('.msg.streaming')"
    page.wait_for_function(idle, timeout=600_000)
    page.fill("#temp", "0")
    page.fill("#max-new", str(CHAT_MAX_NEW))
    replies = []
    for m in CHAT:
        page.fill("#input", m)
        page.click("#send-btn")
        page.wait_for_function(idle, timeout=600_000)
        replies.append(page.evaluate("[...document.querySelectorAll('.msg.assistant')].pop().textContent"))
    return replies


def main():
    from playwright.sync_api import sync_playwright

    if not os.path.exists(CPU):
        raise SystemExit("build the CPU engine first: make cpu")
    tiny = os.path.join(BUILD, "tiny")
    qwen = os.path.join(BUILD, "tiny_qwen")
    if not os.path.exists(os.path.join(tiny, "model.safetensors")):
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_model.py"), "--out", tiny], check=True)
    if not os.path.exists(os.path.join(qwen, "model.safetensors")):
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_model.py"), "--out", qwen, "--qkv-bias",
                        "--untied", "--dtype", "F32", "--merges-as-arrays", "--seed", "1", "--pretok", "qwen2"],
                       check=True)
    strings = FIXED + random_strings(60)
    tokens = cpp_tokenize(tiny, "The quick brown fox jumps over 12 lazy dogs. GPU kernels!")
    gen_prompt, gen_tokens = "Shared memory is", 10
    models = [{"name": "tiny", "path": "/build/tiny/", "engine": True},
              {"name": "tiny_qwen", "path": "/build/tiny_qwen/", "engine": True}]
    with open(os.path.join(BUILD, "web_test_input.json"), "w", encoding="utf-8") as f:
        json.dump({"strings": strings, "tokens": tokens, "models": models, "gen_prompt": gen_prompt,
                   "gen_tokens": gen_tokens}, f, ensure_ascii=False)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=ROOT)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    chrome = os.environ.get("CHROME_PATH") or shutil.which("google-chrome") or shutil.which("chromium")
    flags = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader"]
    if os.environ.get("TWINFER_REAL_GPU"):
        flags = ["--enable-unsafe-webgpu"]
    with sync_playwright() as p:
        if chrome:
            browser = p.chromium.launch(executable_path=chrome, headless=True, args=flags)
        else:  # Playwright's full Chromium ("new" headless mode has WebGPU; the headless shell may not)
            browser = p.chromium.launch(channel="chromium", headless=True, args=flags)
        page = browser.new_page()
        page.on("console", lambda m: print("  [browser]", m.text) if m.type == "error" else None)
        page.goto(f"http://127.0.0.1:{port}/web/test.html?input=/build/web_test_input.json")
        page.wait_for_function("window.__result !== undefined", timeout=600_000)
        res = page.evaluate("window.__result")
        chat_page = browser.new_page()
        chat_page.on("console", lambda m: print("  [browser]", m.text) if m.type == "error" else None)
        js_chat = ui_chat(chat_page, port, "/build/tiny_qwen/")
        browser.close()
    httpd.shutdown()

    if "error" in res:
        raise SystemExit("browser error:\n" + res["error"])
    print(f"WebGPU adapter: {res.get('adapter')}")
    ok = True
    for m in models:
        name, path = m["name"], os.path.join(ROOT, m["path"].strip("/"))
        r = res[name]
        mism = [s for s, ids in zip(strings, r["encodes"]) if ids != cpp_tokenize(path, s)]
        rt = all(r["roundtrip"])
        print(f"[{name}] tokenizer: {len(strings) - len(mism)}/{len(strings)} strings identical to C++, "
              f"lossless round-trip: {rt}")
        for s in mism[:5]:
            print(f"    mismatch: {s!r}\n      js : {r['encodes'][strings.index(s)]}\n      c++: {cpp_tokenize(path, s)}")
        ok &= not mism and rt

        ref_file = os.path.join(BUILD, f"web_ref_{name}.bin")
        subprocess.run([CPU, "-m", path, "--tokens", ",".join(map(str, tokens)), "--dump-logits", ref_file],
                       check=True, stderr=subprocess.DEVNULL)
        ref = np.fromfile(ref_file, np.float32).reshape(len(tokens), -1)
        got = np.array(r["logits"], np.float32)
        rel = (np.abs(ref - got).max(axis=1) / np.abs(ref).max(axis=1)).max()
        agree = (ref.argmax(1) == got.argmax(1)).mean()
        good = rel < 3e-3 and agree == 1.0
        print(f"[{name}] WebGPU logits vs C++ fp32: rel_err={rel:.2e} argmax_agree={agree:.2f}  {'ok' if good else 'FAIL'}")
        ok &= good
        good = r["batch_argmax"] == int(ref[-1].argmax())
        print(f"[{name}] multi-token forward + GPU argmax: {r['batch_argmax']} vs {int(ref[-1].argmax())}  "
              f"{'ok' if good else 'FAIL'}")
        ok &= good
        # Compare raw bytes: a random model happily emits partial UTF-8 sequences.
        cpp_out = subprocess.run([CPU, "-m", path, "--raw", "-p", gen_prompt, "-n", str(gen_tokens), "--temp", "0",
                                  "--quiet"], capture_output=True, check=True).stdout
        cpp_out = cpp_out[:-1] if cpp_out.endswith(b"\n") else cpp_out
        good = bytes(r["gen_bytes"]) == cpp_out
        print(f"[{name}] greedy generation: js={r['gen_text']!r} identical bytes to C++: {good}  {'ok' if good else 'FAIL'}")
        ok &= good
    want = cpp_chat(qwen)
    good = js_chat == want
    print(f"[chat page] {len(CHAT)} turns cut off at {CHAT_MAX_NEW} tokens, same replies as `twinfer-cpu -i`: "
          f"{good}  {'ok' if good else 'FAIL'}")
    if not good:
        print(f"    page: {js_chat!r}\n    c++ : {want!r}")
    ok &= good
    if not ok:
        raise SystemExit("WebGPU tests FAILED")
    print("all WebGPU tests passed")


if __name__ == "__main__":
    main()
