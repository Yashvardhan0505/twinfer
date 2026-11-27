#!/usr/bin/env python3
"""The README's throughput table: prefill and decode speed for each backend, the median of a few runs.

    python3 tools/bench_table.py models/smollm2-135m
    python tools\\bench_table.py models\\smollm2-135m        (Windows)

Runs whichever of twinfer-cuda and twinfer-cpu are built, and prints Markdown for the README's Numbers
section: a line saying what it ran on (GPU, driver, commit), then the table. The WebGPU row is left
empty; fill it in from the browser demo, which shows decode tok/s after each reply.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = ".exe" if os.name == "nt" else ""
ROWS = [  # label, binary (None = not scripted), extra flags; in the README's order
    ("CUDA fp16 + graph", "twinfer-cuda", []),
    ("CUDA fp16, no graph", "twinfer-cuda", ["--no-graph"]),
    ("CUDA int8 + graph", "twinfer-cuda", ["--int8"]),
    ("WebGPU (Chrome)", None, []),
    ("CPU fp32", "twinfer-cpu", []),
]


def bench(cmd):
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        sys.exit(f"failed (exit code {res.returncode}): {' '.join(cmd)}\n{res.stderr}")
    return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", res.stdout)}


def num(x):  # whole numbers from 100 up, otherwise 3 significant digits (a slow CPU doesn't show as 0)
    return f"{x:.0f}" if x >= 100 else f"{x:.3g}"


def output_of(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except OSError:  # not installed
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model directory, e.g. models/smollm2-135m")
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build"))
    ap.add_argument("--runs", type=int, default=3, help="runs per number; the median is reported (default 3)")
    ap.add_argument("--prompt", type=int, default=512, help="prompt length for the prefill number (default 512)")
    ap.add_argument("--decode", type=int, default=256, help="tokens to decode on the GPU (default 256; CPU: a quarter)")
    a = ap.parse_args()

    lines = []
    for label, name, flags in ROWS:
        exe = os.path.join(a.build_dir, (name or "") + EXE)
        if name is None or not os.path.exists(exe):
            if name:
                print(f"skipping {label}: {exe} isn't built", file=sys.stderr)
            lines.append(f"| {label} | | | |")
            continue
        print(f"running {label} ...", file=sys.stderr, flush=True)
        cmd = [exe, "-m", a.model, *flags]
        n_dec = a.decode if name == "twinfer-cuda" else max(1, a.decode // 4)
        pre = statistics.median(bench(cmd + ["--bench", "1", "--bench-prompt", str(a.prompt)])["prefill_tok_s"]
                                for _ in range(a.runs))
        dec = statistics.median(bench(cmd + ["--bench", str(n_dec)])["decode_tok_s"] for _ in range(a.runs))
        lines.append(f"| {label} | {num(pre)} | {num(dec)} | {1000 / dec if dec else 0:.2f} |")

    about = [os.path.basename(os.path.normpath(a.model)), "batch size 1"]
    gpu = output_of(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).splitlines()
    if gpu:
        name, _, driver = gpu[0].partition(", ")
        about.append(f"GPU: {name} (driver {driver})")
    commit = output_of(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"])
    if commit:
        about.append(f"commit {commit}")
    about.append(f"median of {a.runs} runs")
    print(", ".join(about) + "\n")
    print(f"| backend | prefill tok/s ({a.prompt}-token prompt) | decode tok/s | ms/token |")
    print("|---|---|---|---|")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
