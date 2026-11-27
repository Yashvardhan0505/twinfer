#!/usr/bin/env python3
"""The README's throughput table: prefill and decode speed for each backend, the median of a few runs.

    python3 tools/bench_table.py models/smollm2-135m
    python tools\\bench_table.py models\\smollm2-135m        (Windows)

Runs whichever of twinfer-cuda and twinfer-cpu are built, and prints Markdown for the README's Numbers
section: a line saying what it ran on (GPU, driver, commit), then the table. The WebGPU row is left
empty; fill it in from the browser demo, which shows decode tok/s after each reply. Every run's numbers
go to stderr, so an outlier is easy to spot.
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
    ap.add_argument("--runs", type=int, default=3,
                    help="runs per number, after a warm-up round on the GPU; the median is reported (default 3)")
    ap.add_argument("--prompt", type=int, default=512, help="prompt length for the prefill number (default 512)")
    ap.add_argument("--decode", type=int, default=256, help="tokens to decode on the GPU (default 256; CPU: a quarter)")
    a = ap.parse_args()

    gpu_rows, cpu_rows = [], []  # (label, command, tokens to decode)
    for label, name, flags in ROWS:
        exe = os.path.join(a.build_dir, (name or "") + EXE)
        if name and not os.path.exists(exe):
            print(f"skipping {label}: {exe} isn't built", file=sys.stderr)
        elif name == "twinfer-cuda":
            gpu_rows.append((label, [exe, "-m", a.model, *flags], a.decode))
        elif name:
            cpu_rows.append((label, [exe, "-m", a.model, *flags], max(1, a.decode // 4)))

    def measure(label, cmd, n_dec, what):
        print(f"{what}: {label} ...", file=sys.stderr, flush=True)
        return (bench(cmd + ["--bench", "1", "--bench-prompt", str(a.prompt)])["prefill_tok_s"],
                bench(cmd + ["--bench", str(n_dec)])["decode_tok_s"])

    # The GPU rows take turns, one run each per round, after a warm-up round that isn't counted. A GPU speeds its
    # clock up under load and slows it down when hot, and taking turns spreads that over every row instead of
    # whichever runs first or last. The CPU rows run afterwards, so the GPU doesn't sit idle between its rows.
    runs = {}
    for r in range(a.runs + 1):
        for label, cmd, n_dec in gpu_rows:
            result = measure(label, cmd, n_dec, f"run {r}/{a.runs}" if r else "warm-up")
            if r:
                runs.setdefault(label, []).append(result)
    for label, cmd, n_dec in cpu_rows:
        runs[label] = [measure(label, cmd, n_dec, f"run {r}/{a.runs}") for r in range(1, a.runs + 1)]

    lines = []
    for label, _, _ in ROWS:
        got = runs.get(label)
        if not got:
            lines.append(f"| {label} | | | |")
            continue
        print(f"{label}: prefill {', '.join(num(p) for p, _ in got)} | decode {', '.join(num(d) for _, d in got)}",
              file=sys.stderr)
        pre, dec = (statistics.median(x) for x in zip(*got))
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
