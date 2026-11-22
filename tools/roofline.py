#!/usr/bin/env python3
"""Roofline plot from `bench-kernels --csv results.csv`.

    python3 tools/roofline.py results.csv -o docs/roofline.png

Memory roof = measured peak DRAM bandwidth (from the device attributes, or --peak-gbps),
compute roof = best cuBLAS fp16 tensor-core rate measured in the same run (or --peak-tflops).
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="roofline.png")
    ap.add_argument("--peak-gbps", type=float)
    ap.add_argument("--peak-tflops", type=float)
    a = ap.parse_args()

    rows, meta = [], {}
    with open(a.csv) as f:
        for line in f:
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition("=")
                meta[k] = v
    with open(a.csv) as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            rows.append(r)
    bw = a.peak_gbps or float(meta.get("peak_gbps", 0)) or 300.0
    peak = (a.peak_tflops or float(meta.get("cublas_tflops", 0)) or 50.0) * 1e3  # GFLOP/s

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ai = np.logspace(-1, 3, 200)
    ax.loglog(ai, np.minimum(bw * ai, peak), "k-", lw=2, label=f"roof: {bw:.0f} GB/s, {peak / 1e3:.0f} TFLOP/s")
    groups = {}
    for r in rows:
        us, by, fl = float(r["us"]), float(r["bytes"]), float(r["flops"])
        if us <= 0 or by <= 0 or fl <= 0:
            continue
        kind = r["kernel"].split("_")[0] if not r["kernel"].startswith("cublas") else "cublas"
        if r["kernel"].startswith("gemv_int8"):
            kind = "gemv int8"
        elif r["kernel"].startswith("gemv_fp16"):
            kind = "gemv fp16"
        elif r["kernel"].startswith("gemm"):
            kind = "gemm wmma"
        elif r["kernel"].startswith("attn"):
            kind = "decode attention"
        elif r["kernel"].startswith("flash"):
            kind = "flash attention"
        groups.setdefault(kind, []).append((fl / by, fl / us / 1e3))
    for kind, pts in groups.items():
        x, y = zip(*pts)
        ax.scatter(x, y, s=40, label=kind, alpha=0.85)
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title(f"twinfer kernels on {meta.get('gpu', 'GPU')}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
