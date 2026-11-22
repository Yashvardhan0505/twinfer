#!/usr/bin/env python3
"""Rewrite CUDA sources so they compile as plain C++ against the CPU emulator headers.

  kernel<T, N><<<grid, block, smem, stream>>>(args)  ->  ::emu::launch(kernel<T, N>, grid, block, smem, stream)(args)
  extern __shared__ __align__(16) unsigned char buf[];  ->  unsigned char* buf = ::emu::dyn_smem();

Everything else is left untouched, so the emulated build runs the exact kernel code.
"""
import os
import re
import shutil
import sys

EXTERN_SHARED = re.compile(
    r"extern\s+__shared__\s+(?:__align__\(\s*\d+\s*\)\s+)?unsigned\s+char\s+(\w+)\s*\[\s*\]\s*;")
IDENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:")


def rewrite_launches(src: str) -> str:
    out, pos = [], 0
    while True:
        i = src.find("<<<", pos)
        if i < 0:
            out.append(src[pos:])
            return "".join(out)
        # Walk back over an optional template argument list and the (qualified) kernel name.
        j = i - 1
        while j >= 0 and src[j].isspace():
            j -= 1
        if src[j] == ">":
            depth = 0
            while j >= 0:
                if src[j] == ">":
                    depth += 1
                elif src[j] == "<":
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            j -= 1
        while j >= 0 and src[j] in IDENT:
            j -= 1
        start = j + 1
        end_cfg = src.find(">>>", i)
        if end_cfg < 0:
            raise SystemExit("unterminated <<< >>>")
        k = end_cfg + 3
        while src[k].isspace():
            k += 1
        if src[k] != "(":
            raise SystemExit("expected '(' after >>> near: " + src[start:k + 20])
        kernel = src[start:i].strip()
        cfg = src[i + 3:end_cfg].strip()
        out.append(src[pos:start])
        out.append(f"::emu::launch({kernel}, {cfg})(")
        pos = k + 1


def transform(text: str) -> str:
    text = EXTERN_SHARED.sub(r"unsigned char* \1 = ::emu::dyn_smem();", text)
    return rewrite_launches(text)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: preprocess.py REPO_ROOT OUT_DIR")
    root, out = sys.argv[1], sys.argv[2]
    if os.path.isdir(out):
        shutil.rmtree(out)
    for sub in ("common", "cuda"):
        for dirpath, _, files in os.walk(os.path.join(root, sub)):
            rel = os.path.relpath(dirpath, root)
            os.makedirs(os.path.join(out, rel), exist_ok=True)
            for f in files:
                src_path = os.path.join(dirpath, f)
                dst_path = os.path.join(out, rel, f)
                if f.endswith((".cu", ".cuh")):
                    with open(src_path, encoding="utf-8") as fh:
                        text = transform(fh.read())
                    with open(dst_path, "w", encoding="utf-8") as fh:
                        fh.write(text)
                else:
                    shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    main()
