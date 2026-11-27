#!/usr/bin/env python3
"""Build the CUDA code against the CPU emulator, then run the kernel tests and the
end-to-end checks. Slow (every CUDA thread is an OS thread) but needs no GPU."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build")
EMU = os.path.join(BUILD, "emu")
CXX = os.environ.get("CXX", "g++")
FLAGS = ["-std=c++20", "-O2", "-pthread", "-fno-strict-aliasing", "-Wall", "-Wextra", "-Wno-unknown-pragmas",
         "-I" + os.path.join(ROOT, "tests", "cuda_emu")]


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def build():
    src = os.path.join(EMU, "src")
    run([sys.executable, os.path.join(ROOT, "tests", "cuda_emu", "preprocess.py"), ROOT, src])
    common = [os.path.join(ROOT, "common", f) for f in ("model_files.cpp", "tokenizer.cpp", "platform.cpp")]
    run([CXX, *FLAGS, "-o", os.path.join(EMU, "test-kernels"), "-x", "c++", os.path.join(src, "cuda", "test_kernels.cu")])
    run([CXX, *FLAGS, "-o", os.path.join(EMU, "twinfer-cuda"), "-x", "c++", os.path.join(src, "cuda", "main.cu"),
         os.path.join(src, "cuda", "cuda_model.cu"), "-x", "none", *common])
    run([CXX, *FLAGS, "-o", os.path.join(EMU, "bench-kernels"), "-x", "c++", os.path.join(src, "cuda", "bench_kernels.cu")])


def main():
    build()
    print("\n[kernel unit tests]")
    run([os.path.join(EMU, "test-kernels"), "--quick"])
    print("\n[end-to-end]")
    run([sys.executable, os.path.join(ROOT, "tests", "run_tests.py"), "--cuda", "--quick", "--cuda-bin",
         os.path.join(EMU, "twinfer-cuda")])
    print("\nall emulated CUDA tests passed")


if __name__ == "__main__":
    main()
