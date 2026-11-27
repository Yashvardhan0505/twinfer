# twinfer

A small LLM inference engine written from scratch, with two GPU backends: CUDA and WebGPU.

It runs [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (and the
other SmolLM2 and Qwen2-style checkpoints, see [Limitations](#limitations--next-steps)) without PyTorch
or any ML library. Every kernel is hand-written, once in CUDA C++ and once in WGSL, so the same model
also runs in a browser tab.

[![ci](https://github.com/Yashvardhan0505/twinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashvardhan0505/twinfer/actions/workflows/ci.yml)

Browser demo: https://yashvardhan0505.github.io/twinfer/ (Chrome or Edge with WebGPU)

On a Tesla T4 it decodes SmolLM2-135M at 482 tok/s and prefills at 38,000 tok/s, with greedy output identical
to Hugging Face's; its largest decode GEMV reads at 71% of peak memory bandwidth, 1.3× faster than cuBLAS.
Details and what limits it: [Numbers](#numbers).

## Why

I wanted to see where the time actually goes when a model generates text, and the best way to
find out is to write every piece yourself and measure it. A 135M model is small enough to iterate
on quickly, and it makes one thing very obvious: generating a token is almost pure memory traffic.
Each token reads all 269 MB of fp16 weights once, so most of the work is moving bytes efficiently
and not wasting time on kernel launches.

Writing the same pipeline twice is also a good way to separate the ideas from the API. Warp
shuffles, tensor cores and CUDA graphs don't exist in WebGPU, so each of them needs a different
answer there.

## How it works

Generation has two phases, and they use different kernels:

```
prompt (prefill, T tokens)          next token (decode, 1 token)
--------------------------          ----------------------------
tensor-core GEMM (WMMA)             fused RMSNorm + GEMV, fp16 or int8
flash attention (WMMA)              split-KV attention over the KV cache
         |                          GEMV + SwiGLU / residual epilogues
         |                          whole step replayed as one CUDA graph
         +-------> fp16 KV cache <-------+
```

CUDA (`cuda/`):
- GEMV kernels for decode with RMSNorm, bias, residual add and SwiGLU fused in, so a transformer layer is 7 launches instead of 15+
- weight-only int8 with per-row scales, dequantized with a `__byte_perm` trick instead of int-to-float conversion instructions
- decode attention split across the context (flash-decoding style), where one block serves all the query heads that share a KV head
- a WMMA tensor-core GEMM and a FlashAttention-style kernel for the prompt
- the decode step is captured once as a CUDA graph and replayed for every token

WebGPU (`web/`) runs the same decode pipeline in WGSL. Weights are stored as packed fp16 pairs in
`u32` and unpacked in the shader, so it doesn't need the `shader-f16` feature. bf16 weights get
converted on the GPU at load time. There's a small chat page that pulls the model straight from
Hugging Face.

There's also a plain fp32 CPU implementation (`cpu/`) that everything else is checked against, and
a BPE tokenizer written twice (C++ and JS).

The reasoning behind each kernel is in [docs/DESIGN.md](docs/DESIGN.md).

## Numbers

SmolLM2-135M-Instruct, batch size 1, Tesla T4 on Colab (driver 580.82.07), commit 7aaa4e1. Each number is the
median of 3 runs after a warm-up round. To reproduce, open
[the notebook in Colab](https://colab.research.google.com/github/Yashvardhan0505/twinfer/blob/main/notebooks/colab_quickstart.ipynb)
or run `python3 tools/bench_table.py models/smollm2-135m`. Every run and the full kernel tables are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| backend | prefill tok/s (512-token prompt) | decode tok/s | ms/token |
|---|---|---|---|
| CUDA fp16 + graph | 38,142 | 482 | 2.08 |
| CUDA fp16, no graph | 37,669 | 470 | 2.13 |
| CUDA int8 + graph | 41,435 | 570 | 1.75 |
| CPU fp32 (Colab, 2 vCPUs) | 17 | 22.2 | 45.05 |

The output matches Hugging Face on the same model: fp16 logits are within 2.3e-3 of transformers' fp32 with the
same top token at every position, and the greedy text is identical (`tools/hf_compare.py`).

![Roofline of twinfer's kernels on a Tesla T4](docs/roofline-t4.png)

What the numbers say:

- **Decode is limited by the number of kernels, not by memory bandwidth.** Each token reads all 269 MB of weights,
  so the T4's 320 GB/s caps fp16 decode at about 1,190 tok/s, and twinfer reaches 482 (40%). The GEMV kernel isn't
  what holds it back: on the one big matrix, the 56.6 MB lm_head, it reads at 226 GB/s (71% of peak), 1.30× faster
  than cuBLAS. But the per-layer matrices are only 0.7–3.5 MB, and at that size a kernel spends most of its time on
  fixed start-up and wind-down costs, about 6 µs on the T4. cuBLAS pays them too: on those shapes twinfer's GEMVs,
  which also do the RMSNorm, SwiGLU or residual add, are within −7% to +16% of its plain GEMV, and decode attention
  takes 8.7 µs even at a 128-token context. A decode step runs 212 kernels, so these costs add up to roughly half
  of every token.
- **The same thing caps int8 at 1.18×.** int8 halves the bytes but not the kernel count. On lm_head, where the
  bytes do dominate, the int8 GEMV is even slower than fp16 (262 vs 250 µs, 109 GB/s): unpacking int8 costs more
  instructions than the memory it saves.
- **CUDA graphs only give 2.5% here.** A graph removes the CPU's launch overhead, but on the T4 the CPU already
  queues the 212 kernels faster than the GPU runs them.
- **Prefill is mostly GEMM** (about 85%, adding up the kernel timings), and the tensor-core GEMM runs at 45–67% of
  cuBLAS on the 512-token shapes. Matching cuBLAS would make prefill about 1.7× faster.
- **Colab's T4s vary.** In an earlier session on a different T4, decode measured 523 tok/s, and in this session
  one prefill run dropped to 21,293 tok/s (the T4 is a passively cooled 70 W card). That's why each number is a
  median.

## Running it

CUDA needs an NVIDIA GPU with compute capability 7.5 or newer on CUDA 13 (7.0 still works with
CUDA 12.x). RTX 50-series GPUs (sm_120) need CUDA 12.8 or newer. A free Colab T4 works too.

Download the model (on Windows, type `python` instead of `python3`):

```bash
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download as d; d('HuggingFaceTB/SmolLM2-135M-Instruct', local_dir='models/smollm2-135m', allow_patterns=['*.json','*.jinja','*.safetensors'])"
```

Linux:

```bash
make cuda ARCH=native                                       # or `make cpu` for the CPU engine only
./build/twinfer-cuda -m models/smollm2-135m -p "Explain warp divergence in one sentence."
./build/twinfer-cuda -m models/smollm2-135m -i              # chat
./build/twinfer-cuda -m models/smollm2-135m --bench 256     # decode speed; try --int8 and --no-graph too
./build/twinfer-cuda -m models/smollm2-135m --bench 1 --bench-prompt 512   # prefill speed
python3 tools/bench_table.py models/smollm2-135m            # all of the above: the table in Numbers
./build/bench-kernels --csv bench.csv                       # kernels vs cuBLAS
python3 tools/roofline.py bench.csv -o docs/roofline.png
```

Windows, in PowerShell, with Visual Studio 2022 or its Build Tools ("Desktop development with C++"),
the CUDA Toolkit and CMake 3.24 or newer:

```powershell
cmake -B build                               # add -DCMAKE_CUDA_ARCHITECTURES=120 if it can't detect the GPU
cmake --build build --config Release
.\build\twinfer-cuda.exe -m models\smollm2-135m -p "Explain warp divergence in one sentence."
.\build\twinfer-cuda.exe -m models\smollm2-135m --bench 256
python tools\bench_table.py models\smollm2-135m
.\build\bench-kernels.exe --csv bench.csv
```

If `cmake` isn't found, it is only on PATH inside the Developer PowerShell for VS; `winget install
Kitware.CMake` fixes that. If CMake reports "No CUDA toolset found", reinstall the CUDA Toolkit with
its Visual Studio Integration component. CMake also works on Linux, and without CUDA (or with
`-DTWINFER_CUDA=OFF`) it builds just `twinfer-cpu`.

CPU only: the same commands with `twinfer-cpu`.

Browser: open the demo link, or run `python3 -m http.server 8000` in the repo and go to
http://localhost:8000/web/.

Useful flags: `--temp`, `--top-p`, `-n`, `--ctx`, `--system` (defaults to the model's own, from its chat
template), `--raw` (no chat template; with `-i` it keeps continuing your text).

## Testing

Most of the tests run without a GPU:

- `make test` (CMake: `ctest --test-dir build -C Release -LE gpu`): CPU engine vs an independent numpy implementation, the C++ tokenizer vs the JS one, and unit tests for the generation loop, sampler and chat turns (C++ and JS). Needs `pip install numpy`; the JS parts also need Node.
- `make test-emu` (Linux): runs the real CUDA kernels on the CPU through a small emulator (`tests/cuda_emu/`). It's slow, but it catches indexing and sync bugs, and CI runs it on every push.
- `python3 tests/web_test.py`: the WebGPU engine in headless Chrome (software adapter) vs the C++ engine, plus a conversation typed into the chat page vs `twinfer-cpu -i`
- `make test-cuda` (CMake: `ctest --test-dir build -C Release -L gpu`): kernel unit tests and end-to-end checks on an actual GPU
- `python3 tools/hf_compare.py models/smollm2-135m --cuda`: logits and greedy output against Hugging Face transformers, and both tokenizers against the model's `tokenizer.json` (228/228 strings). It compares with the file itself because transformers 5 rebuilds SmolLM2's tokenizer without its digit-splitting step, and so splits numbers like ½ and ٣ differently.

The kernel tests fill padding and unused KV-cache rows with NaN on purpose, so an out-of-bounds
read shows up as NaN instead of quietly passing.

## Layout

```
common/          safetensors loading, config, tokenizer, sampler, CLI
cpu/             fp32 reference engine
cuda/kernels/    the CUDA kernels
cuda/            engine, CLI, kernel tests, benchmarks
web/             WebGPU demo (src/engine.js, src/shaders/*.wgsl)
tests/           test scripts and the CUDA emulator
tools/           tiny test model generator, numpy reference, HF comparison, benchmark table, roofline plot
docs/DESIGN.md   design notes
docs/BENCHMARKS.md  raw benchmark output behind the README's numbers
```

## Limitations / next steps

- Decode at this model size is limited by its 212 kernels per token (see [Numbers](#numbers)). Fusing each layer into fewer kernels, or running the whole step as one persistent kernel, would help more than faster memory access.
- The prefill GEMM is a straightforward WMMA kernel at 32–69% of cuBLAS on a T4. Next is a double-buffered `cp.async` pipeline with bigger tiles, then `mma.sync`.
- Flash attention uses WMMA plus a shared-memory trick for the row rescaling. An FA-2 style version with `mma.sync` would keep that in registers.
- The int8 GEMV spends more instructions unpacking weights than the halved bytes save (on lm_head it's slower than fp16), and int4 would halve the bytes again. Both matter more on bigger models, where decode is bandwidth-bound.
- The WebGPU side doesn't have split-KV attention, batched prefill or int8 yet.
- Models: Llama-architecture checkpoints with a byte-level BPE tokenizer. SmolLM2 matches Hugging Face
  (see [Numbers](#numbers)); Qwen2/Qwen2.5-style models (q/k/v bias, untied head, the Qwen2 pre-tokenizer) are only
  tested on a tiny random model so far. Not supported yet: SentencePiece-style tokenizers (Llama 2,
  Mistral, TinyLlama), RoPE scaling (so no Llama 3.x) and sliding-window attention. The C++ tokenizer
  doesn't apply Qwen2.5's NFC normalization yet (it prints a warning; the browser one does).

## Credits

SmolLM2 by Hugging Face. FlashAttention (Dao et al.) and Flash-Decoding for the attention algorithms.
Karpathy's llama2.c and llm.c were a big inspiration for doing the whole thing from scratch.

MIT license.
