# Design notes

Why the kernels look the way they do. The speed limits below are calculated from specs, not
measured. Measured numbers go in the README once I have them.

## The workload

SmolLM2-135M: 30 layers, hidden size 576, 9 query heads and 3 KV heads (GQA, 3 query heads per KV
head), head_dim 64, FFN size 1536 with SwiGLU, 49,152 vocab with tied embeddings. That's 134.5M
parameters, or 269 MB in fp16.

Decoding one token uses every weight exactly once, for a single multiply-add, so the arithmetic
intensity is about 1 FLOP per byte. A T4's ridge point is around 65 TFLOP/s / 320 GB/s = 200 FLOP
per byte, which means decode is completely memory-bound and the speed limit is just bytes divided
by bandwidth:

- fp16 weights: 269 MB per token, at most ~1,190 tokens/s on a T4
- int8 weights (plus fp32 row scales): 135 MB, at most ~2,370 tokens/s
- the KV cache adds 23 KB per token of context, which doesn't matter until the context gets long

Prefill is different. Each weight gets reused for every prompt token, so it's a real GEMM and turns
compute-bound once the prompt is a few hundred tokens long. That's where tensor cores come in.

Something that's easy to miss with a model this small: the matrices are tiny. The biggest one in a
layer is 3072 x 576 (3.5 MB, about 11 us of DRAM time on a T4) and most are much smaller. A decode
step ends up being ~200 small kernels, and launching a kernel from the host costs a few microseconds,
which is about the same as the work itself. So the two big levers are fusing kernels and replaying
the whole step as a CUDA graph.

## Decode step

Per token:

1. `embed_kernel`: look up the token embedding (dequantize if int8)
2. for each of the 30 layers:
   - `gemv_kernel` with RMSNorm fused in: Q, K and V as one matrix, plus bias if the model has one
   - `rope_kv_kernel`: RoPE on q and k, append k and v to the fp16 KV cache
   - `attn_decode_kernel` + `attn_combine_kernel`: attention over the cache
   - `gemv_kernel`: output projection, added straight into the residual
   - `gemv_kernel` with RMSNorm fused in: gate and up projections, writes silu(gate) * up
   - `gemv_kernel`: down projection, added into the residual
3. `gemv_kernel` with RMSNorm fused in: final norm + LM head

That's 7 launches per layer, 212 per token. Done naively (separate norm, q, k, v, rope, cache write,
scores, softmax, PV, o, add, norm, gate, up, silu*mul, down, add) it would be 15+ per layer.

## GEMV

`cuda/kernels/gemv.cuh`

The obvious mapping is one thread per output row, but then neighbouring threads read from different
rows and nothing coalesces. Instead each row gets a whole warp: lane i reads 16-byte chunks i, i+32,
i+64, ... of the row, so every warp-wide load is 512 contiguous bytes. With 8 warps per block there
are 8 rows in flight per block.

The input vector is copied into shared memory once per block as fp16, and lanes read it back with
16-byte loads. Consecutive lanes hit consecutive addresses, so there are no bank conflicts.

RMSNorm is folded into that copy. Every block recomputes the sum of squares of the 576-float residual
(2.3 KB, which sits in L2) instead of running a separate norm kernel and writing the result back out.

The epilogue does whatever the next op needs:
- bias add
- residual add (`x[row] += acc`; each row belongs to exactly one warp, so no atomics)
- SwiGLU: gate and up are stored as one [2F, D] matrix, the warp computes rows i and F+i and writes silu(g) * u

For int8 weights the per-row scale is applied once in the epilogue. The dequantization avoids the
int-to-float conversion instructions, which run at a fraction of the FADD rate. `q ^ 0x80808080` turns
each signed byte b into b + 128, then `__byte_perm(q, 0x4B000000, 0x754k)` builds a float whose bits
are 0x4B0000xx, which is exactly 2^23 + (b + 128). Subtracting 2^23 + 128 gives b back: one PRMT and
one FADD per weight. The GEMM uses the fp16 version of the same idea (0x6400 | byte is 1024 + byte).

## Decode attention

`cuda/kernels/attention_decode.cuh`

With only 3 KV heads, one block per head would keep 3 SMs busy and leave the rest idle. So the
context is split into 128-key chunks and every (KV head, chunk) pair is its own block, which is the
Flash-Decoding idea. Each block writes a partial result (max, sum, unnormalized output) and a second
small kernel merges them:

```
out = sum_c 2^(m_c - M) * o_c  /  sum_c 2^(m_c - M) * l_c
```

A block also handles all 3 query heads that share its KV head, so each K/V row is read from memory
once instead of three times.

Inside a block:
- scores: one thread per key, 16-byte loads, dot product with the 3 queries kept in shared memory.
  log2(e)/sqrt(d) is folded into the scale so the softmax can use exp2f.
- softmax max/sum: one warp per query head
- PV: warps stride over the keys and each lane owns head_dim/32 dimensions, so every V row is one coalesced read

The grid always has max_seq/128 chunks and blocks past the current length return right away. The
current position is read from device memory. That keeps the launch shape fixed, which is what lets
the decode step be captured in a CUDA graph.

## Prefill GEMM

`cuda/kernels/gemm_wmma.cuh`

64x64x32 block tiles, 4 warps arranged 2x2, each warp computing a 32x32 tile as 2x2 WMMA 16x16x16
fragments (fp16 inputs, fp32 accumulation).

Shared memory rows are padded to 40 halves (80 bytes). That keeps fragment start addresses 32-byte
aligned, which `load_matrix_sync` needs, and moves consecutive rows onto different banks.

The weights are stored [N, K] row-major, which is the same thing as a K x N column-major matrix, so
they load directly as `matrix_b, col_major` without a transpose.

The epilogue goes through shared memory: the fragments get stored, then threads walk the tile row by
row so the global writes coalesce. The int8 scale, bias and residual add happen there.

This one is deliberately simple and it's well behind cuBLAS. Things to try next, in order: 128x128
tiles, a double-buffered `cp.async` pipeline (sm_80+) to overlap loads with math, `mma.sync` +
`ldmatrix` with a swizzled shared-memory layout, and split-K for small M.

## Flash attention (prefill)

`cuda/kernels/flash_attention.cuh`

A block handles 64 query rows of one head (4 warps, 16 rows each) and walks the keys in 32-key tiles
staged in shared memory, so the full T x T score matrix never exists.

- S = Q K^T with WMMA. The Q fragments are loaded once and stay in registers.
- online softmax in the exp2 domain, two lanes per row:
  `m' = max(m, rowmax(S))`, `alpha = 2^(m - m')`, `P = 2^(S - m')`, `l = alpha*l + rowsum(P)`, `O = alpha*O + P V`
- causal masking: a warp skips key tiles that are entirely in its future, and partly masked tiles use -inf.
  Rows that can't see anything yet keep alpha = 1 and P = 0, so there's no (-inf) - (-inf) = NaN.

The awkward part is `O = alpha*O`. WMMA doesn't say which (row, col) each accumulator register holds,
so you can't just multiply row r by alpha_r. What does work: write alpha into a 16x16 shared-memory
tile (every column of row r holds alpha_r), load that as an accumulator fragment of the same type,
and multiply the two fragments element by element. Fragments of the same type use the same element
mapping, so element e lines up in both. The final divide by l uses the same trick.

It needs about 27.6 KB of shared memory at head_dim 64, so a T4 fits 2 blocks per SM.

The better version is FA-2 style with `mma.sync`, where the register layout is documented and the
rescaling can stay in registers.

## CUDA graphs

`cuda/cuda_model.cu`

The decode step (212 kernels) is captured once with `cudaStreamBeginCapture` and replayed with
`cudaGraphLaunch` for every token. The only things that change between tokens are the token id and
the position. They live in a tiny device buffer that gets one async memcpy from pinned memory before
each replay. This only works because every kernel reads the position from device memory and has a
fixed launch shape.

Running `--bench` with and without `--no-graph` shows how much launch overhead the graph removes.

For greedy decoding, argmax runs on the GPU and only 4 bytes come back per token. Sampling copies
the logits back (196 KB) and samples on the CPU.

## WebGPU port

`web/src/`

Most of the CUDA ideas don't carry over directly:

- no warp shuffles, so reductions go through workgroup memory as a tree (subgroups exist but aren't available everywhere yet)
- no guaranteed fp16, so weights are stored as two fp16 values per u32 and unpacked with `unpack2x16float`. That avoids needing `shader-f16`, which Chrome's software adapter and some GPUs don't have.
- no templates, so pipeline-overridable constants (`override K: u32;`) give one specialized pipeline per matrix shape
- no CUDA graphs, so bind groups are built once at load time, each token is one compute pass, and the per-token values go in a 16-byte uniform buffer. Prompt tokens are submitted back to back without waiting on readback.
- `__syncthreads` becomes `workgroupBarrier()`, plus `storageBarrier()` where invocations share data through a storage buffer
- bf16 weights are uploaded raw and converted to fp16 in place by a compute shader

RMSNorm is fused a bit differently here. Each 32-lane group reads the whole input vector for its row
anyway, so it accumulates the sum of squares next to the dot product and scales at the end:
`dot(W, x*gamma) * rsqrt(mean(x^2) + eps)`.

Not ported yet: split-KV attention, batched prefill, int8 weights, subgroup reductions.

## Testing

The checks are layered:

1. numpy (fp32, full causal attention, no cache) is the ground truth for the C++ CPU engine. They agree to about 1e-6.
2. The CUDA and WebGPU engines are compared with the CPU engine on tiny random models: relative error
   around 5e-4 (fp16), the same argmax at every position, and identical greedy output.
3. `tools/hf_compare.py` checks the real model against Hugging Face transformers.

The kernel unit tests use awkward shapes on purpose (odd N, K tails, chunk boundaries, GQA ratios
1 to 8, head_dim 64 and 128). Padding rows and unused KV-cache rows are filled with NaN, so if a
kernel reads anything it shouldn't, the output turns into NaN.

The tokenizer exists twice. The C++ version implements the pre-tokenizer regexes as hand-written
scanners over Unicode category tables generated from Python's `unicodedata`. The JS version just uses
the browser's regex engine with `\p{L}` and `\p{N}`. Comparing them on a few hundred mixed-script
strings (Hindi, CJK, Arabic, emoji, combining marks, odd whitespace) is a good check on the C++ one.

To test CUDA code without a GPU there's a small emulator in `tests/cuda_emu/`: one OS thread per CUDA
thread, real barriers for `__syncthreads` and shuffles, and a functional model of WMMA. The kernels
compile unchanged apart from rewriting the `<<<>>>` launches. It's slow and says nothing about
performance, but CI runs it on every push. A separate CI job builds with the real nvcc.

## Numerics

- fp16 weights and matmul inputs, fp32 accumulation everywhere, fp32 residual stream
- RoPE cos/sin tables are computed once on the host (float32 angles, like HF) and shared by every backend, so positions match exactly between them
- bf16 to fp16 conversion saturates at +-65504
- int8 is symmetric per output row (absmax). On the random test weights it moves the logits by 1-2%; real checkpoints tend to be smoother.

## Next steps

Roughly in order of payoff:

1. GEMM: double-buffered `cp.async` and 128x128 tiles, then `mma.sync` / `ldmatrix`
2. flash attention: FA-2 style softmax in registers with `mma.sync`
3. int4 weights with group-wise scales (AWQ/GPTQ style) for decode
4. WebGPU: split-KV attention, batched prefill, subgroup reductions, int8
5. paged KV cache, continuous batching, speculative decoding
6. more models: Llama 3 RoPE scaling, sliding-window attention, NFC normalization for Qwen2.5
