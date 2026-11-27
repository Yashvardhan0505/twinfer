# Benchmarks

The raw output behind [Numbers](../README.md#numbers) in the README.

## Tesla T4 on Colab, commit 7aaa4e1

Tesla T4 (sm_75, 40 SMs, 4.2 MB L2, 320 GB/s peak DRAM bandwidth), driver 580.82.07, Colab's free runtime with
2 vCPUs. Model: SmolLM2-135M-Instruct, 269.0 MB of fp16 weights. Everything below comes from one run of
[notebooks/colab_quickstart.ipynb](../notebooks/colab_quickstart.ipynb) on a fresh runtime. The kernel unit tests
(`test-kernels`, 30 tests) and `tests/run_tests.py --cpu --cuda` pass on this GPU too.

### End to end: `python3 tools/bench_table.py models/smollm2-135m`

The GPU rows take turns, one run each per round, after a warm-up round that isn't counted. Every run:

```
CUDA fp16 + graph: prefill 38142, 38275, 21293 | decode 482, 481, 490
CUDA fp16, no graph: prefill 37417, 37963, 37669 | decode 480, 470, 468
CUDA int8 + graph: prefill 41435, 42043, 40614 | decode 564, 570, 583
CPU fp32: prefill 17, 16.1, 17.1 | decode 22.7, 22.2, 21.8
```

The medians:

| backend | prefill tok/s (512-token prompt) | decode tok/s | ms/token |
|---|---|---|---|
| CUDA fp16 + graph | 38142 | 482 | 2.08 |
| CUDA fp16, no graph | 37669 | 470 | 2.13 |
| CUDA int8 + graph | 41435 | 570 | 1.75 |
| CPU fp32 | 17 | 22.2 | 45.05 |

### Kernels: `./build/bench-kernels`

Each time is the median of 5 samples. A sample queues a batch of calls behind a short spin kernel and times the
batch, so the calls run back to back, as they do inside the decode graph. GEMV weights rotate through enough copies
to miss the L2 cache, as every layer's weights do in a real decode step.

#### Decode GEMV (batch 1, SmolLM2-135M shapes)

The twinfer kernels also do the fused work the engine needs (RMSNorm on qkv, gate+up and lm_head, SwiGLU on
gate+up, the residual add on o_proj and down). cuBLAS does the plain matrix product.

| matrix | N x K | twinfer fp16 | GB/s | % peak | twinfer int8 | GB/s | cuBLAS fp16 | GB/s |
|---|---|---|---|---|---|---|---|---|
| qkv | 960 x 576 | 13.1 us | 84 | 26% | 12.9 us | 43 | 12.5 us | 89 |
| o_proj | 576 x 576 | 8.7 us | 76 | 24% | 8.9 us | 38 | 10.1 us | 66 |
| gate+up | 3072 x 576 | 31.8 us | 111 | 35% | 29.0 us | 62 | 29.6 us | 120 |
| down | 576 x 1536 | 13.3 us | 133 | 42% | 9.9 us | 90 | 13.4 us | 132 |
| lm_head | 49152 x 576 | 250.3 us | 226 | 71% | 262.0 us | 109 | 325.4 us | 174 |

#### Prefill GEMM (tensor cores, fp16 in / fp32 accumulate)

| M x N x K | twinfer WMMA | TFLOP/s | cuBLAS | TFLOP/s | twinfer / cuBLAS |
|---|---|---|---|---|---|
| 128 x 960 x 576 | 29.1 us | 4.9 | 20.1 us | 7.1 | 69% |
| 512 x 960 x 576 | 75.0 us | 7.6 | 33.6 us | 16.9 | 45% |
| 512 x 3072 x 576 | 197.2 us | 9.2 | 95.5 us | 19.0 | 48% |
| 512 x 576 x 1536 | 98.9 us | 9.2 | 66.2 us | 13.7 | 67% |
| 2048 x 3072 x 576 | 757.2 us | 9.6 | 337.4 us | 21.5 | 45% |
| 4096 x 4096 x 4096 | 17699.3 us | 7.8 | 5729.8 us | 24.0 | 32% |

#### Decode attention (split-KV, GQA 9/3 heads, head_dim 64)

| context | time | KV GB/s | % peak |
|---|---|---|---|
| 128 | 8.7 us | 11 | 4% |
| 512 | 9.0 us | 44 | 14% |
| 1024 | 9.2 us | 85 | 27% |
| 2048 | 10.7 us | 147 | 46% |
| 4096 | 14.6 us | 215 | 67% |

#### Prefill flash attention (causal, 9/3 heads, head_dim 64)

| tokens | time | TFLOP/s |
|---|---|---|
| 128 | 11.8 us | 1.62 |
| 512 | 51.7 us | 5.85 |
| 1024 | 154.5 us | 7.82 |
| 2048 | 489.5 us | 9.88 |

The best cuBLAS rate measured was 24.0 TFLOP/s, which the roofline uses as the practical compute roof.

### Correctness: `python3 tools/hf_compare.py models/smollm2-135m --cuda`

```
tokenizer  C++ vs HF: 228/228 identical
tokenizer  JS  vs HF: 228/228 identical
  note: transformers 5.22.1 AutoTokenizer differs from tokenizer.json on 10 of these (its pre-tokenizer: ByteLevel, the file's: Sequence); not counted
chat template identical: True
logits     CPU fp32 (decode)            max rel err 3.46e-06  top-1 agreement 1.000  ok
logits     CUDA fp16 (decode + graph)   max rel err 2.28e-03  top-1 agreement 1.000  ok
logits     CUDA fp16 (prefill chunks)   max rel err 4.36e-04  top-1 agreement 1.000  ok
logits     CUDA int8 (decode)           max rel err 6.84e-02  top-1 agreement 0.923 (3 differ: all near-ties)  ok

HF greedy:      'Shared memory refers to a type of memory that is shared among multiple processes or threads in a program. It is a shared resource that is shared among multiple processes or threads, allowing them to access and use the shared memory location without having to'
twinfer CPU:    'Shared memory refers to a type of memory that is shared among multiple processes or threads in a program. It is a shared resource that is shared among multiple processes or threads, allowing them to access and use the shared memory location without having to'  (identical)
twinfer CUDA:   'Shared memory refers to a type of memory that is shared among multiple processes or threads in a program. It is a shared resource that is shared among multiple processes or threads, allowing them to access and use the shared memory location without having to'  (identical)

all checks passed
```

The tokenizer reference is `tokenizer.json` run by Hugging Face's tokenizers library, the tokenizer the model was
trained with. transformers 5 builds SmolLM2's tokenizer from the `GPT2Tokenizer` class's defaults instead, which
drops the file's Digits pre-tokenizer, so it splits non-ASCII numbers like ½ and ٣ differently on those 10 strings.
The 3 int8 positions that differ are near-ties: Hugging Face's own top two tokens there are 0.02 to 0.03 logits
apart.
