// Kernel benchmarks vs cuBLAS:  ./build/bench-kernels [--peak-gbps X] [--csv out.csv]
// SmolLM2-135M shapes. GEMV weights rotate through enough copies to blow past L2,
// otherwise the small matrices would just come out of cache.
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "../common/half.h"
#include "kernels/attention_decode.cuh"
#include "kernels/flash_attention.cuh"
#include "kernels/gemm_wmma.cuh"
#include "kernels/gemv.cuh"

using namespace tf;

#define CUBLAS_CHECK(expr)                                                              \
  do {                                                                                  \
    cublasStatus_t st_ = (expr);                                                        \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                                 \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

namespace {

struct Csv {
  FILE* f = nullptr;
  void row(const std::string& kernel, const std::string& shape, double us, double bytes, double flops) {
    if (f) std::fprintf(f, "%s,%s,%.3f,%.0f,%.0f\n", kernel.c_str(), shape.c_str(), us, bytes, flops);
  }
} g_csv;

void* dmalloc_random(size_t bytes, uint32_t seed) {
  void* p;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  std::vector<uint16_t> h(bytes / 2 + 1);
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 0.05f);
  for (auto& v : h) v = f32_to_f16(d(rng));  // valid fp16 bit patterns (and valid int8 bytes)
  CUDA_CHECK(cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice));
  return p;
}

// median time in us over reps runs, i = which weight copy to use
double time_us(const std::function<void(int)>& fn, int reps = 50) {
  for (int i = 0; i < 5; ++i) fn(i);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<float> t(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(a));
    fn(i);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&t[i], a, b));
  }
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  std::sort(t.begin(), t.end());
  return t[reps / 2] * 1000.0;
}

// C[M, N] (row-major) = A[M, K] . W[N, K]^T with cuBLAS (column-major: C^T = W . A^T).
void cublas_gemm(cublasHandle_t h, const __half* A, const __half* W, float* C, int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C,
                            CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

struct Shape {
  const char* name;
  int N, K;
};

}  // namespace

int main(int argc, char** argv) {
  double peak_gbps = 0.0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--peak-gbps") && i + 1 < argc) peak_gbps = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      g_csv.f = std::fopen(argv[++i], "w");
      if (g_csv.f) std::fprintf(g_csv.f, "kernel,shape,us,bytes,flops\n");
    }
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (peak_gbps == 0.0) {
    int mem_clock_khz = 0, bus_bits = 0;
    if (cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0) == cudaSuccess &&
        cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0) == cudaSuccess)
      peak_gbps = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  }
  const size_t l2 = static_cast<size_t>(prop.l2CacheSize);
  std::printf("# twinfer kernel benchmarks\n\nGPU: **%s** (sm_%d%d, %d SMs, L2 %.1f MB), peak DRAM bandwidth %s\n\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, l2 / 1e6,
              peak_gbps > 0 ? (std::to_string(static_cast<int>(peak_gbps)) + " GB/s").c_str() : "unknown (--peak-gbps)");

  cublasHandle_t cub;
  CUBLAS_CHECK(cublasCreate(&cub));
  auto pct = [&](double gbps) { return peak_gbps > 0 ? 100.0 * gbps / peak_gbps : 0.0; };

  // GEMV (decode)
  const int D = 576, F = 1536, V = 49152;
  const Shape gemv_shapes[] = {{"qkv", 960, D}, {"o_proj", D, D}, {"gate+up", 2 * F, D}, {"down", D, F}, {"lm_head", V, D}};
  std::printf("## Decode GEMV (batch 1, SmolLM2-135M shapes)\n\n");
  std::printf("| matrix | N x K | twinfer fp16 | GB/s | %% peak | twinfer int8 | GB/s | cuBLAS fp16 | GB/s |\n");
  std::printf("|---|---|---|---|---|---|---|---|---|\n");
  float* d_xf;
  __half* d_xh;
  float* d_gamma;
  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_xf, 16384 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xh, 16384 * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_gamma, 16384 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(V) * 4 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_xf, 0, 16384 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_xh, 0, 16384 * sizeof(__half)));
  CUDA_CHECK(cudaMemset(d_gamma, 0, 16384 * sizeof(float)));
  for (const Shape& s : gemv_shapes) {
    const size_t bytes16 = static_cast<size_t>(s.N) * s.K * 2, bytes8 = static_cast<size_t>(s.N) * s.K + s.N * 4;
    const int copies = static_cast<int>(std::min<size_t>(64, std::max<size_t>(1, (4 * l2) / bytes8 + 1)));
    std::vector<void*> w16(copies), w8(copies);
    std::vector<float*> sc(copies);
    for (int c = 0; c < copies; ++c) {
      w16[c] = dmalloc_random(bytes16, 1 + c);
      w8[c] = dmalloc_random(static_cast<size_t>(s.N) * s.K, 100 + c);
      CUDA_CHECK(cudaMalloc(&sc[c], s.N * sizeof(float)));
      CUDA_CHECK(cudaMemset(sc[c], 0, s.N * sizeof(float)));
    }
    const std::string nm = s.name;
    const bool swiglu = nm == "gate+up";
    // same variants as the engine: fused norm on qkv / gate+up / lm_head, fp16 input for o / down
    float* gamma = (nm == "o_proj" || nm == "down") ? nullptr : d_gamma;
    auto run16 = [&](int i) {
      const __half* w = static_cast<const __half*>(w16[i % copies]);
      if (swiglu) cuda::gemv_swiglu<__half>(w, nullptr, d_xf, d_gamma, 1e-5f, d_xh + 8192, s.N / 2, s.K, nullptr);
      else cuda::gemv<__half>(w, nullptr, d_xf, d_xh, gamma, 1e-5f, nullptr, d_out, s.N, s.K, false, nullptr);
    };
    auto run8 = [&](int i) {
      const int8_t* w = static_cast<const int8_t*>(w8[i % copies]);
      if (swiglu) cuda::gemv_swiglu<int8_t>(w, sc[i % copies], d_xf, d_gamma, 1e-5f, d_xh + 8192, s.N / 2, s.K, nullptr);
      else cuda::gemv<int8_t>(w, sc[i % copies], d_xf, d_xh, gamma, 1e-5f, nullptr, d_out, s.N, s.K, false, nullptr);
    };
    auto runcb = [&](int i) { cublas_gemm(cub, d_xh, static_cast<const __half*>(w16[i % copies]), d_out, 1, s.N, s.K); };
    const double t16 = time_us(run16), t8 = time_us(run8), tcb = time_us(runcb);
    const double g16 = bytes16 / t16 / 1e3, g8 = bytes8 / t8 / 1e3, gcb = bytes16 / tcb / 1e3;
    std::printf("| %s | %d x %d | %.1f us | %.0f | %.0f%% | %.1f us | %.0f | %.1f us | %.0f |\n", s.name, s.N, s.K, t16,
                g16, pct(g16), t8, g8, tcb, gcb);
    const std::string shape = std::to_string(s.N) + "x" + std::to_string(s.K);
    g_csv.row(std::string("gemv_fp16_") + s.name, shape, t16, bytes16, 2.0 * s.N * s.K);
    g_csv.row(std::string("gemv_int8_") + s.name, shape, t8, bytes8, 2.0 * s.N * s.K);
    g_csv.row(std::string("cublas_gemv_") + s.name, shape, tcb, bytes16, 2.0 * s.N * s.K);
    for (int c = 0; c < copies; ++c) {
      cudaFree(w16[c]);
      cudaFree(w8[c]);
      cudaFree(sc[c]);
    }
  }

  // GEMM (prefill)
  std::printf("\n## Prefill GEMM (tensor cores, fp16 in / fp32 accumulate)\n\n");
  std::printf("| M x N x K | twinfer WMMA | TFLOP/s | cuBLAS | TFLOP/s | twinfer / cuBLAS |\n|---|---|---|---|---|---|\n");
  const int gemm_shapes[][3] = {{128, 960, 576}, {512, 960, 576}, {512, 3072, 576}, {512, 576, 1536}, {2048, 3072, 576},
                                {4096, 4096, 4096}};
  double cublas_best_tflops = 0.0;
  for (const auto& g : gemm_shapes) {
    const int M = g[0], N = g[1], K = g[2];
    __half* A = static_cast<__half*>(dmalloc_random(static_cast<size_t>(M) * K * 2, 7));
    __half* W = static_cast<__half*>(dmalloc_random(static_cast<size_t>(N) * K * 2, 8));
    float* C;
    CUDA_CHECK(cudaMalloc(&C, static_cast<size_t>(M) * N * sizeof(float)));
    const double flops = 2.0 * M * N * K;
    const double tw = time_us([&](int) { cuda::gemm<__half>(A, W, nullptr, nullptr, C, M, N, K, false, nullptr); }, 20);
    const double tc = time_us([&](int) { cublas_gemm(cub, A, W, C, M, N, K); }, 20);
    cublas_best_tflops = std::max(cublas_best_tflops, flops / tc / 1e6);
    std::printf("| %d x %d x %d | %.1f us | %.1f | %.1f us | %.1f | %.0f%% |\n", M, N, K, tw, flops / tw / 1e6, tc,
                flops / tc / 1e6, 100.0 * tc / tw);
    const std::string shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    g_csv.row("gemm_wmma", shape, tw, (static_cast<double>(M) * K + static_cast<double>(N) * K) * 2 + M * N * 4.0, flops);
    g_csv.row("cublas_gemm", shape, tc, (static_cast<double>(M) * K + static_cast<double>(N) * K) * 2 + M * N * 4.0, flops);
    cudaFree(A);
    cudaFree(W);
    cudaFree(C);
  }

  // attention
  const int H = 9, Hkv = 3, HD = 64, max_seq = 4096;
  const size_t cache_elems = static_cast<size_t>(Hkv) * max_seq * HD;
  __half* kc = static_cast<__half*>(dmalloc_random(cache_elems * 2, 11));
  __half* vc = static_cast<__half*>(dmalloc_random(cache_elems * 2, 12));
  __half* q = static_cast<__half*>(dmalloc_random(static_cast<size_t>(max_seq) * H * HD * 2, 13));
  __half* o;
  float *po, *pml;
  int* dpos;
  const int n_chunks = cuda::ceil_div(max_seq, cuda::kAttnChunk);
  CUDA_CHECK(cudaMalloc(&o, static_cast<size_t>(max_seq) * H * HD * 2));
  CUDA_CHECK(cudaMalloc(&po, static_cast<size_t>(H) * n_chunks * HD * 4));
  CUDA_CHECK(cudaMalloc(&pml, static_cast<size_t>(H) * n_chunks * 2 * 4));
  CUDA_CHECK(cudaMalloc(&dpos, sizeof(int)));

  std::printf("\n## Decode attention (split-KV, GQA %d/%d heads, head_dim %d)\n\n", H, Hkv, HD);
  std::printf("| context | time | KV GB/s | %% peak |\n|---|---|---|---|\n");
  for (int ctx : {128, 512, 1024, 2048, 4096}) {
    const int pos = ctx - 1;
    CUDA_CHECK(cudaMemcpy(dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));
    const double t = time_us([&](int) {
      cuda::attn_decode(HD, H / Hkv, q, kc, vc, dpos, Hkv, max_seq, po, pml, o, nullptr);
    });
    const double bytes = 2.0 * Hkv * ctx * HD * 2;
    std::printf("| %d | %.1f us | %.0f | %.0f%% |\n", ctx, t, bytes / t / 1e3, pct(bytes / t / 1e3));
    g_csv.row("attn_decode", std::to_string(ctx), t, bytes, 4.0 * H * ctx * HD);
  }

  std::printf("\n## Prefill flash attention (causal, %d/%d heads, head_dim %d)\n\n", H, Hkv, HD);
  std::printf("| tokens | time | TFLOP/s |\n|---|---|---|\n");
  const int zero = 0;
  CUDA_CHECK(cudaMemcpy(dpos, &zero, sizeof(int), cudaMemcpyHostToDevice));
  for (int T : {128, 512, 1024, 2048}) {
    const double t = time_us([&](int) { cuda::flash_attn_prefill(HD, q, kc, vc, o, dpos, T, H, H / Hkv, max_seq, nullptr); }, 20);
    const double flops = 2.0 * 2.0 * H * HD * (static_cast<double>(T) * (T + 1) / 2);  // causal QK^T + PV
    std::printf("| %d | %.1f us | %.2f |\n", T, t, flops / t / 1e6);
    g_csv.row("flash_attn_prefill", std::to_string(T), t, 2.0 * 3 * T * H * HD, flops);
  }

  std::printf("\nBest cuBLAS rate measured: %.1f TFLOP/s (use as the practical compute roof).\n", cublas_best_tflops);
  if (g_csv.f) {
    std::fprintf(g_csv.f, "# gpu=%s\n# peak_gbps=%.1f\n# cublas_tflops=%.2f\n", prop.name, peak_gbps, cublas_best_tflops);
    std::fclose(g_csv.f);
  }
  CUBLAS_CHECK(cublasDestroy(cub));
  return 0;
}
