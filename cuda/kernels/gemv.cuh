// GEMV for decode (one token): y = W x, W row-major [N, K].
// Memory bound, so it's all about streaming W: one warp per row, 16B loads, x staged once
// per block in smem as fp16, fp32 accumulation.
// NORM folds the RMSNorm into the staging step. Epilogues: bias, residual add, or SwiGLU
// over the [gate; up] rows.
#pragma once
#include "../common.cuh"

namespace tf {
namespace cuda {

enum class GemvOut { Store, AddResidual, SwiGLU };

constexpr int kGemvThreads = 256;  // 8 warps -> 8 rows in flight per block

// this lane's part of row . x
__device__ __forceinline__ float row_dot(const __half* __restrict__ W, size_t row, const __half* xs, int K,
                                         int lane) {
  const uint4* w = reinterpret_cast<const uint4*>(W + row * K);
  const uint4* x = reinterpret_cast<const uint4*>(xs);
  float acc = 0.0f;
#pragma unroll 4
  for (int i = lane; i < K / 8; i += 32) acc += dot_h8(__ldg(w + i), x[i]);
  return acc;
}

__device__ __forceinline__ float row_dot(const int8_t* __restrict__ W, size_t row, const __half* xs, int K,
                                         int lane) {
  const uint4* w = reinterpret_cast<const uint4*>(W + row * K);
  const uint4* x = reinterpret_cast<const uint4*>(xs);
  float acc = 0.0f;
#pragma unroll 2
  for (int i = lane; i < K / 16; i += 32) {
    const uint4 q = __ldg(w + i);
    const uint32_t qs[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float wf[8], xf[8];
      unpack_i8x4(qs[2 * j], wf);
      unpack_i8x4(qs[2 * j + 1], wf + 4);
      unpack_h8(x[2 * i + j], xf);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc = fmaf(wf[e], xf[e], acc);
    }
  }
  return acc;  // requires K % 16 == 0 (checked by the engine)
}

template <typename WT>
__device__ __forceinline__ float row_scale(const float* scale, int row) {
  return scale[row];
}
template <>
__device__ __forceinline__ float row_scale<__half>(const float*, int) {
  return 1.0f;
}

// N = number of outputs. For SwiGLU, W has 2N rows (gate then up) and out_h[i] = silu(g_i) * u_i
template <typename WT, bool NORM, GemvOut OUT>
__global__ void __launch_bounds__(kGemvThreads)
    gemv_kernel(const WT* __restrict__ W, const float* __restrict__ wscale, const float* __restrict__ x_f32,
                const __half* __restrict__ x_f16, const float* __restrict__ gamma, float eps,
                const float* __restrict__ bias, float* __restrict__ out_f32, __half* __restrict__ out_f16, int N,
                int K) {
  extern __shared__ __align__(16) unsigned char gemv_smem[];
  __half* xs = reinterpret_cast<__half*>(gemv_smem);
  __shared__ float red[32];
  const int tid = threadIdx.x;

  // stage x in smem as fp16 (normalized if NORM)
  if (NORM) {
    float ss = 0.0f;
    for (int i = tid; i < K; i += kGemvThreads) {
      const float v = x_f32[i];
      ss = fmaf(v, v, ss);
    }
    ss = block_sum<kGemvThreads>(ss, red);
    const float inv = rsqrtf(ss / static_cast<float>(K) + eps);
    for (int i = tid; i < K; i += kGemvThreads) xs[i] = __float2half(x_f32[i] * inv * gamma[i]);
  } else {
    const uint4* src = reinterpret_cast<const uint4*>(x_f16);
    uint4* dst = reinterpret_cast<uint4*>(xs);
    for (int i = tid; i < K / 8; i += kGemvThreads) dst[i] = src[i];
  }
  __syncthreads();

  // one warp per row. row is the same across the warp, so the shuffles are safe
  const int lane = tid & 31, warp = tid >> 5;
  constexpr int kWarps = kGemvThreads / 32;
  for (int row = blockIdx.x * kWarps + warp; row < N; row += gridDim.x * kWarps) {
    float acc = warp_sum(row_dot(W, row, xs, K, lane)) * row_scale<WT>(wscale, row);
    if (OUT == GemvOut::SwiGLU) {
      const float up = warp_sum(row_dot(W, static_cast<size_t>(row) + N, xs, K, lane)) *
                       row_scale<WT>(wscale, row + N);
      if (lane == 0) out_f16[row] = __float2half(silu(acc) * up);
    } else if (lane == 0) {
      if (bias) acc += bias[row];
      if (OUT == GemvOut::AddResidual) out_f32[row] += acc;
      else out_f32[row] = acc;
    }
  }
}

inline int gemv_grid(int rows) {
  const int blocks = ceil_div(rows, kGemvThreads / 32);
  return blocks < 2048 ? blocks : 2048;
}

// launchers. x_f32 is read when gamma != nullptr (fused norm), x_f16 otherwise
template <typename WT>
void gemv(const WT* W, const float* wscale, const float* x_f32, const __half* x_f16, const float* gamma, float eps,
          const float* bias, float* out, int N, int K, bool add_residual, cudaStream_t s) {
  const size_t smem = static_cast<size_t>(K) * sizeof(__half);
  const int grid = gemv_grid(N);
  if (gamma) {
    if (add_residual)
      gemv_kernel<WT, true, GemvOut::AddResidual><<<grid, kGemvThreads, smem, s>>>(W, wscale, x_f32, x_f16, gamma, eps,
                                                                                bias, out, nullptr, N, K);
    else
      gemv_kernel<WT, true, GemvOut::Store><<<grid, kGemvThreads, smem, s>>>(W, wscale, x_f32, x_f16, gamma, eps, bias,
                                                                          out, nullptr, N, K);
  } else {
    if (add_residual)
      gemv_kernel<WT, false, GemvOut::AddResidual><<<grid, kGemvThreads, smem, s>>>(W, wscale, x_f32, x_f16, gamma,
                                                                                 eps, bias, out, nullptr, N, K);
    else
      gemv_kernel<WT, false, GemvOut::Store><<<grid, kGemvThreads, smem, s>>>(W, wscale, x_f32, x_f16, gamma, eps,
                                                                           bias, out, nullptr, N, K);
  }
  CUDA_CHECK_LAST();
}

// h = silu(Wg . norm(x)) * (Wu . norm(x)), W = [Wg; Wu] is [2F, K]
template <typename WT>
void gemv_swiglu(const WT* W, const float* wscale, const float* x_f32, const float* gamma, float eps, __half* out_h,
                 int F, int K, cudaStream_t s) {
  const size_t smem = static_cast<size_t>(K) * sizeof(__half);
  gemv_kernel<WT, true, GemvOut::SwiGLU><<<gemv_grid(F), kGemvThreads, smem, s>>>(
      W, wscale, x_f32, nullptr, gamma, eps, nullptr, nullptr, out_h, F, K);
  CUDA_CHECK_LAST();
}

}  // namespace cuda
}  // namespace tf
