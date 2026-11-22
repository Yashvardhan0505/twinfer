// Small kernels: embedding, rope + kv cache append, rmsnorm, swiglu, argmax.
// Positions come from device memory (pos_ptr) so the decode step can live in a CUDA graph.
#pragma once
#include <math_constants.h>

#include "../common.cuh"

namespace tf {
namespace cuda {

// x[t, :] = E[tokens[t], :]   (E is fp16, or int8 with per-row scales)
template <typename WT>
__global__ void embed_kernel(const WT* __restrict__ E, const float* __restrict__ scale, const int* __restrict__ tokens,
                             float* __restrict__ x, int D) {
  const int t = blockIdx.x;
  const int tok = tokens[t];
  const WT* row = E + static_cast<size_t>(tok) * D;
  const float s = scale ? scale[tok] : 1.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float v;
    if constexpr (sizeof(WT) == 1) v = static_cast<float>(row[i]) * s;
    else v = __half2float(row[i]);
    x[static_cast<size_t>(t) * D + i] = v;
  }
}

// qkv is [T, (H + 2*Hkv) * hd] fp32 with the bias already added.
// q -> q_out [T, H*hd], k (rotated) and v -> cache [Hkv, max_seq, hd] for this layer.
// HF rotate_half convention: element i pairs with i + hd/2.
__global__ void rope_kv_kernel(const float* __restrict__ qkv, __half* __restrict__ q_out, __half* __restrict__ k_cache,
                               __half* __restrict__ v_cache, const float* __restrict__ cos_t,
                               const float* __restrict__ sin_t, const int* __restrict__ pos_ptr, int H, int Hkv,
                               int hd, int max_seq) {
  const int t = blockIdx.x;
  const int head = blockIdx.y;  // [0, H) q heads, [H, H+Hkv) k heads, [H+Hkv, H+2Hkv) v heads
  const int half = hd / 2;
  const int pos = *pos_ptr + t;
  const float* row = qkv + static_cast<size_t>(t) * (H + 2 * Hkv) * hd + static_cast<size_t>(head) * hd;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const float a = row[i], b = row[i + half];
    if (head < H + Hkv) {
      const float c = cos_t[static_cast<size_t>(pos) * half + i];
      const float s = sin_t[static_cast<size_t>(pos) * half + i];
      const float ra = a * c - b * s, rb = b * c + a * s;
      __half* dst = head < H ? q_out + static_cast<size_t>(t) * H * hd + static_cast<size_t>(head) * hd
                             : k_cache + (static_cast<size_t>(head - H) * max_seq + pos) * hd;
      dst[i] = __float2half(ra);
      dst[i + half] = __float2half(rb);
    } else {
      __half* dst = v_cache + (static_cast<size_t>(head - H - Hkv) * max_seq + pos) * hd;
      dst[i] = __float2half(a);
      dst[i + half] = __float2half(b);
    }
  }
}

// out[t, :] = fp16(rmsnorm(x[t, :]) * gamma), one block per row.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                                                        __half* __restrict__ out, int D, float eps) {
  __shared__ float red[32];
  const float* xr = x + static_cast<size_t>(blockIdx.x) * D;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += BLOCK) ss = fmaf(xr[i], xr[i], ss);
  ss = block_sum<BLOCK>(ss, red);
  const float inv = rsqrtf(ss / static_cast<float>(D) + eps);
  for (int i = threadIdx.x; i < D; i += BLOCK)
    out[static_cast<size_t>(blockIdx.x) * D + i] = __float2half(xr[i] * inv * gamma[i]);
}

// gu: [T, 2F] fp32 = [gate | up]  ->  h: [T, F] fp16
__global__ void swiglu_kernel(const float* __restrict__ gu, __half* __restrict__ h, int T, int F) {
  const size_t n = static_cast<size_t>(T) * F;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t t = i / F, j = i % F;
    const float g = gu[t * 2 * F + j], u = gu[t * 2 * F + F + j];
    h[i] = __float2half(silu(g) * u);
  }
}

// one block. ties go to the lowest index, same as the CPU version
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) argmax_kernel(const float* __restrict__ x, int n, int* __restrict__ out) {
  __shared__ float sv[BLOCK / 32];
  __shared__ int si[BLOCK / 32];
  float best = -CUDART_INF_F;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += BLOCK) {
    const float v = x[i];
    if (v > best) { best = v; bi = i; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(kFullMask, best, o);
    const int oi = __shfl_xor_sync(kFullMask, bi, o);
    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) { sv[warp] = best; si[warp] = bi; }
  __syncthreads();
  if (warp == 0) {
    best = lane < BLOCK / 32 ? sv[lane] : -CUDART_INF_F;
    bi = lane < BLOCK / 32 ? si[lane] : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(kFullMask, best, o);
      const int oi = __shfl_xor_sync(kFullMask, bi, o);
      if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (lane == 0) *out = bi;
  }
}

}  // namespace cuda
}  // namespace tf
