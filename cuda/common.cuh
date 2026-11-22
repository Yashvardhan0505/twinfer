// Device helpers shared by the kernels: error checks, reductions, fp16/int8 unpacking.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                                          \
  do {                                                                                            \
    cudaError_t err_ = (expr);                                                                    \
    if (err_ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", #expr, __FILE__, __LINE__,          \
                   cudaGetErrorString(err_));                                                     \
      std::exit(1);                                                                               \
    }                                                                                             \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

namespace tf {
namespace cuda {

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFullMask, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFullMask, v, o));
  return v;
}

// block-wide sum, BLOCK must be a multiple of 32. red = 32 floats of smem, every thread gets the result
template <int BLOCK>
__device__ __forceinline__ float block_sum(float v, float* red) {
  static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "bad block size");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = lane < BLOCK / 32 ? red[lane] : 0.0f;
  v = warp_sum(v);
  __syncthreads();  // allow `red` to be reused
  return v;
}

// 8 packed halves (one 16-byte load) -> 8 floats.
__device__ __forceinline__ void unpack_h8(const uint4& u, float* f) {
  const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __half22float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

// dot(8 halves, 8 halves) accumulated in fp32.
__device__ __forceinline__ float dot_h8(const uint4& a, const uint4& b) {
  const __half2* ha = reinterpret_cast<const __half2*>(&a);
  const __half2* hb = reinterpret_cast<const __half2*>(&b);
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x = __half22float2(ha[i]);
    const float2 y = __half22float2(hb[i]);
    s = fmaf(x.x, y.x, s);
    s = fmaf(x.y, y.y, s);
  }
  return s;
}

// 4 x int8 -> 4 floats without I2F. After the xor each byte is b+128, PRMT drops it into the
// low mantissa bits of 2^23 (0x4B0000xx), then subtract 2^23+128. See docs/DESIGN.md.
__device__ __forceinline__ void unpack_i8x4(uint32_t q, float* f) {
  q ^= 0x80808080u;
  f[0] = __int_as_float(__byte_perm(q, 0x4B000000u, 0x7540)) - 8388736.0f;
  f[1] = __int_as_float(__byte_perm(q, 0x4B000000u, 0x7541)) - 8388736.0f;
  f[2] = __int_as_float(__byte_perm(q, 0x4B000000u, 0x7542)) - 8388736.0f;
  f[3] = __int_as_float(__byte_perm(q, 0x4B000000u, 0x7543)) - 8388736.0f;
}

// same trick in fp16: 0x6400 | byte == 1024 + byte, so subtract 1152
__device__ __forceinline__ uint4 i8x8_to_h8(const uint2& q8) {
  uint4 r;
  const uint32_t a = q8.x ^ 0x80808080u, b = q8.y ^ 0x80808080u;
  uint32_t w[4] = {__byte_perm(a, 0x64646464u, 0x5140), __byte_perm(a, 0x64646464u, 0x7362),
                   __byte_perm(b, 0x64646464u, 0x5140), __byte_perm(b, 0x64646464u, 0x7362)};
  const __half2 bias = __float2half2_rn(1152.0f);
  __half2* h = reinterpret_cast<__half2*>(w);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __hsub2(h[i], bias);
  r.x = w[0];
  r.y = w[1];
  r.z = w[2];
  r.w = w[3];
  return r;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace cuda
}  // namespace tf
