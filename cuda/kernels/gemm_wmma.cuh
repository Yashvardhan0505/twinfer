// Prefill GEMM on tensor cores: C[M, N] = A[M, K] . W[N, K]^T, fp16 in, fp32 accumulate.
// 64x64x32 block tiles, 4 warps in 2x2, each warp does 32x32 as 2x2 wmma fragments.
// smem rows padded to 40 halves (keeps the 32B alignment wmma wants and spreads banks).
// int8 weights are expanded to fp16 while staging, the row scale is applied in the epilogue.
// TODO: double buffer with cp.async (sm80+), 128x128 tiles, then mma.sync + ldmatrix.
#pragma once
#include <mma.h>

#include "../common.cuh"

namespace tf {
namespace cuda {

enum class GemmEpi { StoreF32, AddF32 };

constexpr int kGemmBM = 64, kGemmBN = 64, kGemmBK = 32, kGemmThreads = 128;

__device__ __forceinline__ uint4 load_w8(const __half* W, size_t idx) {
  return __ldg(reinterpret_cast<const uint4*>(W + idx));
}
__device__ __forceinline__ uint4 load_w8(const int8_t* W, size_t idx) {
  return i8x8_to_h8(__ldg(reinterpret_cast<const uint2*>(W + idx)));
}

template <typename WT, GemmEpi EPI>
__global__ void __launch_bounds__(kGemmThreads)
    gemm_wmma_kernel(const __half* __restrict__ A, const WT* __restrict__ W, const float* __restrict__ wscale,
                     const float* __restrict__ bias, float* __restrict__ C, int M, int N, int K) {
  using namespace nvcuda;
  constexpr int BM = kGemmBM, BN = kGemmBN, BK = kGemmBK;
  constexpr int LDS = BK + 8;   // halves
  constexpr int LDC = BN + 4;   // floats
  __shared__ __align__(32) __half As[BM][LDS];
  __shared__ __align__(32) __half Bs[BN][LDS];
  __shared__ __align__(32) float Cs[BM][LDC];

  const int tid = threadIdx.x, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    // load the A and W tiles (64x32 each), 2 x 16B per thread, zeros outside the matrix
#pragma unroll
    for (int it = 0; it < (BM * BK / 8) / kGemmThreads; ++it) {
      const int i = tid + it * kGemmThreads;
      const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
      const int gm = m0 + r, gk = k0 + c8;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (gm < M && gk < K) v = *reinterpret_cast<const uint4*>(A + static_cast<size_t>(gm) * K + gk);
      *reinterpret_cast<uint4*>(&As[r][c8]) = v;
      const int gn = n0 + r;
      uint4 w = make_uint4(0u, 0u, 0u, 0u);
      if (gn < N && gk < K) w = load_w8(W, static_cast<size_t>(gn) * K + gk);
      *reinterpret_cast<uint4*>(&Bs[r][c8]) = w;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[wm * 32 + i * 16][kk], LDS);
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[wn * 32 + j * 16][kk], LDS);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

  // epilogue through smem so the global writes coalesce
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(&Cs[wm * 32 + i * 16][wn * 32 + j * 16], acc[i][j], LDC, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < BM * BN; i += kGemmThreads) {
    const int r = i / BN, c = i % BN;
    const int gm = m0 + r, gn = n0 + c;
    if (gm < M && gn < N) {
      float v = Cs[r][c];
      if (wscale) v *= wscale[gn];
      if (bias) v += bias[gn];
      float* dst = C + static_cast<size_t>(gm) * N + gn;
      if (EPI == GemmEpi::AddF32) *dst += v;
      else *dst = v;
    }
  }
}

// K has to be a multiple of 8, M and N can be anything
template <typename WT>
void gemm(const __half* A, const WT* W, const float* wscale, const float* bias, float* C, int M, int N, int K,
          bool add, cudaStream_t s) {
  const dim3 grid(ceil_div(N, kGemmBN), ceil_div(M, kGemmBM));
  if (add)
    gemm_wmma_kernel<WT, GemmEpi::AddF32><<<grid, kGemmThreads, 0, s>>>(A, W, wscale, bias, C, M, N, K);
  else
    gemm_wmma_kernel<WT, GemmEpi::StoreF32><<<grid, kGemmThreads, 0, s>>>(A, W, wscale, bias, C, M, N, K);
  CUDA_CHECK_LAST();
}

}  // namespace cuda
}  // namespace tf
