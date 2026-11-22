// Prefill attention, flash-attention style, on tensor cores (wmma).
// Block = 4 warps = 64 query rows of one head. Keys go through smem in tiles of 32:
//   S = QK^T (wmma) -> online softmax (exp2) -> O = alpha*O + PV (wmma), O stays in registers.
//
// Scaling O by a per-row alpha is awkward because wmma doesn't document the fragment layout.
// Trick: write alpha into a 16x16 smem tile (row r is alpha_r everywhere), load it as an
// accumulator fragment and multiply element-wise. Same fragment type -> same layout.
// TODO: redo this with mma.sync (FA-2 style) so the rescale stays in registers.
//
// Q: [T, H, HD] fp16, rows padded to a multiple of 64
// K/V: [Hkv, max_seq, HD] fp16 cache for this layer, filled up to pos0 + T
// O: [T, H*HD] fp16
#pragma once
#include <math_constants.h>
#include <mma.h>

#include "../common.cuh"

namespace tf {
namespace cuda {

constexpr int kFaBlockQ = 64, kFaBlockK = 32, kFaThreads = 128;

template <int HD>
__global__ void __launch_bounds__(kFaThreads)
    flash_attn_prefill_kernel(const __half* __restrict__ Q, const __half* __restrict__ k_cache,
                              const __half* __restrict__ v_cache, __half* __restrict__ O, const int* __restrict__ pos_ptr,
                              int T, int H, int n_rep, int max_seq, float scale_log2) {
  using namespace nvcuda;
  constexpr int BK = kFaBlockK;
  constexpr int LDKV = HD + 8;  // halves
  constexpr int LDS = BK + 4;   // floats
  constexpr int LDP = BK + 8;   // halves
  constexpr int NF = HD / 16;   // output fragments per warp (16 x HD)

  __shared__ __align__(32) __half Ks[BK][LDKV];
  __shared__ __align__(32) __half Vs[BK][LDKV];
  __shared__ __align__(32) float Ss[4][16][LDS];
  __shared__ __align__(32) __half Ps[4][16][LDP];
  __shared__ __align__(32) float Rs[4][16][16];  // per-row scale tile

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = blockIdx.y, g = h / n_rep;
  const int pos0 = *pos_ptr;
  const int q_base = blockIdx.x * kFaBlockQ;          // first query row of the block
  const int w_row0 = q_base + warp * 16;              // first query row of this warp
  const int kv_end = pos0 + min(q_base + kFaBlockQ, T);  // keys needed by this block
  const int ldq = H * HD;

  // this warp's 16 rows of Q, loaded once
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf[NF];
#pragma unroll
  for (int f = 0; f < NF; ++f)
    wmma::load_matrix_sync(qf[f], Q + static_cast<size_t>(w_row0) * ldq + static_cast<size_t>(h) * HD + f * 16, ldq);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> of[NF];
#pragma unroll
  for (int f = 0; f < NF; ++f) wmma::fill_fragment(of[f], 0.0f);

  // lanes 2r and 2r+1 both track the softmax state of row r
  const int r = lane >> 1, half_sel = lane & 1;
  const int q_pos = pos0 + w_row0 + r;  // absolute position of this row
  float m_run = -CUDART_INF_F, l_run = 0.0f;

  const __half* kbase = k_cache + static_cast<size_t>(g) * max_seq * HD;
  const __half* vbase = v_cache + static_cast<size_t>(g) * max_seq * HD;

  for (int j0 = 0; j0 < kv_end; j0 += BK) {
    // K/V tile into smem, zeros past kv_end
    for (int i = tid; i < BK * HD / 8; i += kFaThreads) {
      const int row = i / (HD / 8), c8 = (i % (HD / 8)) * 8;
      uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = make_uint4(0u, 0u, 0u, 0u);
      if (j0 + row < kv_end) {
        kv = *reinterpret_cast<const uint4*>(kbase + static_cast<size_t>(j0 + row) * HD + c8);
        vv = *reinterpret_cast<const uint4*>(vbase + static_cast<size_t>(j0 + row) * HD + c8);
      }
      *reinterpret_cast<uint4*>(&Ks[row][c8]) = kv;
      *reinterpret_cast<uint4*>(&Vs[row][c8]) = vv;
    }
    __syncthreads();

    // skip tiles that are completely in the future for this warp
    if (j0 <= pos0 + w_row0 + 15) {
      // S = QK^T, 16 rows x 32 keys
#pragma unroll
      for (int nf = 0; nf < BK / 16; ++nf) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf;
        wmma::fill_fragment(sf, 0.0f);
#pragma unroll
        for (int f = 0; f < NF; ++f) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
          wmma::load_matrix_sync(kf, &Ks[nf * 16][f * 16], LDKV);
          wmma::mma_sync(sf, qf[f], kf, sf);
        }
        wmma::store_matrix_sync(&Ss[warp][0][nf * 16], sf, LDS, wmma::mem_row_major);
      }
      __syncwarp();

      // online softmax, each lane of the pair takes 16 of the 32 columns
      float s[BK / 2];
      float mx = -CUDART_INF_F;
#pragma unroll
      for (int c = 0; c < BK / 2; ++c) {
        const int col = half_sel * (BK / 2) + c;
        const float v = (j0 + col <= q_pos) ? Ss[warp][r][col] * scale_log2 : -CUDART_INF_F;
        s[c] = v;
        mx = fmaxf(mx, v);
      }
      mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
      const float m_new = fmaxf(m_run, mx);
      const float alpha = (m_new == -CUDART_INF_F) ? 1.0f : exp2f(m_run - m_new);
      float sum = 0.0f;
#pragma unroll
      for (int c = 0; c < BK / 2; ++c) {
        const float p = (m_new == -CUDART_INF_F) ? 0.0f : exp2f(s[c] - m_new);
        sum += p;
        Ps[warp][r][half_sel * (BK / 2) + c] = __float2half(p);
      }
      sum += __shfl_xor_sync(kFullMask, sum, 1);
      l_run = l_run * alpha + sum;
      m_run = m_new;
#pragma unroll
      for (int c = 0; c < 8; ++c) Rs[warp][r][half_sel * 8 + c] = alpha;
      __syncwarp();

      // O = alpha * O + P V
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> af;
      wmma::load_matrix_sync(af, &Rs[warp][0][0], 16, wmma::mem_row_major);
#pragma unroll
      for (int f = 0; f < NF; ++f) {
#pragma unroll
        for (int e = 0; e < of[f].num_elements; ++e) of[f].x[e] *= af.x[e];
      }
#pragma unroll
      for (int kf = 0; kf < BK / 16; ++kf) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
        wmma::load_matrix_sync(pf, &Ps[warp][0][kf * 16], LDP);
#pragma unroll
        for (int f = 0; f < NF; ++f) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
          wmma::load_matrix_sync(vf, &Vs[kf * 16][f * 16], LDKV);
          wmma::mma_sync(of[f], pf, vf, of[f]);
        }
      }
    }
    __syncthreads();  // Ks / Vs are overwritten by the next tile
  }

  // divide by l and write fp16, one 16x16 fragment at a time through the S buffer
  const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
#pragma unroll
  for (int c = 0; c < 8; ++c) Rs[warp][r][half_sel * 8 + c] = inv_l;
  __syncwarp();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> lf;
  wmma::load_matrix_sync(lf, &Rs[warp][0][0], 16, wmma::mem_row_major);
#pragma unroll
  for (int f = 0; f < NF; ++f) {
#pragma unroll
    for (int e = 0; e < of[f].num_elements; ++e) of[f].x[e] *= lf.x[e];
    wmma::store_matrix_sync(&Ss[warp][0][0], of[f], LDS, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 16 * 16; i += 32) {
      const int rr = i / 16, cc = i % 16;
      const int row = w_row0 + rr;
      if (row < T) O[static_cast<size_t>(row) * ldq + static_cast<size_t>(h) * HD + f * 16 + cc] = __float2half(Ss[warp][rr][cc]);
    }
    __syncwarp();
  }
}

inline void flash_attn_prefill(int head_dim, const __half* Q, const __half* kc, const __half* vc, __half* O,
                               const int* pos_ptr, int T, int H, int n_rep, int max_seq, cudaStream_t s) {
  const dim3 grid(ceil_div(T, kFaBlockQ), H);
  const float scale_log2 = 1.4426950408889634f / sqrtf(static_cast<float>(head_dim));
  if (head_dim == 64)
    flash_attn_prefill_kernel<64><<<grid, kFaThreads, 0, s>>>(Q, kc, vc, O, pos_ptr, T, H, n_rep, max_seq, scale_log2);
  else if (head_dim == 128)
    flash_attn_prefill_kernel<128><<<grid, kFaThreads, 0, s>>>(Q, kc, vc, O, pos_ptr, T, H, n_rep, max_seq, scale_log2);
  else {
    std::fprintf(stderr, "unsupported head_dim %d\n", head_dim);
    std::exit(1);
  }
  CUDA_CHECK_LAST();
}

}  // namespace cuda
}  // namespace tf
