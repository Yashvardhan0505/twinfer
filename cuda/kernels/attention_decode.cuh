// Decode attention (single query token) over the fp16 KV cache.
// Split-KV like flash-decoding: grid = (kv_heads, max_seq / 128) so even 3 kv heads keep the
// GPU busy, then attn_combine_kernel merges the chunks (log-sum-exp).
// Each block handles all NREP query heads sharing its kv head, so K/V is read once, not NREP times.
// Blocks past the current length just exit, so the grid never changes (needed for CUDA graphs).
#pragma once
#include <math_constants.h>

#include "../common.cuh"

namespace tf {
namespace cuda {

constexpr int kAttnChunk = 128;   // keys per block
constexpr int kAttnThreads = 128; // 4 warps

// q:      [H, HD] fp16 (H = n_kv_heads * NREP)
// k/v:    [Hkv, max_seq, HD] fp16 (this layer)
// part_o: [H, n_chunks, HD] fp32 (unnormalized sum p*v), part_ml: [H, n_chunks, 2] (max, sum)
template <int HD, int NREP>
__global__ void __launch_bounds__(kAttnThreads)
    attn_decode_kernel(const __half* __restrict__ q, const __half* __restrict__ k_cache,
                       const __half* __restrict__ v_cache, const int* __restrict__ pos_ptr, int max_seq,
                       float scale_log2, float* __restrict__ part_o, float* __restrict__ part_ml, int n_chunks) {
  static_assert(HD % 32 == 0 && HD <= 256, "head_dim must be a multiple of 32");
  constexpr int DPL = HD / 32;  // dims per lane in the PV phase
  const int g = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
  const int len = *pos_ptr + 1;
  const int start = c * kAttnChunk;

  if (start >= len) {  // nothing in this chunk yet, mark it empty
    if (tid < NREP) {
      const size_t idx = (static_cast<size_t>(g * NREP + tid) * n_chunks + c) * 2;
      part_ml[idx] = -CUDART_INF_F;
      part_ml[idx + 1] = 0.0f;
    }
    return;
  }
  const int n = min(kAttnChunk, len - start);

  __shared__ float qs[NREP][HD];
  __shared__ float sc[NREP][kAttnChunk];
  __shared__ float red_o[kAttnThreads / 32][NREP][HD];
  __shared__ float m_s[NREP], l_s[NREP];

  for (int i = tid; i < NREP * HD; i += kAttnThreads)
    qs[i / HD][i % HD] = __half2float(q[static_cast<size_t>(g) * NREP * HD + i]);
  __syncthreads();

  // scores: thread j takes key start+j and dots it with all NREP queries
  {
    float acc[NREP];
#pragma unroll
    for (int r = 0; r < NREP; ++r) acc[r] = 0.0f;
    if (tid < n) {
      const uint4* kr = reinterpret_cast<const uint4*>(k_cache + (static_cast<size_t>(g) * max_seq + start + tid) * HD);
#pragma unroll
      for (int d8 = 0; d8 < HD / 8; ++d8) {
        float kf[8];
        unpack_h8(__ldg(kr + d8), kf);
#pragma unroll
        for (int r = 0; r < NREP; ++r)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[r] = fmaf(kf[e], qs[r][d8 * 8 + e], acc[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < NREP; ++r) sc[r][tid] = tid < n ? acc[r] * scale_log2 : -CUDART_INF_F;
  }
  __syncthreads();

  // softmax stats, one warp per head
  const int warp = tid >> 5, lane = tid & 31;
  for (int r = warp; r < NREP; r += kAttnThreads / 32) {
    float mx = -CUDART_INF_F;
    for (int j = lane; j < kAttnChunk; j += 32) mx = fmaxf(mx, sc[r][j]);
    mx = warp_max(mx);
    float sum = 0.0f;
    for (int j = lane; j < kAttnChunk; j += 32) {
      const float p = exp2f(sc[r][j] - mx);
      sc[r][j] = p;
      sum += p;
    }
    sum = warp_sum(sum);
    if (lane == 0) { m_s[r] = mx; l_s[r] = sum; }
  }
  __syncthreads();

  // o = sum p_j v_j. warp w takes keys w, w+4, ..., each lane owns DPL dims (coalesced V reads)
  float o[NREP][DPL];
#pragma unroll
  for (int r = 0; r < NREP; ++r)
#pragma unroll
    for (int e = 0; e < DPL; ++e) o[r][e] = 0.0f;
  for (int j = warp; j < n; j += kAttnThreads / 32) {
    const __half* vr = v_cache + (static_cast<size_t>(g) * max_seq + start + j) * HD + lane * DPL;
    float vf[DPL];
#pragma unroll
    for (int e = 0; e < DPL; e += 2) {
      const float2 t = __half22float2(*reinterpret_cast<const __half2*>(vr + e));
      vf[e] = t.x;
      vf[e + 1] = t.y;
    }
#pragma unroll
    for (int r = 0; r < NREP; ++r) {
      const float p = sc[r][j];
#pragma unroll
      for (int e = 0; e < DPL; ++e) o[r][e] = fmaf(p, vf[e], o[r][e]);
    }
  }
#pragma unroll
  for (int r = 0; r < NREP; ++r)
#pragma unroll
    for (int e = 0; e < DPL; ++e) red_o[warp][r][lane * DPL + e] = o[r][e];
  __syncthreads();

  for (int i = tid; i < NREP * HD; i += kAttnThreads) {
    const int r = i / HD, d = i % HD;
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < kAttnThreads / 32; ++w) v += red_o[w][r][d];
    part_o[(static_cast<size_t>(g * NREP + r) * n_chunks + c) * HD + d] = v;
  }
  if (tid < NREP) {
    const size_t idx = (static_cast<size_t>(g * NREP + tid) * n_chunks + c) * 2;
    part_ml[idx] = m_s[tid];
    part_ml[idx + 1] = l_s[tid];
  }
}

// Merge chunk partials: out[h, :] = sum_c 2^(m_c - M) o_c / sum_c 2^(m_c - M) l_c
template <int HD>
__global__ void __launch_bounds__(HD)
    attn_combine_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml,
                        const int* __restrict__ pos_ptr, int n_chunks, __half* __restrict__ out) {
  const int h = blockIdx.x, d = threadIdx.x;
  const int used = (*pos_ptr + kAttnChunk) / kAttnChunk;  // ceil((pos + 1) / chunk)
  const float* ml = part_ml + static_cast<size_t>(h) * n_chunks * 2;
  float M = -CUDART_INF_F;
  for (int c = 0; c < used; ++c) M = fmaxf(M, ml[2 * c]);
  float L = 0.0f, acc = 0.0f;
  for (int c = 0; c < used; ++c) {
    const float w = exp2f(ml[2 * c] - M);
    L = fmaf(w, ml[2 * c + 1], L);
    acc = fmaf(w, part_o[(static_cast<size_t>(h) * n_chunks + c) * HD + d], acc);
  }
  out[static_cast<size_t>(h) * HD + d] = __float2half(acc / L);
}

template <int HD, int NREP>
void attn_decode_launch(const __half* q, const __half* kc, const __half* vc, const int* pos_ptr, int n_kv_heads,
                        int max_seq, float* part_o, float* part_ml, __half* out, cudaStream_t s) {
  const int n_chunks = ceil_div(max_seq, kAttnChunk);
  const float scale_log2 = 1.4426950408889634f / sqrtf(static_cast<float>(HD));
  attn_decode_kernel<HD, NREP><<<dim3(n_kv_heads, n_chunks), kAttnThreads, 0, s>>>(q, kc, vc, pos_ptr, max_seq,
                                                                                  scale_log2, part_o, part_ml, n_chunks);
  CUDA_CHECK_LAST();
  attn_combine_kernel<HD><<<n_kv_heads * NREP, HD, 0, s>>>(part_o, part_ml, pos_ptr, n_chunks, out);
  CUDA_CHECK_LAST();
}

template <int HD>
void attn_decode_dispatch_rep(int n_rep, const __half* q, const __half* kc, const __half* vc, const int* pos_ptr,
                              int n_kv_heads, int max_seq, float* part_o, float* part_ml, __half* out,
                              cudaStream_t s) {
  switch (n_rep) {
    case 1: attn_decode_launch<HD, 1>(q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s); break;
    case 2: attn_decode_launch<HD, 2>(q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s); break;
    case 3: attn_decode_launch<HD, 3>(q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s); break;
    case 4: attn_decode_launch<HD, 4>(q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s); break;
    case 5: attn_decode_launch<HD, 5>(q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s); break;
    case 6: attn_decode_launch<HD, 6>(q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s); break;
    case 7: attn_decode_launch<HD, 7>(q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s); break;
    case 8: attn_decode_launch<HD, 8>(q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s); break;
    default: std::fprintf(stderr, "unsupported GQA ratio %d (max 8)\n", n_rep); std::exit(1);
  }
}

inline void attn_decode(int head_dim, int n_rep, const __half* q, const __half* kc, const __half* vc,
                        const int* pos_ptr, int n_kv_heads, int max_seq, float* part_o, float* part_ml, __half* out,
                        cudaStream_t s) {
  if (head_dim == 64)
    attn_decode_dispatch_rep<64>(n_rep, q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s);
  else if (head_dim == 128)
    attn_decode_dispatch_rep<128>(n_rep, q, kc, vc, pos_ptr, n_kv_heads, max_seq, part_o, part_ml, out, s);
  else {
    std::fprintf(stderr, "unsupported head_dim %d (64 or 128)\n", head_dim);
    std::exit(1);
  }
}

}  // namespace cuda
}  // namespace tf
