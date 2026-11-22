// wmma on the CPU (16x16x16, fp16 in, fp32 accumulate). Every lane keeps a full row-major copy
// of the tile, so all fragment types share one element mapping, which is the property the
// flash attention rescale trick relies on. mma_sync is computed once per warp and shared.
#pragma once
#include <type_traits>

#include "cuda_fp16.h"
#include "cuda_runtime.h"

namespace nvcuda {
namespace wmma {

struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
enum layout_t { mem_row_major, mem_col_major };

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment {
  static constexpr int kRows = std::is_same<Use, matrix_b>::value ? K : M;
  static constexpr int kCols = std::is_same<Use, matrix_a>::value ? K : N;
  enum { num_elements = kRows * kCols };
  T x[num_elements];
};

namespace detail {
inline float to_f(const __half& h) { return __half2float(h); }
inline float to_f(float f) { return f; }
}  // namespace detail

template <typename Use, int M, int N, int K, typename T, typename L>
inline void fill_fragment(fragment<Use, M, N, K, T, L>& f, float v) {
  for (int i = 0; i < f.num_elements; ++i) {
    if constexpr (std::is_same<T, __half>::value) f.x[i] = __float2half(v);
    else f.x[i] = v;
  }
}

// matrix_a / matrix_b with a compile-time layout.
template <typename Use, int M, int N, int K, typename L>
inline void load_matrix_sync(fragment<Use, M, N, K, __half, L>& f, const __half* p, unsigned ldm) {
  using F = fragment<Use, M, N, K, __half, L>;
  for (int r = 0; r < F::kRows; ++r)
    for (int c = 0; c < F::kCols; ++c)
      f.x[r * F::kCols + c] = std::is_same<L, row_major>::value ? p[r * ldm + c] : p[c * ldm + r];
}

// accumulator with a run-time layout.
template <int M, int N, int K>
inline void load_matrix_sync(fragment<accumulator, M, N, K, float>& f, const float* p, unsigned ldm, layout_t l) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) f.x[r * N + c] = l == mem_row_major ? p[r * ldm + c] : p[c * ldm + r];
}

template <int M, int N, int K>
inline void store_matrix_sync(float* p, const fragment<accumulator, M, N, K, float>& f, unsigned ldm, layout_t l) {
  if (emu::lane() == 0) {  // warp-collective store; one lane writes, callers __syncwarp() before reading
    for (int r = 0; r < M; ++r)
      for (int c = 0; c < N; ++c) (l == mem_row_major ? p[r * ldm + c] : p[c * ldm + r]) = f.x[r * N + c];
  }
}

template <int M, int N, int K, typename LA, typename LB>
inline void mma_sync(fragment<accumulator, M, N, K, float>& d, const fragment<matrix_a, M, N, K, __half, LA>& a,
                     const fragment<matrix_b, M, N, K, __half, LB>& b, const fragment<accumulator, M, N, K, float>& c) {
  static_assert(M * N <= 256, "scratch too small");
  float* s = emu::warp_scratch();
  if (emu::lane() == 0) {
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        float acc = c.x[m * N + n];
        for (int k = 0; k < K; ++k) acc += detail::to_f(a.x[m * K + k]) * detail::to_f(b.x[k * N + n]);
        s[m * N + n] = acc;
      }
  }
  emu::warp_barrier();
  for (int i = 0; i < M * N; ++i) d.x[i] = s[i];
  emu::warp_barrier();
}

}  // namespace wmma
}  // namespace nvcuda
