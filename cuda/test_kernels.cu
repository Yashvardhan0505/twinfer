// Unit tests for each kernel against a double-precision CPU version.
// Shapes are chosen to hit edge cases, and padding / unused cache rows are NaN so that
// out-of-bounds reads show up.
//   ./build/test-kernels          full set
//   ./build/test-kernels --quick  smaller set (for the CPU emulator)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../common/engine.h"
#include "../common/half.h"
#include "kernels/attention_decode.cuh"
#include "kernels/elementwise.cuh"
#include "kernels/flash_attention.cuh"
#include "kernels/gemm_wmma.cuh"
#include "kernels/gemv.cuh"

using namespace tf;

namespace {

int g_failures = 0;
std::mt19937 g_rng(1234);

template <typename T>
struct DBuf {
  T* p = nullptr;
  size_t n = 0;
  explicit DBuf(size_t count) : n(count) {
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
  }
  DBuf(const DBuf&) = delete;
  ~DBuf() { cudaFree(p); }
  void upload(const std::vector<T>& v) { CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> download() const {
    std::vector<T> v(n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
  __half* h() const { return reinterpret_cast<__half*>(p); }
};

std::vector<float> randn(size_t n, float scale = 1.0f) {
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g_rng);
  return v;
}

// Round to fp16 and back, returning both representations.
std::vector<uint16_t> to_h(std::vector<float>& v) {
  std::vector<uint16_t> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    h[i] = f32_to_f16(v[i]);
    v[i] = f16_to_f32(h[i]);
  }
  return h;
}

void check(const std::string& name, double err, double tol) {
  const bool ok = err <= tol && std::isfinite(err);
  std::printf("  %-58s max_err=%.3e tol=%.1e  %s\n", name.c_str(), err, tol, ok ? "ok" : "FAIL");
  if (!ok) ++g_failures;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d <= m)) m = d;  // propagates NaN
  }
  return m;
}

double max_abs(const std::vector<double>& v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::fabs(x));
  return m;
}

// per-row int8 quantization, same as the engine
void quantize_rows(const std::vector<float>& w, int rows, int cols, std::vector<int8_t>& q, std::vector<float>& s,
                   std::vector<float>& deq) {
  q.resize(w.size());
  s.resize(rows);
  deq.resize(w.size());
  for (int r = 0; r < rows; ++r) {
    float amax = 0.0f;
    for (int j = 0; j < cols; ++j) amax = std::max(amax, std::fabs(w[static_cast<size_t>(r) * cols + j]));
    s[r] = amax > 0 ? amax / 127.0f : 1.0f;
    for (int j = 0; j < cols; ++j) {
      const size_t i = static_cast<size_t>(r) * cols + j;
      q[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::nearbyint(w[i] / s[r]))));
      deq[i] = q[i] * s[r];
    }
  }
}

void test_gemv(int N, int K, bool int8, bool norm, bool residual, bool use_bias) {
  std::vector<float> w = randn(static_cast<size_t>(N) * K, 0.05f), xf = randn(K), gamma = randn(K, 0.2f),
                     bias = randn(N), base = randn(N);
  for (auto& g : gamma) g += 1.0f;
  std::vector<float> xs(K);  // what the kernel stages: fp16(norm(x) * gamma) or fp16 x
  std::vector<uint16_t> x_h;
  if (norm) {
    double ss = 0;
    for (float v : xf) ss += static_cast<double>(v) * v;
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / K) + 1e-5f);
    for (int i = 0; i < K; ++i) xs[i] = f16_to_f32(f32_to_f16(xf[i] * inv * gamma[i]));
  } else {
    xs = xf;
    x_h = to_h(xs);
  }
  DBuf<float> d_x(K), d_gamma(K), d_bias(N), d_out(N), d_scale(N);
  DBuf<uint16_t> d_xh(K), d_wh(int8 ? 1 : w.size());
  DBuf<int8_t> d_wq(int8 ? w.size() : 1);
  std::vector<float> wref;
  if (int8) {
    std::vector<int8_t> q;
    std::vector<float> s;
    quantize_rows(w, N, K, q, s, wref);
    d_wq.upload(q);
    d_scale.upload(s);
  } else {
    wref = w;
    d_wh.upload(to_h(wref));
  }
  d_x.upload(xf);
  d_gamma.upload(gamma);
  d_bias.upload(bias);
  d_out.upload(base);
  if (!norm) d_xh.upload(x_h);

  if (int8)
    cuda::gemv<int8_t>(d_wq.p, d_scale.p, d_x.p, d_xh.h(), norm ? d_gamma.p : nullptr, 1e-5f,
                       use_bias ? d_bias.p : nullptr, d_out.p, N, K, residual, nullptr);
  else
    cuda::gemv<__half>(d_wh.h(), nullptr, d_x.p, d_xh.h(), norm ? d_gamma.p : nullptr, 1e-5f,
                       use_bias ? d_bias.p : nullptr, d_out.p, N, K, residual, nullptr);
  std::vector<float> out = d_out.download();
  std::vector<double> ref(N);
  for (int n = 0; n < N; ++n) {
    double acc = 0;
    for (int k = 0; k < K; ++k) acc += static_cast<double>(wref[static_cast<size_t>(n) * K + k]) * xs[k];
    if (use_bias) acc += bias[n];
    ref[n] = residual ? base[n] + acc : acc;
  }
  char name[128];
  std::snprintf(name, sizeof(name), "gemv %s N=%d K=%d%s%s%s", int8 ? "int8" : "fp16", N, K, norm ? " +rmsnorm" : "",
                residual ? " +residual" : "", use_bias ? " +bias" : "");
  check(name, max_abs_diff(out, ref), 1e-3 * (1.0 + max_abs(ref)));
}

void test_gemv_swiglu(int F, int K, bool int8) {
  std::vector<float> w = randn(static_cast<size_t>(2) * F * K, 0.05f), xf = randn(K), gamma(K, 1.0f);
  double ss = 0;
  for (float v : xf) ss += static_cast<double>(v) * v;
  const float inv = 1.0f / std::sqrt(static_cast<float>(ss / K) + 1e-5f);
  std::vector<float> xs(K);
  for (int i = 0; i < K; ++i) xs[i] = f16_to_f32(f32_to_f16(xf[i] * inv));
  DBuf<float> d_x(K), d_gamma(K), d_scale(2 * F);
  DBuf<uint16_t> d_out(F), d_wh(int8 ? 1 : w.size());
  DBuf<int8_t> d_wq(int8 ? w.size() : 1);
  std::vector<float> wref;
  if (int8) {
    std::vector<int8_t> q;
    std::vector<float> s;
    quantize_rows(w, 2 * F, K, q, s, wref);
    d_wq.upload(q);
    d_scale.upload(s);
  } else {
    wref = w;
    d_wh.upload(to_h(wref));
  }
  d_x.upload(xf);
  d_gamma.upload(gamma);
  if (int8) cuda::gemv_swiglu<int8_t>(d_wq.p, d_scale.p, d_x.p, d_gamma.p, 1e-5f, d_out.h(), F, K, nullptr);
  else cuda::gemv_swiglu<__half>(d_wh.h(), nullptr, d_x.p, d_gamma.p, 1e-5f, d_out.h(), F, K, nullptr);
  std::vector<uint16_t> oh = d_out.download();
  std::vector<float> out(F);
  std::vector<double> ref(F);
  for (int i = 0; i < F; ++i) {
    double g = 0, u = 0;
    for (int k = 0; k < K; ++k) {
      g += static_cast<double>(wref[static_cast<size_t>(i) * K + k]) * xs[k];
      u += static_cast<double>(wref[static_cast<size_t>(F + i) * K + k]) * xs[k];
    }
    ref[i] = g / (1.0 + std::exp(-g)) * u;
    out[i] = f16_to_f32(oh[i]);
  }
  char name[128];
  std::snprintf(name, sizeof(name), "gemv+swiglu %s F=%d K=%d", int8 ? "int8" : "fp16", F, K);
  check(name, max_abs_diff(out, ref), 2e-3 * (1.0 + max_abs(ref)));
}

void test_gemm(int M, int N, int K, bool int8, bool add, bool use_bias) {
  std::vector<float> a = randn(static_cast<size_t>(M) * K), w = randn(static_cast<size_t>(N) * K, 0.05f), bias = randn(N),
                     base = randn(static_cast<size_t>(M) * N);
  DBuf<uint16_t> d_a(a.size()), d_wh(int8 ? 1 : w.size());
  DBuf<int8_t> d_wq(int8 ? w.size() : 1);
  DBuf<float> d_scale(N), d_bias(N), d_c(static_cast<size_t>(M) * N);
  d_a.upload(to_h(a));
  std::vector<float> wref;
  if (int8) {
    std::vector<int8_t> q;
    std::vector<float> s;
    quantize_rows(w, N, K, q, s, wref);
    d_wq.upload(q);
    d_scale.upload(s);
  } else {
    wref = w;
    d_wh.upload(to_h(wref));
  }
  d_bias.upload(bias);
  d_c.upload(base);
  if (int8)
    cuda::gemm<int8_t>(d_a.h(), d_wq.p, d_scale.p, use_bias ? d_bias.p : nullptr, d_c.p, M, N, K, add, nullptr);
  else
    cuda::gemm<__half>(d_a.h(), d_wh.h(), nullptr, use_bias ? d_bias.p : nullptr, d_c.p, M, N, K, add, nullptr);
  std::vector<float> c = d_c.download();
  std::vector<double> ref(static_cast<size_t>(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<double>(a[static_cast<size_t>(m) * K + k]) * wref[static_cast<size_t>(n) * K + k];
      if (use_bias) acc += bias[n];
      ref[static_cast<size_t>(m) * N + n] = add ? base[static_cast<size_t>(m) * N + n] + acc : acc;
    }
  char name[128];
  std::snprintf(name, sizeof(name), "gemm wmma %s M=%d N=%d K=%d%s%s", int8 ? "int8" : "fp16", M, N, K,
                add ? " +residual" : "", use_bias ? " +bias" : "");
  check(name, max_abs_diff(c, ref), 1e-3 * (1.0 + max_abs(ref)));
}

// Reference causal attention for query rows [0, T) at absolute positions pos0 + t.
std::vector<double> ref_attention(const std::vector<float>& q, const std::vector<float>& kc, const std::vector<float>& vc,
                                  int T, int pos0, int H, int Hkv, int HD, int max_seq, int ldq) {
  std::vector<double> out(static_cast<size_t>(T) * H * HD, 0.0);
  const int rep = H / Hkv;
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      const int g = h / rep, len = pos0 + t + 1;
      std::vector<double> s(len);
      double mx = -1e300;
      for (int j = 0; j < len; ++j) {
        double acc = 0;
        for (int d = 0; d < HD; ++d)
          acc += static_cast<double>(q[static_cast<size_t>(t) * ldq + h * HD + d]) *
                 kc[(static_cast<size_t>(g) * max_seq + j) * HD + d];
        s[j] = acc / std::sqrt(static_cast<double>(HD));
        mx = std::max(mx, s[j]);
      }
      double l = 0;
      for (int j = 0; j < len; ++j) { s[j] = std::exp(s[j] - mx); l += s[j]; }
      for (int j = 0; j < len; ++j)
        for (int d = 0; d < HD; ++d)
          out[(static_cast<size_t>(t) * H + h) * HD + d] += s[j] / l * vc[(static_cast<size_t>(g) * max_seq + j) * HD + d];
    }
  return out;
}

void test_flash_attn(int HD, int T, int pos0, int H, int Hkv) {
  const int max_seq = 256, ldq = H * HD;
  const int Tpad = cuda::ceil_div(T, cuda::kFaBlockQ) * cuda::kFaBlockQ;
  std::vector<float> q = randn(static_cast<size_t>(Tpad) * ldq), kc = randn(static_cast<size_t>(Hkv) * max_seq * HD),
                     vc = randn(static_cast<size_t>(Hkv) * max_seq * HD);
  std::vector<uint16_t> qh = to_h(q), kh = to_h(kc), vh = to_h(vc);
  // Poison everything the kernel must not read: padded query rows and cache beyond pos0 + T.
  const uint16_t kNaN = 0x7E00;
  for (size_t i = static_cast<size_t>(T) * ldq; i < qh.size(); ++i) qh[i] = kNaN;
  for (int g = 0; g < Hkv; ++g)
    for (int j = pos0 + T; j < max_seq; ++j)
      for (int d = 0; d < HD; ++d) {
        kh[(static_cast<size_t>(g) * max_seq + j) * HD + d] = kNaN;
        vh[(static_cast<size_t>(g) * max_seq + j) * HD + d] = kNaN;
      }
  DBuf<uint16_t> d_q(qh.size()), d_k(kh.size()), d_v(vh.size()), d_o(static_cast<size_t>(T) * ldq);
  DBuf<int> d_pos(1);
  d_q.upload(qh);
  d_k.upload(kh);
  d_v.upload(vh);
  d_pos.upload({pos0});
  cuda::flash_attn_prefill(HD, d_q.h(), d_k.h(), d_v.h(), d_o.h(), d_pos.p, T, H, H / Hkv, max_seq, nullptr);
  std::vector<uint16_t> oh = d_o.download();
  std::vector<float> o(oh.size());
  for (size_t i = 0; i < oh.size(); ++i) o[i] = f16_to_f32(oh[i]);
  std::vector<double> ref = ref_attention(q, kc, vc, T, pos0, H, Hkv, HD, max_seq, ldq);
  char name[128];
  std::snprintf(name, sizeof(name), "flash-attn prefill HD=%d T=%d pos0=%d H=%d Hkv=%d", HD, T, pos0, H, Hkv);
  check(name, max_abs_diff(o, ref), 6e-3);
}

void test_attn_decode(int HD, int pos, int Hkv, int rep) {
  const int max_seq = 512, H = Hkv * rep;
  const int n_chunks = cuda::ceil_div(max_seq, cuda::kAttnChunk);
  std::vector<float> q = randn(static_cast<size_t>(H) * HD), kc = randn(static_cast<size_t>(Hkv) * max_seq * HD),
                     vc = randn(static_cast<size_t>(Hkv) * max_seq * HD);
  std::vector<uint16_t> qh = to_h(q), kh = to_h(kc), vh = to_h(vc);
  const uint16_t kNaN = 0x7E00;
  for (int g = 0; g < Hkv; ++g)
    for (int j = pos + 1; j < max_seq; ++j)
      for (int d = 0; d < HD; ++d) {
        kh[(static_cast<size_t>(g) * max_seq + j) * HD + d] = kNaN;
        vh[(static_cast<size_t>(g) * max_seq + j) * HD + d] = kNaN;
      }
  DBuf<uint16_t> d_q(qh.size()), d_k(kh.size()), d_v(vh.size()), d_o(static_cast<size_t>(H) * HD);
  DBuf<float> d_po(static_cast<size_t>(H) * n_chunks * HD), d_pml(static_cast<size_t>(H) * n_chunks * 2);
  DBuf<int> d_pos(1);
  d_q.upload(qh);
  d_k.upload(kh);
  d_v.upload(vh);
  d_pos.upload({pos});
  cuda::attn_decode(HD, rep, d_q.h(), d_k.h(), d_v.h(), d_pos.p, Hkv, max_seq, d_po.p, d_pml.p, d_o.h(), nullptr);
  std::vector<uint16_t> oh = d_o.download();
  std::vector<float> o(oh.size());
  for (size_t i = 0; i < oh.size(); ++i) o[i] = f16_to_f32(oh[i]);
  std::vector<double> ref = ref_attention(q, kc, vc, 1, pos, H, Hkv, HD, max_seq, H * HD);
  char name[128];
  std::snprintf(name, sizeof(name), "attn decode split-KV HD=%d pos=%d Hkv=%d rep=%d", HD, pos, Hkv, rep);
  check(name, max_abs_diff(o, ref), 3e-3);
}

void test_rope_kv() {
  const int T = 3, pos0 = 7, H = 4, Hkv = 2, hd = 64, max_seq = 16;
  const int W = (H + 2 * Hkv) * hd;
  std::vector<float> qkv = randn(static_cast<size_t>(T) * W);
  ModelConfig c;
  c.head_dim = hd;
  c.rope_theta = 10000.0f;
  std::vector<float> cs, sn;
  rope_table(c, max_seq, cs, sn);
  DBuf<float> d_qkv(qkv.size()), d_cos(cs.size()), d_sin(sn.size());
  DBuf<uint16_t> d_q(static_cast<size_t>(T) * H * hd), d_k(static_cast<size_t>(Hkv) * max_seq * hd),
      d_v(static_cast<size_t>(Hkv) * max_seq * hd);
  DBuf<int> d_pos(1);
  d_qkv.upload(qkv);
  d_cos.upload(cs);
  d_sin.upload(sn);
  d_pos.upload({pos0});
  cuda::rope_kv_kernel<<<dim3(T, H + 2 * Hkv), hd / 2>>>(d_qkv.p, d_q.h(), d_k.h(), d_v.h(), d_cos.p, d_sin.p, d_pos.p, H,
                                                       Hkv, hd, max_seq);
  CUDA_CHECK_LAST();
  std::vector<uint16_t> q = d_q.download(), k = d_k.download(), v = d_v.download();
  double err = 0;
  const int half = hd / 2;
  for (int t = 0; t < T; ++t) {
    const int p = pos0 + t;
    for (int hh = 0; hh < H + 2 * Hkv; ++hh) {
      const float* x = &qkv[static_cast<size_t>(t) * W + hh * hd];
      for (int i = 0; i < half; ++i) {
        const float c0 = cs[p * half + i], s0 = sn[p * half + i];
        float e0 = x[i], e1 = x[i + half];
        if (hh < H + Hkv) { e0 = x[i] * c0 - x[i + half] * s0; e1 = x[i + half] * c0 + x[i] * s0; }
        const uint16_t* dst = hh < H ? &q[(static_cast<size_t>(t) * H + hh) * hd]
                              : hh < H + Hkv ? &k[(static_cast<size_t>(hh - H) * max_seq + p) * hd]
                                             : &v[(static_cast<size_t>(hh - H - Hkv) * max_seq + p) * hd];
        err = std::max(err, static_cast<double>(std::fabs(f16_to_f32(dst[i]) - e0)));
        err = std::max(err, static_cast<double>(std::fabs(f16_to_f32(dst[i + half]) - e1)));
      }
    }
  }
  check("rope + kv-cache append T=3 pos0=7", err, 4e-3);
}

void test_argmax(int n) {
  std::vector<float> x = randn(n);
  const int a = n / 3, b = (2 * n) / 3;
  x[a] = x[b] = 100.0f;  // tie: lowest index must win
  DBuf<float> d_x(n);
  DBuf<int> d_out(1);
  d_x.upload(x);
  cuda::argmax_kernel<1024><<<1, 1024>>>(d_x.p, n, d_out.p);
  CUDA_CHECK_LAST();
  const int got = d_out.download()[0];
  char name[64];
  std::snprintf(name, sizeof(name), "argmax n=%d (tie -> lowest index)", n);
  check(name, got == a ? 0.0 : 1.0, 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
  std::printf("twinfer kernel tests%s\n", quick ? " (quick)" : "");

  std::printf("[gemv]\n");
  test_gemv(1000, 576, false, true, false, true);
  test_gemv(77, 1536, false, false, true, false);
  test_gemv(1000, 576, true, true, false, false);
  test_gemv(129, 1536, true, false, true, false);
  test_gemv_swiglu(300, 576, false);
  test_gemv_swiglu(300, 576, true);
  if (!quick) {
    test_gemv(20000, 576, false, true, false, false);  // more rows than the grid: grid-stride loop
    test_gemv(20000, 576, true, true, false, false);
    test_gemv(2048, 4096, false, false, true, false);
  }

  std::printf("[gemm]\n");
  test_gemm(1, 64, 64, false, false, true);
  test_gemm(17, 200, 72, false, false, true);   // K % 32 == 8 tail
  test_gemm(100, 96, 264, false, true, false);
  test_gemm(37, 130, 136, true, true, false);
  test_gemm(64, 128, 576, true, false, true);
  if (!quick) {
    test_gemm(512, 960, 576, false, false, true);
    test_gemm(333, 1536, 576, true, true, false);
  }

  std::printf("[flash attention prefill]\n");
  test_flash_attn(64, 1, 0, 4, 2);
  test_flash_attn(64, 37, 0, 4, 2);
  test_flash_attn(64, 64, 50, 3, 3);
  test_flash_attn(128, 70, 5, 4, 1);
  if (!quick) {
    test_flash_attn(64, 200, 40, 9, 3);
    test_flash_attn(128, 130, 100, 8, 2);
  }

  std::printf("[decode attention]\n");
  test_attn_decode(64, 0, 2, 1);
  test_attn_decode(64, 127, 3, 3);
  test_attn_decode(64, 128, 2, 8);
  test_attn_decode(128, 300, 1, 4);
  if (!quick) {
    test_attn_decode(64, 511, 3, 3);
    test_attn_decode(128, 450, 2, 7);
  }

  std::printf("[misc]\n");
  test_rope_kv();
  test_argmax(quick ? 559 : 49152);

  if (g_failures) {
    std::printf("%d test(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all tests passed\n");
  return 0;
}
