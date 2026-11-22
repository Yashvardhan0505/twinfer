#include "cuda_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/half.h"
#include "kernels/attention_decode.cuh"
#include "kernels/elementwise.cuh"
#include "kernels/flash_attention.cuh"
#include "kernels/gemm_wmma.cuh"
#include "kernels/gemv.cuh"

namespace tf {

using cuda::ceil_div;

struct CudaModel::Impl {
  struct Layer {
    void* w_qkv = nullptr;   // [Q + 2KV, D]   fused q/k/v projection
    float* s_qkv = nullptr;  // per-row int8 scales (nullptr for fp16)
    float* b_qkv = nullptr;  // optional bias
    void* w_o = nullptr;     // [D, Q]
    float* s_o = nullptr;
    void* w_gu = nullptr;    // [2F, D]        fused [gate; up]
    float* s_gu = nullptr;
    void* w_down = nullptr;  // [D, F]
    float* s_down = nullptr;
    float* attn_norm = nullptr;
    float* ffn_norm = nullptr;
  };

  ModelConfig c;
  int max_seq = 0;
  CudaOptions opt;
  std::string name;
  std::vector<Layer> layers;
  void* embed = nullptr;
  float* s_embed = nullptr;
  void* lm_head = nullptr;
  float* s_lm = nullptr;
  float* final_norm = nullptr;
  float* rope_cos = nullptr;
  float* rope_sin = nullptr;
  __half* k_cache = nullptr;  // [L, Hkv, max_seq, hd]
  __half* v_cache = nullptr;

  // Activations (sized for one prefill chunk).
  float* x = nullptr;       // [Tmax, D] residual stream
  __half* xn = nullptr;     // [Tmax, D] normalised
  float* qkv = nullptr;     // [Tmax, Q + 2KV]
  __half* q = nullptr;      // [Tpad, Q]
  __half* attn = nullptr;   // [Tmax, Q]
  float* gu = nullptr;      // [Tmax, 2F]
  __half* hbuf = nullptr;   // [Tmax, F]
  float* logits = nullptr;  // [V]
  float* part_o = nullptr;  // [H, n_chunks, hd]
  float* part_ml = nullptr; // [H, n_chunks, 2]
  int* d_meta = nullptr;    // [0] = pos0, [1..] = token ids
  int* d_argmax = nullptr;
  int* h_meta = nullptr;    // pinned staging
  float* h_logits = nullptr;
  int* h_argmax = nullptr;

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  size_t weight_bytes = 0;
  size_t device_bytes = 0;
  std::vector<void*> allocs;

  template <typename T>
  T* dalloc(size_t n) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
    allocs.push_back(p);
    device_bytes += n * sizeof(T);
    return static_cast<T*>(p);
  }

  float* upload_f32(const std::vector<float>& v) {
    float* d = dalloc<float>(v.size());
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
  }

  // Upload a row-major [rows, cols] matrix as fp16, or as int8 with per-row absmax scales.
  void* upload_matrix(const std::vector<float>& w, int rows, int cols, float** scale) {
    const size_t n = static_cast<size_t>(rows) * cols;
    if (!opt.int8) {
      std::vector<uint16_t> h(n);
      for (size_t i = 0; i < n; ++i) h[i] = f32_to_f16_sat(w[i]);
      __half* d = dalloc<__half>(n);
      CUDA_CHECK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice));
      *scale = nullptr;
      weight_bytes += n * 2;
      return d;
    }
    std::vector<int8_t> qv(n);
    std::vector<float> s(rows);
    for (int r = 0; r < rows; ++r) {
      const float* row = &w[static_cast<size_t>(r) * cols];
      float amax = 0.0f;
      for (int j = 0; j < cols; ++j) amax = std::max(amax, std::fabs(row[j]));
      const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
      s[r] = sc;
      for (int j = 0; j < cols; ++j) {
        const float v = std::nearbyint(row[j] / sc);
        qv[static_cast<size_t>(r) * cols + j] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, v)));
      }
    }
    int8_t* d = dalloc<int8_t>(n);
    CUDA_CHECK(cudaMemcpy(d, qv.data(), n, cudaMemcpyHostToDevice));
    *scale = upload_f32(s);
    weight_bytes += n + rows * sizeof(float);
    return d;
  }

  void check_shapes() const {
    const int align = opt.int8 ? 16 : 8;
    for (int k : {c.dim, c.q_dim(), c.hidden_dim})
      if (k % align != 0)
        throw std::runtime_error("CUDA backend needs dim, n_heads*head_dim and ffn dim to be multiples of " +
                                 std::to_string(align));
    if (c.head_dim != 64 && c.head_dim != 128) throw std::runtime_error("CUDA backend supports head_dim 64 or 128");
    if (c.n_heads / c.n_kv_heads > 8) throw std::runtime_error("CUDA backend supports GQA ratios up to 8");
    if (c.q_dim() % 16 != 0) throw std::runtime_error("n_heads * head_dim must be a multiple of 16");
  }

  void load(const ModelFiles& mf) {
    check_shapes();
    const int D = c.dim, Q = c.q_dim(), KV = c.kv_dim(), F = c.hidden_dim, V = c.vocab_size;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<float> e = mf.f32("model.embed_tokens.weight", {V, D});
    embed = upload_matrix(e, V, D, &s_embed);
    if (c.tied_embeddings) {
      lm_head = embed;
      s_lm = s_embed;
    } else {
      lm_head = upload_matrix(mf.f32("lm_head.weight", {V, D}), V, D, &s_lm);
    }
    final_norm = upload_f32(mf.f32("model.norm.weight", {D}));

    layers.resize(c.n_layers);
    for (int l = 0; l < c.n_layers; ++l) {
      Layer& L = layers[l];
      auto t = [&](const char* s) { return ModelFiles::layer(l, s); };
      std::vector<float> qkv(static_cast<size_t>(Q + 2 * KV) * D);
      mf.read_f32(t("self_attn.q_proj.weight"), {Q, D}, qkv.data());
      mf.read_f32(t("self_attn.k_proj.weight"), {KV, D}, qkv.data() + static_cast<size_t>(Q) * D);
      mf.read_f32(t("self_attn.v_proj.weight"), {KV, D}, qkv.data() + static_cast<size_t>(Q + KV) * D);
      L.w_qkv = upload_matrix(qkv, Q + 2 * KV, D, &L.s_qkv);
      if (c.qkv_bias) {
        std::vector<float> b(Q + 2 * KV);
        mf.read_f32(t("self_attn.q_proj.bias"), {Q}, b.data());
        mf.read_f32(t("self_attn.k_proj.bias"), {KV}, b.data() + Q);
        mf.read_f32(t("self_attn.v_proj.bias"), {KV}, b.data() + Q + KV);
        L.b_qkv = upload_f32(b);
      }
      L.w_o = upload_matrix(mf.f32(t("self_attn.o_proj.weight"), {D, Q}), D, Q, &L.s_o);
      std::vector<float> gu(static_cast<size_t>(2) * F * D);
      mf.read_f32(t("mlp.gate_proj.weight"), {F, D}, gu.data());
      mf.read_f32(t("mlp.up_proj.weight"), {F, D}, gu.data() + static_cast<size_t>(F) * D);
      L.w_gu = upload_matrix(gu, 2 * F, D, &L.s_gu);
      L.w_down = upload_matrix(mf.f32(t("mlp.down_proj.weight"), {D, F}), D, F, &L.s_down);
      L.attn_norm = upload_f32(mf.f32(t("input_layernorm.weight"), {D}));
      L.ffn_norm = upload_f32(mf.f32(t("post_attention_layernorm.weight"), {D}));
    }

    std::vector<float> cs, sn;
    rope_table(c, max_seq, cs, sn);
    rope_cos = upload_f32(cs);
    rope_sin = upload_f32(sn);

    const size_t cache = static_cast<size_t>(c.n_layers) * c.n_kv_heads * max_seq * c.head_dim;
    k_cache = dalloc<__half>(cache);
    v_cache = dalloc<__half>(cache);

    const int Tmax = opt.max_prefill;
    const int Tpad = ceil_div(Tmax, cuda::kFaBlockQ) * cuda::kFaBlockQ;
    x = dalloc<float>(static_cast<size_t>(Tmax) * D);
    xn = dalloc<__half>(static_cast<size_t>(Tmax) * D);
    qkv = dalloc<float>(static_cast<size_t>(Tmax) * (Q + 2 * KV));
    q = dalloc<__half>(static_cast<size_t>(Tpad) * Q);
    attn = dalloc<__half>(static_cast<size_t>(Tmax) * Q);
    gu = dalloc<float>(static_cast<size_t>(Tmax) * 2 * F);
    hbuf = dalloc<__half>(static_cast<size_t>(Tmax) * F);
    logits = dalloc<float>(V);
    const int n_chunks = ceil_div(max_seq, cuda::kAttnChunk);
    part_o = dalloc<float>(static_cast<size_t>(c.n_heads) * n_chunks * c.head_dim);
    part_ml = dalloc<float>(static_cast<size_t>(c.n_heads) * n_chunks * 2);
    d_meta = dalloc<int>(Tmax + 1);
    d_argmax = dalloc<int>(1);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_meta), (Tmax + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_logits), V * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_argmax), sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  template <typename WT>
  void decode_kernels() {
    const int D = c.dim, Q = c.q_dim(), F = c.hidden_dim, V = c.vocab_size, hd = c.head_dim;
    const int H = c.n_heads, Hkv = c.n_kv_heads, QKV = c.qkv_dim();
    const float eps = c.norm_eps;
    const int* pos = d_meta;
    const int* tok = d_meta + 1;
    const size_t layer_cache = static_cast<size_t>(Hkv) * max_seq * hd;
    cudaStream_t s = stream;

    cuda::embed_kernel<WT><<<1, 256, 0, s>>>(static_cast<const WT*>(embed), s_embed, tok, x, D);
    CUDA_CHECK_LAST();
    for (int l = 0; l < c.n_layers; ++l) {
      const Layer& L = layers[l];
      __half* kc = k_cache + l * layer_cache;
      __half* vc = v_cache + l * layer_cache;
      // qkv = Wqkv . rmsnorm(x) + b
      cuda::gemv<WT>(static_cast<const WT*>(L.w_qkv), L.s_qkv, x, nullptr, L.attn_norm, eps, L.b_qkv, qkv, QKV, D,
                     false, s);
      cuda::rope_kv_kernel<<<dim3(1, H + 2 * Hkv), hd / 2, 0, s>>>(qkv, q, kc, vc, rope_cos, rope_sin, pos, H, Hkv,
                                                                   hd, max_seq);
      CUDA_CHECK_LAST();
      cuda::attn_decode(hd, H / Hkv, q, kc, vc, pos, Hkv, max_seq, part_o, part_ml, attn, s);
      // x += Wo . attn
      cuda::gemv<WT>(static_cast<const WT*>(L.w_o), L.s_o, nullptr, attn, nullptr, 0.0f, nullptr, x, D, Q, true, s);
      // h = silu(Wg . rmsnorm(x)) * (Wu . rmsnorm(x));  x += Wd . h
      cuda::gemv_swiglu<WT>(static_cast<const WT*>(L.w_gu), L.s_gu, x, L.ffn_norm, eps, hbuf, F, D, s);
      cuda::gemv<WT>(static_cast<const WT*>(L.w_down), L.s_down, nullptr, hbuf, nullptr, 0.0f, nullptr, x, D, F,
                     true, s);
    }
    cuda::gemv<WT>(static_cast<const WT*>(lm_head), s_lm, x, nullptr, final_norm, eps, nullptr, logits, V, D, false,
                   s);
  }

  template <typename WT>
  void prefill_kernels(int T) {
    const int D = c.dim, Q = c.q_dim(), F = c.hidden_dim, V = c.vocab_size, hd = c.head_dim;
    const int H = c.n_heads, Hkv = c.n_kv_heads, QKV = c.qkv_dim();
    const float eps = c.norm_eps;
    const int* pos = d_meta;
    const int* tok = d_meta + 1;
    const size_t layer_cache = static_cast<size_t>(Hkv) * max_seq * hd;
    cudaStream_t s = stream;

    cuda::embed_kernel<WT><<<T, 256, 0, s>>>(static_cast<const WT*>(embed), s_embed, tok, x, D);
    CUDA_CHECK_LAST();
    for (int l = 0; l < c.n_layers; ++l) {
      const Layer& L = layers[l];
      __half* kc = k_cache + l * layer_cache;
      __half* vc = v_cache + l * layer_cache;
      cuda::rmsnorm_kernel<256><<<T, 256, 0, s>>>(x, L.attn_norm, xn, D, eps);
      CUDA_CHECK_LAST();
      cuda::gemm<WT>(xn, static_cast<const WT*>(L.w_qkv), L.s_qkv, L.b_qkv, qkv, T, QKV, D, false, s);
      cuda::rope_kv_kernel<<<dim3(T, H + 2 * Hkv), hd / 2, 0, s>>>(qkv, q, kc, vc, rope_cos, rope_sin, pos, H, Hkv,
                                                                   hd, max_seq);
      CUDA_CHECK_LAST();
      cuda::flash_attn_prefill(hd, q, kc, vc, attn, pos, T, H, H / Hkv, max_seq, s);
      cuda::gemm<WT>(attn, static_cast<const WT*>(L.w_o), L.s_o, nullptr, x, T, D, Q, true, s);
      cuda::rmsnorm_kernel<256><<<T, 256, 0, s>>>(x, L.ffn_norm, xn, D, eps);
      CUDA_CHECK_LAST();
      cuda::gemm<WT>(xn, static_cast<const WT*>(L.w_gu), L.s_gu, nullptr, gu, T, 2 * F, D, false, s);
      const int blocks = std::min(ceil_div(T * F, 256), 4096);
      cuda::swiglu_kernel<<<blocks, 256, 0, s>>>(gu, hbuf, T, F);
      CUDA_CHECK_LAST();
      cuda::gemm<WT>(hbuf, static_cast<const WT*>(L.w_down), L.s_down, nullptr, x, T, D, F, true, s);
    }
    // Only the last position's logits are needed.
    cuda::gemv<WT>(static_cast<const WT*>(lm_head), s_lm, x + static_cast<size_t>(T - 1) * D, nullptr, final_norm, eps,
                   nullptr, logits, V, D, false, s);
  }

  void decode() {
    if (!opt.graph) {
      if (opt.int8) decode_kernels<int8_t>();
      else decode_kernels<__half>();
      return;
    }
    if (!graph_exec) {
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      if (opt.int8) decode_kernels<int8_t>();
      else decode_kernels<__half>();
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
    }
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  }

  void run(const int* tokens, int n, int pos0) {
    if (n <= 0) throw std::runtime_error("forward() needs at least one token");
    if (pos0 < 0 || pos0 + n > max_seq) throw std::runtime_error("sequence exceeds the context length");
    for (int i = 0; i < n; ++i)
      if (tokens[i] < 0 || tokens[i] >= c.vocab_size) throw std::runtime_error("token id out of range");
    bool first = true;
    while (n > 0) {
      const int chunk = std::min(n, opt.max_prefill);
      if (!first) CUDA_CHECK(cudaStreamSynchronize(stream));  // h_meta is about to be reused
      first = false;
      h_meta[0] = pos0;
      std::memcpy(h_meta + 1, tokens, chunk * sizeof(int));
      CUDA_CHECK(cudaMemcpyAsync(d_meta, h_meta, (chunk + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
      if (chunk == 1) {
        decode();
      } else if (opt.int8) {
        prefill_kernels<int8_t>(chunk);
      } else {
        prefill_kernels<__half>(chunk);
      }
      tokens += chunk;
      n -= chunk;
      pos0 += chunk;
    }
  }

  ~Impl() {
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
    for (void* p : allocs) cudaFree(p);
    if (h_meta) cudaFreeHost(h_meta);
    if (h_logits) cudaFreeHost(h_logits);
    if (h_argmax) cudaFreeHost(h_argmax);
    if (stream) cudaStreamDestroy(stream);
  }
};

CudaModel::CudaModel(const ModelFiles& mf, int max_seq, const CudaOptions& opt) : p_(new Impl) {
  p_->c = mf.config();
  p_->max_seq = max_seq;
  p_->opt = opt;
  if (const char* e = std::getenv("TWINFER_MAX_PREFILL")) p_->opt.max_prefill = std::atoi(e);  // tests
  p_->opt.max_prefill = std::max(2, std::min(p_->opt.max_prefill, max_seq));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  p_->load(mf);
  p_->name = std::string("cuda-") + (opt.int8 ? "int8" : "fp16") + (opt.graph ? "+graph" : "") + " [" + prop.name +
             "]";
  std::fprintf(stderr, "cuda: %s (sm_%d%d, %d SMs), weights %.1f MB, total device memory %.1f MB\n", prop.name,
               prop.major, prop.minor, prop.multiProcessorCount, p_->weight_bytes / 1e6, p_->device_bytes / 1e6);
}

CudaModel::~CudaModel() = default;

const ModelConfig& CudaModel::config() const { return p_->c; }
int CudaModel::max_seq() const { return p_->max_seq; }
const char* CudaModel::name() const { return p_->name.c_str(); }
size_t CudaModel::weight_bytes() const { return p_->weight_bytes; }

const float* CudaModel::forward(const int* tokens, int n, int pos0) {
  p_->run(tokens, n, pos0);
  CUDA_CHECK(cudaMemcpyAsync(p_->h_logits, p_->logits, p_->c.vocab_size * sizeof(float), cudaMemcpyDeviceToHost,
                             p_->stream));
  CUDA_CHECK(cudaStreamSynchronize(p_->stream));
  return p_->h_logits;
}

int CudaModel::forward_argmax(const int* tokens, int n, int pos0) {
  p_->run(tokens, n, pos0);
  cuda::argmax_kernel<1024><<<1, 1024, 0, p_->stream>>>(p_->logits, p_->c.vocab_size, p_->d_argmax);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpyAsync(p_->h_argmax, p_->d_argmax, sizeof(int), cudaMemcpyDeviceToHost, p_->stream));
  CUDA_CHECK(cudaStreamSynchronize(p_->stream));
  return *p_->h_argmax;
}

}  // namespace tf
