#include "cpu_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tf {
namespace {

// y[n] = W[n, d] . x[d]
void matvec(float* y, const float* W, const float* x, int n, int d) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    const float* w = W + static_cast<size_t>(i) * d;
    float s = 0.0f;
#pragma omp simd reduction(+ : s)
    for (int j = 0; j < d; ++j) s += w[j] * x[j];
    y[i] = s;
  }
}

void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float inv = 1.0f / std::sqrt(static_cast<float>(ss / n) + eps);
  for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

// HF rotate_half convention: pairs (i, i + half).
void rope(float* v, int n_heads, int head_dim, const float* cos_p, const float* sin_p) {
  const int half = head_dim / 2;
  for (int h = 0; h < n_heads; ++h) {
    float* x = v + h * head_dim;
    for (int i = 0; i < half; ++i) {
      const float a = x[i], b = x[i + half];
      x[i] = a * cos_p[i] - b * sin_p[i];
      x[i + half] = b * cos_p[i] + a * sin_p[i];
    }
  }
}

}  // namespace

CpuModel::CpuModel(const ModelFiles& mf, int max_seq) : c_(mf.config()), max_seq_(max_seq) {
  const int D = c_.dim, Q = c_.q_dim(), KV = c_.kv_dim(), F = c_.hidden_dim, V = c_.vocab_size;
  embed_ = mf.f32("model.embed_tokens.weight", {V, D});
  final_norm_ = mf.f32("model.norm.weight", {D});
  if (!c_.tied_embeddings) lm_head_ = mf.f32("lm_head.weight", {V, D});
  layers_.resize(static_cast<size_t>(c_.n_layers));
  for (int l = 0; l < c_.n_layers; ++l) {
    Layer& L = layers_[l];
    auto t = [&](const char* s) { return ModelFiles::layer(l, s); };
    L.attn_norm = mf.f32(t("input_layernorm.weight"), {D});
    L.wq = mf.f32(t("self_attn.q_proj.weight"), {Q, D});
    L.wk = mf.f32(t("self_attn.k_proj.weight"), {KV, D});
    L.wv = mf.f32(t("self_attn.v_proj.weight"), {KV, D});
    if (c_.qkv_bias) {
      L.bq = mf.f32(t("self_attn.q_proj.bias"), {Q});
      L.bk = mf.f32(t("self_attn.k_proj.bias"), {KV});
      L.bv = mf.f32(t("self_attn.v_proj.bias"), {KV});
    }
    L.wo = mf.f32(t("self_attn.o_proj.weight"), {D, Q});
    L.ffn_norm = mf.f32(t("post_attention_layernorm.weight"), {D});
    L.w_gate = mf.f32(t("mlp.gate_proj.weight"), {F, D});
    L.w_up = mf.f32(t("mlp.up_proj.weight"), {F, D});
    L.w_down = mf.f32(t("mlp.down_proj.weight"), {D, F});
  }
  const size_t cache = static_cast<size_t>(c_.n_layers) * max_seq_ * KV;
  k_cache_.assign(cache, 0.0f);
  v_cache_.assign(cache, 0.0f);
  rope_table(c_, max_seq_, rope_cos_, rope_sin_);
  x_.resize(D);
  xb_.resize(D);
  q_.resize(Q);
  k_.resize(KV);
  v_.resize(KV);
  att_out_.resize(Q);
  scores_.resize(max_seq_);
  hb_.resize(F);
  hb2_.resize(F);
  logits_.resize(V);
}

const float* CpuModel::forward(const int* tokens, int n, int pos0) {
  const float* out = nullptr;
  for (int i = 0; i < n; ++i) out = step(tokens[i], pos0 + i);
  return out;
}

const float* CpuModel::step(int token, int pos) {
  if (pos < 0 || pos >= max_seq_) throw std::runtime_error("position out of range");
  if (token < 0 || token >= c_.vocab_size) throw std::runtime_error("token id out of range");
  const int D = c_.dim, Q = c_.q_dim(), KV = c_.kv_dim(), F = c_.hidden_dim, hd = c_.head_dim;
  const int n_rep = c_.n_heads / c_.n_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const float* cos_p = &rope_cos_[static_cast<size_t>(pos) * (hd / 2)];
  const float* sin_p = &rope_sin_[static_cast<size_t>(pos) * (hd / 2)];

  std::copy_n(&embed_[static_cast<size_t>(token) * D], D, x_.data());

  for (int l = 0; l < c_.n_layers; ++l) {
    const Layer& L = layers_[l];
    // attention
    rmsnorm(xb_.data(), x_.data(), L.attn_norm.data(), D, c_.norm_eps);
    matvec(q_.data(), L.wq.data(), xb_.data(), Q, D);
    matvec(k_.data(), L.wk.data(), xb_.data(), KV, D);
    matvec(v_.data(), L.wv.data(), xb_.data(), KV, D);
    if (c_.qkv_bias) {
      for (int i = 0; i < Q; ++i) q_[i] += L.bq[i];
      for (int i = 0; i < KV; ++i) { k_[i] += L.bk[i]; v_[i] += L.bv[i]; }
    }
    rope(q_.data(), c_.n_heads, hd, cos_p, sin_p);
    rope(k_.data(), c_.n_kv_heads, hd, cos_p, sin_p);
    float* kc = &k_cache_[(static_cast<size_t>(l) * max_seq_) * KV];
    float* vc = &v_cache_[(static_cast<size_t>(l) * max_seq_) * KV];
    std::copy_n(k_.data(), KV, kc + static_cast<size_t>(pos) * KV);
    std::copy_n(v_.data(), KV, vc + static_cast<size_t>(pos) * KV);

    for (int h = 0; h < c_.n_heads; ++h) {
      const int g = h / n_rep;
      const float* q = &q_[h * hd];
      float mx = -INFINITY;
      for (int t = 0; t <= pos; ++t) {
        const float* k = kc + static_cast<size_t>(t) * KV + g * hd;
        float s = 0.0f;
        for (int i = 0; i < hd; ++i) s += q[i] * k[i];
        scores_[t] = s * scale;
        mx = std::max(mx, scores_[t]);
      }
      float sum = 0.0f;
      for (int t = 0; t <= pos; ++t) {
        scores_[t] = std::exp(scores_[t] - mx);
        sum += scores_[t];
      }
      float* o = &att_out_[h * hd];
      std::fill_n(o, hd, 0.0f);
      for (int t = 0; t <= pos; ++t) {
        const float p = scores_[t] / sum;
        const float* v = vc + static_cast<size_t>(t) * KV + g * hd;
        for (int i = 0; i < hd; ++i) o[i] += p * v[i];
      }
    }
    matvec(xb_.data(), L.wo.data(), att_out_.data(), D, Q);
    for (int i = 0; i < D; ++i) x_[i] += xb_[i];

    // MLP (SwiGLU)
    rmsnorm(xb_.data(), x_.data(), L.ffn_norm.data(), D, c_.norm_eps);
    matvec(hb_.data(), L.w_gate.data(), xb_.data(), F, D);
    matvec(hb2_.data(), L.w_up.data(), xb_.data(), F, D);
    for (int i = 0; i < F; ++i) {
      const float g = hb_[i];
      hb_[i] = g / (1.0f + std::exp(-g)) * hb2_[i];
    }
    matvec(xb_.data(), L.w_down.data(), hb_.data(), D, F);
    for (int i = 0; i < D; ++i) x_[i] += xb_[i];
  }

  rmsnorm(xb_.data(), x_.data(), final_norm_.data(), D, c_.norm_eps);
  const float* head = c_.tied_embeddings ? embed_.data() : lm_head_.data();
  matvec(logits_.data(), head, xb_.data(), c_.vocab_size, D);
  return logits_.data();
}

}  // namespace tf
