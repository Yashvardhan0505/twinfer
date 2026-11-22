// Plain fp32 Llama forward pass. Kept simple on purpose since everything else is checked against it.
#pragma once
#include <vector>

#include "../common/engine.h"

namespace tf {

class CpuModel : public Engine {
 public:
  CpuModel(const ModelFiles& mf, int max_seq);

  const ModelConfig& config() const override { return c_; }
  int max_seq() const override { return max_seq_; }
  const char* name() const override { return "cpu-fp32"; }
  const float* forward(const int* tokens, int n, int pos0) override;

  // One token; returns logits.
  const float* step(int token, int pos);

 private:
  struct Layer {
    std::vector<float> attn_norm, wq, wk, wv, bq, bk, bv, wo, ffn_norm, w_gate, w_up, w_down;
  };
  ModelConfig c_;
  int max_seq_;
  std::vector<Layer> layers_;
  std::vector<float> embed_, final_norm_, lm_head_;
  std::vector<float> k_cache_, v_cache_;  // [layer][pos][kv_dim]
  std::vector<float> rope_cos_, rope_sin_;
  std::vector<float> x_, xb_, q_, k_, v_, att_out_, scores_, hb_, hb2_, logits_;
};

}  // namespace tf
