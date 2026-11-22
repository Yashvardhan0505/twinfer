// CUDA backend (fp16, or int8 weights).
// prefill (n > 1): wmma gemm + flash attention. decode (n == 1): fused gemv kernels,
// captured once as a CUDA graph.
#pragma once
#include <memory>

#include "../common/engine.h"

namespace tf {

struct CudaOptions {
  bool int8 = false;      // weight-only int8 with per-row scales
  bool graph = true;      // capture/replay the decode step as a CUDA graph
  int max_prefill = 512;  // tokens per prefill chunk
};

class CudaModel : public Engine {
 public:
  CudaModel(const ModelFiles& mf, int max_seq, const CudaOptions& opt);
  ~CudaModel() override;

  const ModelConfig& config() const override;
  int max_seq() const override;
  const char* name() const override;
  const float* forward(const int* tokens, int n, int pos0) override;
  int forward_argmax(const int* tokens, int n, int pos0) override;

  size_t weight_bytes() const;  // bytes streamed per decode step (all matrices)

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace tf
