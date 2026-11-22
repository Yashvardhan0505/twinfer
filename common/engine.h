// Backend-agnostic engine interface and the generation loop shared by the CPU and CUDA CLIs.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_files.h"
#include "sampler.h"
#include "tokenizer.h"

namespace tf {

class Engine {
 public:
  virtual ~Engine() = default;
  virtual const ModelConfig& config() const = 0;
  virtual int max_seq() const = 0;
  virtual const char* name() const = 0;

  // Run tokens[0..n) at positions [pos0, pos0 + n) and return the logits of the last token
  // (host memory, config().vocab_size floats). The KV cache for those positions is filled.
  virtual const float* forward(const int* tokens, int n, int pos0) = 0;

  // Same, but returns argmax(logits). Backends can override this to avoid copying logits.
  virtual int forward_argmax(const int* tokens, int n, int pos0) {
    return Sampler::argmax(forward(tokens, n, pos0), config().vocab_size);
  }
};

struct GenStats {
  int prompt_tokens = 0;
  double prefill_s = 0.0;
  int gen_tokens = 0;
  double decode_s = 0.0;
  bool hit_stop = false;
  double prefill_tps() const { return prefill_s > 0 ? prompt_tokens / prefill_s : 0.0; }
  double decode_tps() const { return decode_s > 0 ? gen_tokens / decode_s : 0.0; }
};

inline double now_s() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// Prefill `prompt` starting at position `pos`, then sample up to max_new tokens.
// `pos` is advanced past everything that was fed to the model. The final sampled token
// (e.g. the stop token) is returned in `pending` and has NOT been fed yet.
inline GenStats generate(Engine& eng, Sampler& sampler, const std::vector<int>& prompt, int& pos, int max_new,
                         const std::vector<int>& stop_ids, const std::function<void(int)>& on_token,
                         int* pending = nullptr) {
  GenStats st;
  st.prompt_tokens = static_cast<int>(prompt.size());
  const int vocab = eng.config().vocab_size;
  auto pick = [&](const int* toks, int n, int p) {
    if (sampler.greedy()) return eng.forward_argmax(toks, n, p);
    return sampler.sample(eng.forward(toks, n, p), vocab);
  };
  if (pos + static_cast<int>(prompt.size()) >= eng.max_seq()) throw std::runtime_error("prompt exceeds context");
  double t0 = now_s();
  int next = pick(prompt.data(), static_cast<int>(prompt.size()), pos);
  pos += static_cast<int>(prompt.size());
  st.prefill_s = now_s() - t0;

  t0 = now_s();
  for (int i = 0; i < max_new; ++i) {
    bool stop = false;
    for (int s : stop_ids) stop |= (next == s);
    if (stop) { st.hit_stop = true; break; }
    on_token(next);
    st.gen_tokens++;
    if (i + 1 == max_new || pos + 1 >= eng.max_seq()) { next = -1; break; }
    next = pick(&next, 1, pos);
    pos += 1;
  }
  st.decode_s = now_s() - t0;
  if (pending) *pending = next;
  return st;
}

// Stop tokens: config eos ids plus the usual chat end markers if the tokenizer has them.
inline std::vector<int> default_stop_ids(const ModelConfig& c, const Tokenizer& tok) {
  std::vector<int> ids = c.eos_ids;
  for (const char* s : {"<|im_end|>", "<|endoftext|>", "</s>"}) {
    int id = tok.token_id(s);
    if (id >= 0) ids.push_back(id);
  }
  return ids;
}

// RoPE table shared by all backends so they agree bit-for-bit on cos/sin.
// Matches Hugging Face: inv_freq and angle are float32, rotate_half pairing (i, i + head_dim/2).
inline void rope_table(const ModelConfig& c, int max_seq, std::vector<float>& cos_t, std::vector<float>& sin_t) {
  const int half = c.head_dim / 2;
  cos_t.resize(static_cast<size_t>(max_seq) * half);
  sin_t.resize(static_cast<size_t>(max_seq) * half);
  for (int i = 0; i < half; ++i) {
    const float inv = static_cast<float>(1.0 / std::pow(static_cast<double>(c.rope_theta),
                                                        static_cast<double>(2 * i) / c.head_dim));
    for (int p = 0; p < max_seq; ++p) {
      const float ang = static_cast<float>(p) * inv;
      cos_t[static_cast<size_t>(p) * half + i] = static_cast<float>(std::cos(static_cast<double>(ang)));
      sin_t[static_cast<size_t>(p) * half + i] = static_cast<float>(std::sin(static_cast<double>(ang)));
    }
  }
}

}  // namespace tf
