// Token sampling (greedy, temperature, top-p) and the ChatML prompt template.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tf {

struct SamplerConfig {
  float temperature = 0.0f;  // 0 = greedy (argmax)
  float top_p = 0.9f;
  uint64_t seed = 1234;
};

class Sampler {
 public:
  explicit Sampler(const SamplerConfig& c) : cfg_(c), state_(c.seed ? c.seed : 0x9E3779B97F4A7C15ull) {}

  bool greedy() const { return cfg_.temperature <= 0.0f; }

  int sample(const float* logits, int n) {
    if (greedy()) return argmax(logits, n);
    // Softmax with temperature.
    float mx = logits[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, logits[i]);
    probs_.resize(static_cast<size_t>(n));
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      probs_[i] = std::exp((logits[i] - mx) / cfg_.temperature);
      sum += probs_[i];
    }
    for (int i = 0; i < n; ++i) probs_[i] = static_cast<float>(probs_[i] / sum);
    const float r = uniform();
    if (cfg_.top_p <= 0.0f || cfg_.top_p >= 1.0f) {
      float cdf = 0.0f;
      for (int i = 0; i < n; ++i) {
        cdf += probs_[i];
        if (r < cdf) return i;
      }
      return n - 1;
    }
    // Nucleus sampling: tokens below (1-p)/(n-1) can never be in the nucleus, skip them before sorting.
    const float cutoff = (1.0f - cfg_.top_p) / static_cast<float>(n - 1);
    cand_.clear();
    for (int i = 0; i < n; ++i)
      if (probs_[i] >= cutoff) cand_.emplace_back(probs_[i], i);
    std::sort(cand_.begin(), cand_.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    float cum = 0.0f;
    size_t last = cand_.size() - 1;
    for (size_t i = 0; i < cand_.size(); ++i) {
      cum += cand_[i].first;
      if (cum > cfg_.top_p) { last = i; break; }
    }
    const float target = r * cum;
    float cdf = 0.0f;
    for (size_t i = 0; i <= last; ++i) {
      cdf += cand_[i].first;
      if (target < cdf) return cand_[i].second;
    }
    return cand_[last].second;
  }

  static int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
      if (x[i] > x[best]) best = i;
    return best;
  }

 private:
  float uniform() {  // xorshift64*
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return static_cast<float>((state_ * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
  }

  SamplerConfig cfg_;
  uint64_t state_;
  std::vector<float> probs_;
  std::vector<std::pair<float, int>> cand_;
};

struct ChatMessage {
  std::string role;
  std::string content;
};

// ChatML, as used by SmolLM2-Instruct and Qwen2.5-Instruct.
inline std::string chatml_prompt(const std::vector<ChatMessage>& msgs, const std::string& default_system,
                                 bool add_generation_prompt = true) {
  std::string s;
  if (!default_system.empty() && (msgs.empty() || msgs[0].role != "system"))
    s += "<|im_start|>system\n" + default_system + "<|im_end|>\n";
  for (const auto& m : msgs) s += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
  if (add_generation_prompt) s += "<|im_start|>assistant\n";
  return s;
}

inline const char* kSmolLMSystemPrompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

}  // namespace tf
