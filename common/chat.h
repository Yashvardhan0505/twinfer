// Multi-turn chat on top of generate(). The KV cache is kept between turns, so each turn only
// feeds the new text (plus the last token of the previous reply if that reply was cut off).
#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"

namespace tf {

class ChatSession {
 public:
  ChatSession(Engine& eng, const Tokenizer& tok, std::string system)
      : eng_(eng), tok_(tok), system_(std::move(system)) {}

  // One user turn: feeds the new text and generates up to max_new tokens. If the conversation
  // would not fit in the context it starts over (on_reset is called first).
  GenStats turn(Sampler& sampler, const std::string& user, int max_new, const std::vector<int>& stop_ids,
                const std::function<void(int)>& on_token, const std::function<void()>& on_reset = nullptr) {
    std::vector<int> ids = turn_ids(user, first_);
    if (carry_ >= 0) ids.insert(ids.begin(), carry_);
    if (pos_ + static_cast<int>(ids.size()) + max_new >= eng_.max_seq()) {
      if (on_reset) on_reset();
      pos_ = 0;
      carry_ = -1;
      ids = turn_ids(user, true);
    }
    first_ = false;
    carry_ = -1;
    int pending = -1;
    GenStats st = generate(eng_, sampler, ids, pos_, max_new, stop_ids, on_token, &pending);
    // A reply that ended on a stop token is closed by the next turn's "<|im_end|>". One that ran
    // into max_new was shown in full, but its last token never went through the model: keep it.
    if (!st.hit_stop && st.gen_tokens > 0) carry_ = pending;
    return st;
  }

  int pos() const { return pos_; }      // tokens in the KV cache
  int carry() const { return carry_; }  // last shown reply token not fed yet, or -1

  // The text of a turn, as tokens.
  std::vector<int> turn_ids(const std::string& user, bool first) const {
    if (first) return tok_.encode(chatml_prompt({{"user", user}}, system_));
    return tok_.encode("<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n");
  }

 private:
  Engine& eng_;
  const Tokenizer& tok_;
  std::string system_;
  int pos_ = 0;
  bool first_ = true;
  int carry_ = -1;
};

}  // namespace tf
