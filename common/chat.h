// Multi-turn chat on top of generate(). The KV cache is kept between turns, so each turn only
// feeds the new text (plus the last token of the previous reply if that reply was cut off).
// With chatml = false (--raw, or a model without ChatML tokens) there is no template: every line
// is appended to the text so far and the model continues it.
#pragma once
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"
#include "json.h"

namespace tf {

// The system prompt a ChatML chat template inserts when the conversation doesn't start with one
// (SmolLM2: "You are a helpful AI assistant named SmolLM, trained by Hugging Face"; Qwen2.5:
// "You are Qwen, created by Alibaba Cloud. ..."): the literal text between "<|im_start|>system\n"
// and "<|im_end|>" in the Jinja source. Empty if the template has none.
inline std::string template_system_prompt(const std::string& tmpl) {
  const std::string open = "<|im_start|>system", close = "<|im_end|>";
  for (size_t at = tmpl.find(open); at != std::string::npos; at = tmpl.find(open, at + 1)) {
    size_t b = at + open.size();
    if (tmpl.compare(b, 1, "\n") == 0) b += 1;        // a newline inside the Jinja string literal
    else if (tmpl.compare(b, 2, "\\n") == 0) b += 2;  // or its \n escape (how Qwen2.5 writes it)
    else continue;
    const size_t e = tmpl.find(close, b);
    if (e == std::string::npos) break;
    std::string s = tmpl.substr(b, e - b);
    if (s.empty() || s.find_first_of("'\"{}") != std::string::npos) continue;  // template code, not text
    for (size_t p = s.find("\\n"); p != std::string::npos; p = s.find("\\n", p + 1)) s.replace(p, 2, "\n");
    return s;
  }
  return "";
}

// Default system prompt of the model in `dir`, from chat_template.jinja or tokenizer_config.json
// (where Hugging Face keeps the chat template). Empty if neither has one.
inline std::string model_system_prompt(const std::string& dir) {
  try {
    if (file_exists(dir + "/chat_template.jinja")) return template_system_prompt(read_text_file(dir + "/chat_template.jinja"));
    if (!file_exists(dir + "/tokenizer_config.json")) return "";
    const Json cfg = Json::parse(read_text_file(dir + "/tokenizer_config.json"));
    const Json* t = cfg.get("chat_template");
    if (t && t->is_string()) return template_system_prompt(t->str);
    if (t && t->is_array())  // [{"name": "default", "template": "..."}, {"name": "tool_use", ...}]
      for (const Json& x : t->arr)
        if (x.string_or("name", "") == "default") return template_system_prompt(x.string_or("template", ""));
  } catch (const std::exception&) {
    // unreadable metadata only costs the default system prompt
  }
  return "";
}

class ChatSession {
 public:
  ChatSession(Engine& eng, const Tokenizer& tok, std::string system, bool chatml = true)
      : eng_(eng), tok_(tok), system_(std::move(system)), chatml_(chatml) {}

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
    if (!chatml_) return tok_.encode(first ? user : "\n" + user);
    if (first) return tok_.encode(chatml_prompt({{"user", user}}, system_));
    return tok_.encode("<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n");
  }

 private:
  Engine& eng_;
  const Tokenizer& tok_;
  std::string system_;
  bool chatml_;
  int pos_ = 0;
  bool first_ = true;
  int carry_ = -1;
};

}  // namespace tf
