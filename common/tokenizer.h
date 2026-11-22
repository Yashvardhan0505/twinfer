// Byte-level BPE that reads an HF tokenizer.json.
// Same pipeline as HF tokenizers: split out special tokens -> Digits (optional) -> ByteLevel
// regex -> bytes to unicode -> BPE merges. The regexes are hand-written scanners (no regex
// library), with the \p{L} / \p{N} tables generated into unicode_tables.h.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tf {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json_path);

  // Encode text. Added/special tokens that appear literally in the text (e.g. "<|im_start|>")
  // are mapped to their ids, like HF does by default.
  std::vector<int> encode(const std::string& text) const;

  // Raw bytes of one token (special tokens return their literal content).
  const std::string& token_bytes(int id) const;
  std::string decode(const std::vector<int>& ids, bool skip_special = true) const;

  int token_id(const std::string& content) const;  // -1 if absent; accepts special-token content
  bool is_special(int id) const;
  int vocab_size() const { return static_cast<int>(id_to_bytes_.size()); }

  enum class Pattern { GPT2, Qwen2 };
  Pattern pattern() const { return pattern_; }
  bool individual_digits() const { return individual_digits_; }

  // Exposed for tests: the pre-tokenizer pieces (raw byte strings) of a text without special tokens.
  std::vector<std::string> pretokenize(const std::string& text) const;

 private:
  void encode_ordinary(const std::string& text, std::vector<int>& out) const;
  void bpe(const std::string& piece, std::vector<int>& out) const;

  std::unordered_map<std::string, int> vocab_;  // byte-level token string -> id
  std::unordered_map<uint64_t, std::pair<int, int>> merges_;  // (left,right) -> (rank, merged id)
  std::vector<std::string> id_to_bytes_;
  std::vector<bool> special_;
  std::vector<std::pair<std::string, int>> added_;  // literal content -> id
  int byte_to_id_[256];
  Pattern pattern_ = Pattern::GPT2;
  bool individual_digits_ = false;
  bool add_prefix_space_ = false;
  mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

}  // namespace tf
