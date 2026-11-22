#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "json.h"
#include "model_files.h"
#include "unicode_tables.h"

namespace tf {
namespace {

// Unicode helpers

bool in_ranges(uint32_t cp, const uint32_t (*r)[2], int n) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (cp < r[mid][0]) hi = mid - 1;
    else if (cp > r[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

bool is_L(uint32_t c) {
  if (c < 128) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  return in_ranges(c, kLetterRanges, kNumLetterRanges);
}
bool is_N(uint32_t c) {
  if (c < 128) return c >= '0' && c <= '9';
  return in_ranges(c, kNumberRanges, kNumNumberRanges);
}
// Unicode White_Space property (what \s means in the HF regexes).
bool is_S(uint32_t c) {
  return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool is_other(uint32_t c) { return !is_S(c) && !is_L(c) && !is_N(c); }
bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }

// Decode UTF-8 into code points; offs[i] is the byte offset of code point i (offs[n] = size).
// Invalid bytes become U+FFFD but keep their original byte so the byte-level BPE is lossless.
void utf8_decode(const std::string& s, std::vector<uint32_t>& cps, std::vector<size_t>& offs) {
  cps.clear();
  offs.clear();
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  size_t i = 0, n = s.size();
  while (i < n) {
    offs.push_back(i);
    unsigned c = p[i];
    uint32_t cp = 0xFFFD;
    size_t len = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c >> 5) == 0x6 && i + 1 < n && (p[i + 1] & 0xC0) == 0x80) {
      cp = ((c & 0x1F) << 6) | (p[i + 1] & 0x3F);
      len = cp >= 0x80 ? 2 : 1;
      if (len == 1) cp = 0xFFFD;
    } else if ((c >> 4) == 0xE && i + 2 < n && (p[i + 1] & 0xC0) == 0x80 && (p[i + 2] & 0xC0) == 0x80) {
      cp = ((c & 0x0F) << 12) | ((p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F);
      len = (cp >= 0x800 && (cp < 0xD800 || cp > 0xDFFF)) ? 3 : 1;
      if (len == 1) cp = 0xFFFD;
    } else if ((c >> 3) == 0x1E && i + 3 < n && (p[i + 1] & 0xC0) == 0x80 && (p[i + 2] & 0xC0) == 0x80 &&
               (p[i + 3] & 0xC0) == 0x80) {
      cp = ((c & 0x07) << 18) | ((p[i + 1] & 0x3F) << 12) | ((p[i + 2] & 0x3F) << 6) | (p[i + 3] & 0x3F);
      len = (cp >= 0x10000 && cp <= 0x10FFFF) ? 4 : 1;
      if (len == 1) cp = 0xFFFD;
    }
    cps.push_back(cp);
    i += len;
  }
  offs.push_back(n);
}

// Pre-tokenizer regexes, implemented as hand-written scanners.

using Span = std::pair<size_t, size_t>;  // code-point index range [a, b)

size_t run_while(const std::vector<uint32_t>& cp, size_t i, size_t e, bool (*pred)(uint32_t)) {
  while (i < e && pred(cp[i])) ++i;
  return i;
}

// 's|'t|'re|'ve|'m|'ll|'d   (optionally case-insensitive). Returns match length or 0.
size_t contraction(const std::vector<uint32_t>& cp, size_t i, size_t e, bool ci) {
  if (cp[i] != '\'' || i + 1 >= e) return 0;
  auto low = [&](uint32_t c) -> uint32_t { return (ci && c >= 'A' && c <= 'Z') ? c + 32 : c; };
  uint32_t a = low(cp[i + 1]);
  if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
  if (i + 2 < e) {
    uint32_t b = low(cp[i + 2]);
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
  }
  return 0;
}

// \s+(?!\S)|\s+   starting at whitespace position i.
size_t whitespace_tail(const std::vector<uint32_t>& cp, size_t i, size_t e) {
  size_t j = run_while(cp, i, e, is_S);
  if (j < e && j - i >= 2) return j - 1;  // leave the last space to prefix the next word
  return j;
}

// GPT-2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
void split_gpt2(const std::vector<uint32_t>& cp, size_t s, size_t e, std::vector<Span>& out) {
  size_t i = s;
  while (i < e) {
    size_t j;
    if (size_t n = contraction(cp, i, e, false)) {
      j = i + n;
    } else {
      const uint32_t c = cp[i];
      const bool sp = c == ' ' && i + 1 < e;
      if (sp && is_L(cp[i + 1])) j = run_while(cp, i + 1, e, is_L);
      else if (is_L(c)) j = run_while(cp, i, e, is_L);
      else if (sp && is_N(cp[i + 1])) j = run_while(cp, i + 1, e, is_N);
      else if (is_N(c)) j = run_while(cp, i, e, is_N);
      else if (sp && is_other(cp[i + 1])) j = run_while(cp, i + 1, e, is_other);
      else if (is_other(c)) j = run_while(cp, i, e, is_other);
      else j = whitespace_tail(cp, i, e);
    }
    out.emplace_back(i, j);
    i = j;
  }
}

// Qwen2: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
void split_qwen2(const std::vector<uint32_t>& cp, size_t s, size_t e, std::vector<Span>& out) {
  size_t i = s;
  while (i < e) {
    size_t j = 0;
    const uint32_t c = cp[i];
    if (size_t n = contraction(cp, i, e, true)) {
      j = i + n;
    } else if (!is_crlf(c) && !is_L(c) && !is_N(c) && i + 1 < e && is_L(cp[i + 1])) {
      j = run_while(cp, i + 1, e, is_L);
    } else if (is_L(c)) {
      j = run_while(cp, i, e, is_L);
    } else if (is_N(c)) {
      j = i + 1;
    } else if ((c == ' ' && i + 1 < e && is_other(cp[i + 1])) || is_other(c)) {
      j = run_while(cp, is_other(c) ? i : i + 1, e, is_other);
      j = run_while(cp, j, e, is_crlf);
    } else {
      // c is whitespace.  \s*[\r\n]+ : up to and including the last CR/LF of the run.
      size_t q = run_while(cp, i, e, is_S);
      size_t last = q;
      for (size_t r = i; r < q; ++r)
        if (is_crlf(cp[r])) last = r;
      j = (last < q) ? last + 1 : whitespace_tail(cp, i, e);
    }
    out.emplace_back(i, j);
    i = j;
  }
}

const char* kGpt2Regex = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
const char* kQwen2Regex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|\\s+";

// GPT-2 bytes_to_unicode(): printable bytes map to themselves, the rest to U+0100...
void byte_to_codepoint(uint32_t table[256]) {
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    table[b] = printable ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
  }
}

void append_utf8(std::string& s, uint32_t cp) {
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

}  // namespace

// Loading

Tokenizer::Tokenizer(const std::string& path) {
  Json j = Json::parse(read_text_file(path));
  const Json& model = j.at("model");
  if (model.string_or("type", "BPE") != "BPE") throw std::runtime_error("only BPE tokenizers are supported");

  // Pre-tokenizer configuration.
  bool have_bytelevel_split = false;
  auto visit = [&](const Json& pt) {
    std::string t = pt.string_or("type", "");
    if (t == "Digits") {
      individual_digits_ = pt.bool_or("individual_digits", false);
    } else if (t == "ByteLevel") {
      add_prefix_space_ = pt.bool_or("add_prefix_space", false);
      if (pt.bool_or("use_regex", true)) { pattern_ = Pattern::GPT2; have_bytelevel_split = true; }
    } else if (t == "Split") {
      std::string re = pt.at("pattern").string_or("Regex", "");
      if (re == kGpt2Regex) pattern_ = Pattern::GPT2;
      else if (re == kQwen2Regex) pattern_ = Pattern::Qwen2;
      else throw std::runtime_error("unsupported Split pre-tokenizer regex: " + re);
      have_bytelevel_split = true;
    } else {
      throw std::runtime_error("unsupported pre-tokenizer type: " + t);
    }
  };
  const Json* pre = j.get("pre_tokenizer");
  if (!pre || pre->is_null()) throw std::runtime_error("tokenizer has no pre_tokenizer (not byte-level BPE?)");
  if (pre->string_or("type", "") == "Sequence") {
    for (const Json& pt : pre->at("pretokenizers").arr) visit(pt);
  } else {
    visit(*pre);
  }
  if (!have_bytelevel_split) throw std::runtime_error("expected a ByteLevel or Split pre-tokenizer");

  // Vocabulary.
  uint32_t b2u[256];
  byte_to_codepoint(b2u);
  std::unordered_map<uint32_t, int> u2b;
  for (int b = 0; b < 256; ++b) u2b[b2u[b]] = b;

  int max_id = -1;
  for (const auto& kv : model.at("vocab").obj) max_id = std::max(max_id, static_cast<int>(kv.second.num));
  const Json* added = j.get("added_tokens");
  if (added && added->is_array())
    for (const Json& a : added->arr) max_id = std::max(max_id, static_cast<int>(a.at("id").num));
  id_to_bytes_.assign(static_cast<size_t>(max_id + 1), std::string());
  special_.assign(static_cast<size_t>(max_id + 1), false);

  std::vector<uint32_t> cps;
  std::vector<size_t> offs;
  for (const auto& kv : model.at("vocab").obj) {
    const int id = static_cast<int>(kv.second.num);
    vocab_[kv.first] = id;
    utf8_decode(kv.first, cps, offs);
    std::string bytes;
    bool ok = true;
    for (uint32_t c : cps) {
      auto it = u2b.find(c);
      if (it == u2b.end()) { ok = false; break; }
      bytes.push_back(static_cast<char>(it->second));
    }
    id_to_bytes_[id] = ok ? bytes : kv.first;
  }
  for (int b = 0; b < 256; ++b) {
    std::string s;
    append_utf8(s, b2u[b]);
    auto it = vocab_.find(s);
    byte_to_id_[b] = it == vocab_.end() ? -1 : it->second;
  }

  if (added && added->is_array()) {
    for (const Json& a : added->arr) {
      const int id = static_cast<int>(a.at("id").num);
      const std::string content = a.at("content").str;
      added_.emplace_back(content, id);
      id_to_bytes_[id] = content;
      special_[id] = a.bool_or("special", false);
    }
  }

  // Merges: either "a b" strings or ["a", "b"] pairs (newer tokenizers versions).
  const Json& merges = model.at("merges");
  int rank = 0;
  for (const Json& m : merges.arr) {
    std::string a, b;
    if (m.is_string()) {
      size_t sp = m.str.find(' ');
      if (sp == std::string::npos) { ++rank; continue; }
      a = m.str.substr(0, sp);
      b = m.str.substr(sp + 1);
    } else if (m.is_array() && m.arr.size() == 2) {
      a = m.arr[0].str;
      b = m.arr[1].str;
    } else {
      ++rank;
      continue;
    }
    auto ia = vocab_.find(a), ib = vocab_.find(b), iab = vocab_.find(a + b);
    if (ia != vocab_.end() && ib != vocab_.end() && iab != vocab_.end()) {
      const uint64_t key = (static_cast<uint64_t>(ia->second) << 32) | static_cast<uint32_t>(ib->second);
      merges_.emplace(key, std::make_pair(rank, iab->second));  // keep the first (lowest) rank
    }
    ++rank;
  }
}

// Encoding

std::vector<std::string> Tokenizer::pretokenize(const std::string& text_in) const {
  std::string text = text_in;
  if (add_prefix_space_ && !text.empty() && text[0] != ' ') text = " " + text;
  std::vector<uint32_t> cp;
  std::vector<size_t> offs;
  utf8_decode(text, cp, offs);

  // Digits(individual_digits=true): every \p{N} char becomes its own piece.
  std::vector<Span> chunks;
  if (individual_digits_) {
    size_t i = 0, n = cp.size();
    while (i < n) {
      if (is_N(cp[i])) { chunks.emplace_back(i, i + 1); ++i; continue; }
      size_t j = i;
      while (j < n && !is_N(cp[j])) ++j;
      chunks.emplace_back(i, j);
      i = j;
    }
  } else {
    chunks.emplace_back(0, cp.size());
  }

  std::vector<Span> spans;
  for (const Span& c : chunks) {
    if (pattern_ == Pattern::Qwen2) split_qwen2(cp, c.first, c.second, spans);
    else split_gpt2(cp, c.first, c.second, spans);
  }
  std::vector<std::string> pieces;
  pieces.reserve(spans.size());
  for (const Span& s : spans) pieces.push_back(text.substr(offs[s.first], offs[s.second] - offs[s.first]));
  return pieces;
}

void Tokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
  auto hit = cache_.find(piece);
  if (hit != cache_.end()) {
    out.insert(out.end(), hit->second.begin(), hit->second.end());
    return;
  }
  std::vector<int> sym;
  sym.reserve(piece.size());
  for (unsigned char b : piece)
    if (byte_to_id_[b] >= 0) sym.push_back(byte_to_id_[b]);
  // Repeatedly apply the lowest-rank merge (leftmost on ties), like HF's BPE.
  while (sym.size() > 1) {
    int best_rank = INT_MAX, best_i = -1, best_id = -1;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      const uint64_t key = (static_cast<uint64_t>(sym[i]) << 32) | static_cast<uint32_t>(sym[i + 1]);
      auto it = merges_.find(key);
      if (it != merges_.end() && it->second.first < best_rank) {
        best_rank = it->second.first;
        best_i = static_cast<int>(i);
        best_id = it->second.second;
      }
    }
    if (best_i < 0) break;
    sym[best_i] = best_id;
    sym.erase(sym.begin() + best_i + 1);
  }
  if (cache_.size() < 200000) cache_.emplace(piece, sym);
  out.insert(out.end(), sym.begin(), sym.end());
}

void Tokenizer::encode_ordinary(const std::string& text, std::vector<int>& out) const {
  if (text.empty()) return;
  for (const std::string& piece : pretokenize(text)) bpe(piece, out);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  size_t pos = 0;
  while (pos < text.size()) {
    // Leftmost (then longest) added-token occurrence.
    size_t best_at = std::string::npos, best_len = 0;
    int best_id = -1;
    for (const auto& a : added_) {
      if (a.first.empty()) continue;
      size_t f = text.find(a.first, pos);
      if (f == std::string::npos) continue;
      if (f < best_at || (f == best_at && a.first.size() > best_len)) {
        best_at = f;
        best_len = a.first.size();
        best_id = a.second;
      }
    }
    const size_t seg_end = best_at == std::string::npos ? text.size() : best_at;
    encode_ordinary(text.substr(pos, seg_end - pos), ids);
    if (best_at == std::string::npos) break;
    ids.push_back(best_id);
    pos = best_at + best_len;
  }
  return ids;
}

// Decoding

const std::string& Tokenizer::token_bytes(int id) const {
  static const std::string kEmpty;
  if (id < 0 || id >= vocab_size()) return kEmpty;
  return id_to_bytes_[id];
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  std::string s;
  for (int id : ids) {
    if (skip_special && is_special(id)) continue;
    s += token_bytes(id);
  }
  return s;
}

int Tokenizer::token_id(const std::string& content) const {
  for (const auto& a : added_)
    if (a.first == content) return a.second;
  auto it = vocab_.find(content);
  return it == vocab_.end() ? -1 : it->second;
}

bool Tokenizer::is_special(int id) const { return id >= 0 && id < vocab_size() && special_[id]; }

}  // namespace tf
