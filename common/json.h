// Small JSON parser, just enough for config.json, tokenizer.json and safetensors headers.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tf {

struct Json {
  enum Type { Null, Bool, Number, String, Array, Object };
  Type type = Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  bool is_null() const { return type == Null; }
  bool is_number() const { return type == Number; }
  bool is_string() const { return type == String; }
  bool is_array() const { return type == Array; }
  bool is_object() const { return type == Object; }

  const Json* get(const std::string& key) const {
    if (type != Object) return nullptr;
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  const Json& at(const std::string& key) const {
    const Json* p = get(key);
    if (!p) throw std::runtime_error("json: missing key '" + key + "'");
    return *p;
  }
  double number_or(const std::string& key, double def) const {
    const Json* p = get(key);
    return (p && p->type == Number) ? p->num : def;
  }
  bool bool_or(const std::string& key, bool def) const {
    const Json* p = get(key);
    return (p && p->type == Bool) ? p->b : def;
  }
  std::string string_or(const std::string& key, const std::string& def) const {
    const Json* p = get(key);
    return (p && p->type == String) ? p->str : def;
  }

  static Json parse(const char* begin, const char* end);
  static Json parse(const std::string& s) { return parse(s.data(), s.data() + s.size()); }
};

namespace detail {

inline void put_utf8(std::string& s, uint32_t cp) {
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

class JsonParser {
 public:
  JsonParser(const char* b, const char* e) : p_(b), end_(e) {}

  Json document() {
    Json v = value();
    ws();
    if (p_ != end_) fail("trailing characters");
    return v;
  }

 private:
  const char* p_;
  const char* end_;

  [[noreturn]] void fail(const char* msg) { throw std::runtime_error(std::string("json parse error: ") + msg); }

  void ws() {
    while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
  }

  void literal(const char* lit) {
    size_t n = std::strlen(lit);
    if (static_cast<size_t>(end_ - p_) < n || std::memcmp(p_, lit, n) != 0) fail("bad literal");
    p_ += n;
  }

  Json value() {
    ws();
    if (p_ >= end_) fail("unexpected end of input");
    Json j;
    switch (*p_) {
      case '{': return object();
      case '[': return array();
      case '"': j.type = Json::String; j.str = string(); return j;
      case 't': literal("true"); j.type = Json::Bool; j.b = true; return j;
      case 'f': literal("false"); j.type = Json::Bool; j.b = false; return j;
      case 'n': literal("null"); return j;
      default: return number();
    }
  }

  Json number() {
    // Copy into a small buffer so strtod never reads past end_.
    char buf[64];
    size_t n = 0;
    while (p_ + n < end_ && n < sizeof(buf) - 1) {
      char c = p_[n];
      if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
        buf[n++] = c;
      } else {
        break;
      }
    }
    buf[n] = 0;
    char* e = nullptr;
    double v = std::strtod(buf, &e);
    if (e == buf) fail("bad number");
    p_ += (e - buf);
    Json j;
    j.type = Json::Number;
    j.num = v;
    return j;
  }

  uint32_t hex4() {
    if (end_ - p_ < 4) fail("bad \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = *p_++;
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
      else fail("bad hex digit");
    }
    return v;
  }

  std::string string() {
    ++p_;  // opening quote
    std::string s;
    for (;;) {
      const char* run = p_;
      while (p_ < end_ && *p_ != '"' && *p_ != '\\') ++p_;
      s.append(run, p_);
      if (p_ >= end_) fail("unterminated string");
      if (*p_ == '"') { ++p_; return s; }
      ++p_;  // backslash
      if (p_ >= end_) fail("bad escape");
      char e = *p_++;
      switch (e) {
        case '"': s.push_back('"'); break;
        case '\\': s.push_back('\\'); break;
        case '/': s.push_back('/'); break;
        case 'b': s.push_back('\b'); break;
        case 'f': s.push_back('\f'); break;
        case 'n': s.push_back('\n'); break;
        case 'r': s.push_back('\r'); break;
        case 't': s.push_back('\t'); break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
              p_ += 2;
              uint32_t lo = hex4();
              if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                put_utf8(s, 0xFFFD);
                cp = lo;
              }
            } else {
              cp = 0xFFFD;
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            cp = 0xFFFD;
          }
          put_utf8(s, cp);
          break;
        }
        default: fail("bad escape character");
      }
    }
  }

  Json array() {
    ++p_;
    Json j;
    j.type = Json::Array;
    ws();
    if (p_ < end_ && *p_ == ']') { ++p_; return j; }
    for (;;) {
      j.arr.push_back(value());
      ws();
      if (p_ < end_ && *p_ == ',') { ++p_; continue; }
      if (p_ < end_ && *p_ == ']') { ++p_; return j; }
      fail("expected ',' or ']'");
    }
  }

  Json object() {
    ++p_;
    Json j;
    j.type = Json::Object;
    ws();
    if (p_ < end_ && *p_ == '}') { ++p_; return j; }
    for (;;) {
      ws();
      if (p_ >= end_ || *p_ != '"') fail("expected object key");
      std::string key = string();
      ws();
      if (p_ >= end_ || *p_ != ':') fail("expected ':'");
      ++p_;
      Json v = value();
      j.obj.emplace_back(std::move(key), std::move(v));
      ws();
      if (p_ < end_ && *p_ == ',') { ++p_; continue; }
      if (p_ < end_ && *p_ == '}') { ++p_; return j; }
      fail("expected ',' or '}'");
    }
  }
};

}  // namespace detail

inline Json Json::parse(const char* begin, const char* end) {
  return detail::JsonParser(begin, end).document();
}

}  // namespace tf
