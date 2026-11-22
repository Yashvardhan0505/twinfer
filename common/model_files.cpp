#include "model_files.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "half.h"
#include "json.h"

namespace tf {

// Files

std::string read_text_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool file_exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

class MappedFile {
 public:
  explicit MappedFile(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    if (::fstat(fd_, &st) != 0) throw std::runtime_error("cannot stat " + path);
    size_ = static_cast<size_t>(st.st_size);
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
    data_ = static_cast<const uint8_t*>(p);
  }
  ~MappedFile() {
    if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
    if (fd_ >= 0) ::close(fd_);
  }
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

// Config

int64_t TensorView::numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

int64_t ModelConfig::param_count() const {
  int64_t per_layer = static_cast<int64_t>(dim) * qkv_dim() + static_cast<int64_t>(q_dim()) * dim +
                      3LL * dim * hidden_dim + 2LL * dim;
  int64_t n = per_layer * n_layers + static_cast<int64_t>(vocab_size) * dim + dim;
  if (!tied_embeddings) n += static_cast<int64_t>(vocab_size) * dim;
  return n;
}

std::string ModelConfig::summary() const {
  std::ostringstream s;
  s << arch << ": " << n_layers << " layers, dim " << dim << ", heads " << n_heads << "/" << n_kv_heads
    << " (head_dim " << head_dim << "), ffn " << hidden_dim << ", vocab " << vocab_size << ", "
    << (param_count() / 1e6) << "M params" << (tied_embeddings ? ", tied embeddings" : "")
    << (qkv_bias ? ", qkv bias" : "");
  return s.str();
}

std::string ModelFiles::layer(int l, const char* suffix) {
  return "model.layers." + std::to_string(l) + "." + suffix;
}

std::string ModelFiles::lm_head_name() const {
  return config_.tied_embeddings ? "model.embed_tokens.weight" : "lm_head.weight";
}

void ModelFiles::load_config() {
  Json j = Json::parse(read_text_file(dir_ + "/config.json"));
  ModelConfig& c = config_;
  if (const Json* a = j.get("architectures"); a && a->is_array() && !a->arr.empty()) c.arch = a->arr[0].str;
  static const std::set<std::string> kSupported = {"LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM"};
  if (!kSupported.count(c.arch))
    throw std::runtime_error("unsupported architecture '" + c.arch + "' (supported: Llama, Qwen2, Mistral)");
  if (j.string_or("hidden_act", "silu") != "silu") throw std::runtime_error("only hidden_act=silu is supported");
  if (const Json* rs = j.get("rope_scaling"); rs && !rs->is_null()) {
    std::string t = rs->string_or("rope_type", rs->string_or("type", ""));
    if (t != "default") throw std::runtime_error("rope_scaling type '" + t + "' is not supported yet");
  }
  if (const Json* sw = j.get("sliding_window"); sw && sw->is_number() && j.bool_or("use_sliding_window", false))
    throw std::runtime_error("sliding-window attention is not supported");

  c.dim = static_cast<int>(j.at("hidden_size").num);
  c.n_layers = static_cast<int>(j.at("num_hidden_layers").num);
  c.n_heads = static_cast<int>(j.at("num_attention_heads").num);
  c.n_kv_heads = static_cast<int>(j.number_or("num_key_value_heads", c.n_heads));
  c.head_dim = static_cast<int>(j.number_or("head_dim", c.dim / c.n_heads));
  c.hidden_dim = static_cast<int>(j.at("intermediate_size").num);
  c.vocab_size = static_cast<int>(j.at("vocab_size").num);
  c.max_pos = static_cast<int>(j.number_or("max_position_embeddings", 2048));
  c.rope_theta = static_cast<float>(j.number_or("rope_theta", 10000.0));
  c.norm_eps = static_cast<float>(j.number_or("rms_norm_eps", 1e-6));
  c.tied_embeddings = j.bool_or("tie_word_embeddings", false);
  if (const Json* e = j.get("eos_token_id")) {
    if (e->is_number()) c.eos_ids.push_back(static_cast<int>(e->num));
    if (e->is_array())
      for (const Json& x : e->arr) c.eos_ids.push_back(static_cast<int>(x.num));
  }
  if (c.n_heads % c.n_kv_heads != 0) throw std::runtime_error("n_heads must be a multiple of n_kv_heads");
  if (c.head_dim % 2 != 0) throw std::runtime_error("head_dim must be even");
}

// Safetensors

void ModelFiles::open_safetensors(const std::string& path) {
  auto f = std::make_unique<MappedFile>(path);
  if (f->size() < 8) throw std::runtime_error("truncated safetensors file " + path);
  uint64_t hlen = 0;
  std::memcpy(&hlen, f->data(), 8);  // little-endian
  if (8 + hlen > f->size()) throw std::runtime_error("bad safetensors header in " + path);
  const char* hb = reinterpret_cast<const char*>(f->data() + 8);
  Json header = Json::parse(hb, hb + hlen);
  const uint8_t* base = f->data() + 8 + hlen;
  const size_t data_size = f->size() - 8 - hlen;
  for (const auto& kv : header.obj) {
    if (kv.first == "__metadata__") continue;
    const Json& t = kv.second;
    TensorView v;
    v.dtype = t.at("dtype").str;
    for (const Json& d : t.at("shape").arr) v.shape.push_back(static_cast<int64_t>(d.num));
    const Json& off = t.at("data_offsets");
    size_t b = static_cast<size_t>(off.arr.at(0).num), e = static_cast<size_t>(off.arr.at(1).num);
    if (e < b || e > data_size) throw std::runtime_error("bad offsets for tensor " + kv.first);
    v.data = base + b;
    v.nbytes = e - b;
    tensors_[kv.first] = v;
  }
  files_.push_back(std::move(f));
}

ModelFiles::ModelFiles(const std::string& dir) : dir_(dir) {
  while (dir_.size() > 1 && dir_.back() == '/') dir_.pop_back();
  load_config();
  const std::string single = dir_ + "/model.safetensors";
  const std::string index = dir_ + "/model.safetensors.index.json";
  if (file_exists(single)) {
    open_safetensors(single);
  } else if (file_exists(index)) {
    Json j = Json::parse(read_text_file(index));
    std::set<std::string> shards;
    for (const auto& kv : j.at("weight_map").obj) shards.insert(kv.second.str);
    for (const auto& s : shards) open_safetensors(dir_ + "/" + s);
  } else {
    throw std::runtime_error("no model.safetensors(.index.json) in " + dir_);
  }
  config_.qkv_bias = find(layer(0, "self_attn.q_proj.bias")) != nullptr;
  if (find(layer(0, "mlp.gate_proj.bias")) || find(layer(0, "self_attn.o_proj.bias")))
    throw std::runtime_error("MLP / output-projection biases are not supported");
  if (!config_.tied_embeddings && !find("lm_head.weight")) config_.tied_embeddings = true;
}

ModelFiles::~ModelFiles() = default;

const TensorView* ModelFiles::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& ModelFiles::get(const std::string& name) const {
  const TensorView* t = find(name);
  if (!t) throw std::runtime_error("missing tensor " + name);
  return *t;
}

void ModelFiles::read_f32(const std::string& name, const std::vector<int64_t>& expect, float* dst) const {
  const TensorView& t = get(name);
  if (t.shape != expect) {
    std::ostringstream s;
    s << "shape mismatch for " << name << ": got [";
    for (size_t i = 0; i < t.shape.size(); ++i) s << (i ? "," : "") << t.shape[i];
    s << "] expected [";
    for (size_t i = 0; i < expect.size(); ++i) s << (i ? "," : "") << expect[i];
    s << "]";
    throw std::runtime_error(s.str());
  }
  const int64_t n = t.numel();
  if (t.dtype == "F32") {
    if (t.nbytes != static_cast<size_t>(n) * 4) throw std::runtime_error("size mismatch for " + name);
    std::memcpy(dst, t.data, t.nbytes);
  } else if (t.dtype == "BF16" || t.dtype == "F16") {
    if (t.nbytes != static_cast<size_t>(n) * 2) throw std::runtime_error("size mismatch for " + name);
    const bool bf = t.dtype == "BF16";
    for (int64_t i = 0; i < n; ++i) {
      uint16_t h;
      std::memcpy(&h, t.data + 2 * i, 2);
      dst[i] = bf ? bf16_to_f32(h) : f16_to_f32(h);
    }
  } else {
    throw std::runtime_error("unsupported dtype " + t.dtype + " for " + name);
  }
}

std::vector<float> ModelFiles::f32(const std::string& name, const std::vector<int64_t>& expect) const {
  int64_t n = 1;
  for (int64_t d : expect) n *= d;
  std::vector<float> v(static_cast<size_t>(n));
  read_f32(name, expect, v.data());
  return v;
}

}  // namespace tf
