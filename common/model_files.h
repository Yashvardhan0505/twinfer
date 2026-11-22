// Reads a Hugging Face model folder as-is: config.json, tokenizer.json and *.safetensors (mmap'd).
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tf {

struct ModelConfig {
  std::string arch;      // e.g. LlamaForCausalLM
  int dim = 0;           // hidden_size
  int n_layers = 0;
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int hidden_dim = 0;    // intermediate_size
  int vocab_size = 0;
  int max_pos = 0;       // max_position_embeddings
  float rope_theta = 10000.0f;
  float norm_eps = 1e-5f;
  bool tied_embeddings = false;
  bool qkv_bias = false;  // Qwen2-style q/k/v projection bias (detected from tensors)
  std::vector<int> eos_ids;

  int q_dim() const { return n_heads * head_dim; }
  int kv_dim() const { return n_kv_heads * head_dim; }
  int qkv_dim() const { return q_dim() + 2 * kv_dim(); }
  int64_t param_count() const;
  std::string summary() const;
};

struct TensorView {
  std::string dtype;  // "BF16", "F16" or "F32"
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
  int64_t numel() const;
};

class MappedFile;

class ModelFiles {
 public:
  explicit ModelFiles(const std::string& dir);
  ~ModelFiles();
  ModelFiles(const ModelFiles&) = delete;
  ModelFiles& operator=(const ModelFiles&) = delete;

  const ModelConfig& config() const { return config_; }
  const std::string& dir() const { return dir_; }
  std::string tokenizer_path() const { return dir_ + "/tokenizer.json"; }

  const TensorView* find(const std::string& name) const;
  const TensorView& get(const std::string& name) const;  // throws if missing

  // Convert a tensor to fp32 (checks the expected shape).
  std::vector<float> f32(const std::string& name, const std::vector<int64_t>& expect_shape) const;
  void read_f32(const std::string& name, const std::vector<int64_t>& expect_shape, float* dst) const;

  // Tensor names.
  static std::string layer(int l, const char* suffix);
  std::string lm_head_name() const;

 private:
  void open_safetensors(const std::string& path);
  void load_config();

  std::string dir_;
  ModelConfig config_;
  std::vector<std::unique_ptr<MappedFile>> files_;
  std::map<std::string, TensorView> tensors_;
};

std::string read_text_file(const std::string& path);
bool file_exists(const std::string& path);

}  // namespace tf
