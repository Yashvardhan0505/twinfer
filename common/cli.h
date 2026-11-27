// Command-line front end shared by the CPU and CUDA executables.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat.h"
#include "engine.h"
#include "platform.h"

namespace tf {

struct Options {
  std::string model;
  std::string prompt = "Explain what a GPU kernel is in two sentences.";
  std::string system = kSmolLMSystemPrompt;
  int max_new = 256;
  int ctx = 2048;
  float temperature = 0.6f;
  float top_p = 0.9f;
  uint64_t seed = 1234;
  bool raw = false;
  bool interactive = false;
  bool quiet = false;
  std::vector<int> tokens;   // --tokens: feed raw ids
  std::string dump_logits;   // --dump-logits FILE
  int prefill_chunk = 1;     // with --dump-logits
  std::string tokenize;      // --tokenize TEXT
  bool tokenize_mode = false;
  // CUDA-specific (ignored by the CPU build).
  bool int8 = false;
  bool cuda_graph = true;
  int bench_tokens = 0;      // --bench N: decode N tokens greedily and report throughput
};

inline void print_usage(const char* prog, bool cuda) {
  std::fprintf(stderr,
               "usage: %s -m MODEL_DIR [options]\n"
               "  -p, --prompt TEXT     prompt (wrapped in the chat template unless --raw)\n"
               "  -i, --interactive     multi-turn chat\n"
               "  -n, --max-new N       max new tokens (default 256)\n"
               "  --temp T              temperature, 0 = greedy (default 0.6)\n"
               "  --top-p P             nucleus sampling threshold (default 0.9)\n"
               "  --seed S              RNG seed\n"
               "  --ctx N               context length / KV cache size (default 2048)\n"
               "  --system TEXT         system prompt for chat mode\n"
               "  --raw                 plain completion, no chat template\n"
               "  --quiet               print only the generated text\n"
               "  --bench N             greedy-decode N tokens and report tokens/s\n"
               "testing:\n"
               "  --tokens 1,2,3        feed these token ids instead of a prompt\n"
               "  --dump-logits FILE    write float32 logits per chunk and exit\n"
               "  --prefill-chunk C     chunk size for --dump-logits (default 1)\n"
               "  --tokenize TEXT       print token ids and exit\n",
               prog);
  if (cuda)
    std::fprintf(stderr,
                 "cuda:\n"
                 "  --int8                weight-only int8 quantization (per-row scales)\n"
                 "  --no-graph            disable CUDA graph capture of the decode step\n");
}

inline std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) v.push_back(std::atoi(item.c_str()));
  return v;
}

inline Options parse_options(const std::vector<std::string>& args, bool cuda) {
  Options o;
  const int argc = static_cast<int>(args.size());
  const char* prog = argc > 0 ? args[0].c_str() : "twinfer";
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", args[i].c_str());
      std::exit(2);
    }
    return args[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string& a = args[i];
    if (a == "-m" || a == "--model") o.model = need(i);
    else if (a == "-p" || a == "--prompt") o.prompt = need(i);
    else if (a == "-n" || a == "--max-new") o.max_new = std::atoi(need(i).c_str());
    else if (a == "--temp") o.temperature = static_cast<float>(std::atof(need(i).c_str()));
    else if (a == "--top-p") o.top_p = static_cast<float>(std::atof(need(i).c_str()));
    else if (a == "--seed") o.seed = std::strtoull(need(i).c_str(), nullptr, 10);
    else if (a == "--ctx") o.ctx = std::atoi(need(i).c_str());
    else if (a == "--system") o.system = need(i);
    else if (a == "--raw") o.raw = true;
    else if (a == "-i" || a == "--interactive") o.interactive = true;
    else if (a == "--quiet") o.quiet = true;
    else if (a == "--tokens") o.tokens = parse_ids(need(i));
    else if (a == "--dump-logits") o.dump_logits = need(i);
    else if (a == "--prefill-chunk") o.prefill_chunk = std::max(1, std::atoi(need(i).c_str()));
    else if (a == "--tokenize") { o.tokenize = need(i); o.tokenize_mode = true; }
    else if (a == "--bench") o.bench_tokens = std::atoi(need(i).c_str());
    else if (cuda && a == "--int8") o.int8 = true;
    else if (cuda && a == "--no-graph") o.cuda_graph = false;
    else if (a == "-h" || a == "--help") { print_usage(prog, cuda); std::exit(0); }
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      print_usage(prog, cuda);
      std::exit(2);
    }
  }
  if (o.model.empty()) {
    print_usage(prog, cuda);
    std::exit(2);
  }
  return o;
}

using EngineFactory = std::function<std::unique_ptr<Engine>(const ModelFiles&, const Options&)>;

inline int run_cli(int argc, char** argv, bool cuda, const EngineFactory& make_engine) {
  setup_console();
  Options o = parse_options(utf8_args(argc, argv), cuda);
  try {
    ModelFiles mf(o.model);
    Tokenizer tok(mf.tokenizer_path());
    if (o.tokenize_mode) {
      std::vector<int> ids = tok.encode(o.tokenize);
      for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? "," : "", ids[i]);
      std::printf("\n");
      return 0;
    }
    const ModelConfig& c = mf.config();
    o.ctx = std::min(o.ctx, c.max_pos);
    if (!o.quiet) std::fprintf(stderr, "model: %s\n", c.summary().c_str());
    double t0 = now_s();
    std::unique_ptr<Engine> eng = make_engine(mf, o);
    if (!o.quiet) std::fprintf(stderr, "backend: %s, loaded in %.2fs, context %d\n", eng->name(), now_s() - t0, o.ctx);

    const bool chat = !o.raw && tok.token_id("<|im_start|>") >= 0;
    auto build_prompt = [&](const std::string& user) {
      if (!chat) return tok.encode(user);
      return tok.encode(chatml_prompt({{"user", user}}, o.system));
    };

    // testing: dump logits
    if (!o.dump_logits.empty()) {
      std::vector<int> ids = o.tokens.empty() ? build_prompt(o.prompt) : o.tokens;
      FILE* f = std::fopen(o.dump_logits.c_str(), "wb");
      if (!f) throw std::runtime_error("cannot write " + o.dump_logits);
      for (size_t i = 0; i < ids.size(); i += o.prefill_chunk) {
        int n = static_cast<int>(std::min<size_t>(o.prefill_chunk, ids.size() - i));
        const float* logits = eng->forward(&ids[i], n, static_cast<int>(i));
        std::fwrite(logits, sizeof(float), c.vocab_size, f);
      }
      std::fclose(f);
      return 0;
    }

    SamplerConfig sc;
    sc.temperature = o.temperature;
    sc.top_p = o.top_p;
    sc.seed = o.seed;
    Sampler sampler(sc);
    const std::vector<int> stops = default_stop_ids(c, tok);
    auto emit = [&](int id) {
      if (tok.is_special(id)) return;
      const std::string& b = tok.token_bytes(id);
      std::fwrite(b.data(), 1, b.size(), stdout);
      std::fflush(stdout);
    };
    auto report = [&](const GenStats& st) {
      if (o.quiet) return;
      std::fprintf(stderr, "\n[prefill %d tok, %.1f ms, %.1f tok/s | decode %d tok, %.1f ms, %.1f tok/s]\n",
                   st.prompt_tokens, st.prefill_s * 1e3, st.prefill_tps(), st.gen_tokens, st.decode_s * 1e3,
                   st.decode_tps());
    };

    // benchmark
    if (o.bench_tokens > 0) {
      SamplerConfig g;
      g.temperature = 0.0f;
      Sampler greedy(g);
      std::vector<int> ids = o.tokens.empty() ? build_prompt(o.prompt) : o.tokens;
      int pos = 0;
      // Warm-up run (builds CUDA graphs, warms caches), then the measured run.
      generate(*eng, greedy, ids, pos, 8, {}, [](int) {});
      pos = 0;
      GenStats st = generate(*eng, greedy, ids, pos, o.bench_tokens, {}, [](int) {});
      std::printf("prompt_tokens=%d prefill_tok_s=%.1f decode_tokens=%d decode_tok_s=%.1f ms_per_token=%.3f backend=%s\n",
                  st.prompt_tokens, st.prefill_tps(), st.gen_tokens, st.decode_tps(),
                  st.gen_tokens ? st.decode_s * 1e3 / st.gen_tokens : 0.0, eng->name());
      return 0;
    }

    // single prompt
    if (!o.interactive) {
      std::vector<int> ids = o.tokens.empty() ? build_prompt(o.prompt) : o.tokens;
      int pos = 0;
      GenStats st = generate(*eng, sampler, ids, pos, o.max_new, stops, emit);
      std::printf("\n");
      report(st);
      return 0;
    }

    // interactive chat (KV cache is kept across turns)
    ChatSession session(*eng, tok, o.system);
    std::string line;
    std::fprintf(stderr, "chat mode: type a message, empty line or Ctrl-D to quit\n");
    while (true) {
      std::fprintf(stderr, "\n> ");
      if (!std::getline(std::cin, line) || line.empty()) break;
      GenStats st = session.turn(sampler, line, o.max_new, stops, emit,
                                 [] { std::fprintf(stderr, "[context full, starting a new conversation]\n"); });
      report(st);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

}  // namespace tf
