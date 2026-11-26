// Unit tests for the backend-independent code (generation loop, sampler, chat turns), run on the
// tiny random test model with the CPU engine. tests/run_tests.py --cpu runs it after creating the model:
//   ./build/test-common build/tiny
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/engine.h"
#include "../cpu/cpu_model.h"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++g_failures;
}

bool throws(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s TINY_MODEL_DIR\n", argv[0]);
    return 2;
  }
  tf::ModelFiles mf(argv[1]);
  tf::Tokenizer tok(mf.tokenizer_path());
  tf::CpuModel eng(mf, 128);
  tf::SamplerConfig gc;
  gc.temperature = 0.0f;
  tf::Sampler greedy(gc);

  std::printf("[generate]\n");
  {
    int pos = 0;
    check(throws([&] { tf::generate(eng, greedy, {}, pos, 4, {}, [](int) {}); }),
          "empty prompt is an error, not a crash");
    check(throws([&] { eng.forward(nullptr, 0, 0); }), "CpuModel::forward with no tokens is an error");
  }

  if (g_failures) {
    std::printf("%d test(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all tests passed\n");
  return 0;
}
