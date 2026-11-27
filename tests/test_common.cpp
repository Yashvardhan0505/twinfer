// Unit tests for the backend-independent code (generation loop, sampler, chat turns), run on the
// tiny random test model with the CPU engine. tests/run_tests.py --cpu runs it after creating the model:
//   ./build/test-common build/tiny
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/chat.h"
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

  std::printf("[sampler]\n");
  {
    auto sampler = [](float temp, float top_p) {
      tf::SamplerConfig c;
      c.temperature = temp;
      c.top_p = top_p;
      c.seed = 7;
      return tf::Sampler(c);
    };
    const int n = 559;
    std::vector<float> nan_logits(n, NAN), uniform(n, 0.0f), peaked(n, 0.0f);
    peaked[123] = 10.0f;
    check(throws([&] { sampler(0.6f, 0.9f).sample(nan_logits.data(), n); }), "NaN logits are an error, not a crash");
    std::vector<float> pinf(n, 0.0f), all_ninf(n, -INFINITY);
    pinf[5] = INFINITY;
    check(throws([&] { sampler(0.6f, 0.9f).sample(pinf.data(), n); }), "+inf logit is an error");
    check(throws([&] { sampler(0.6f, 0.9f).sample(all_ninf.data(), n); }), "all -inf logits are an error");
    bool in_range = true;
    tf::Sampler tiny_p = sampler(1e9f, 1e-5f);  // every probability under the nucleus cutoff
    for (int i = 0; i < 50; ++i) {
      const int t = tiny_p.sample(uniform.data(), n);
      in_range &= t >= 0 && t < n;
    }
    check(in_range, "tiny top-p on a flat distribution still returns a valid id");
    tf::Sampler sp = sampler(0.6f, 0.9f);
    bool always_top = true;
    for (int i = 0; i < 50; ++i) always_top &= sp.sample(peaked.data(), n) == 123;
    check(always_top, "nucleus of a peaked distribution is just the top token");
    const float tie[4] = {0.0f, 5.0f, 5.0f, 0.0f};
    tf::Sampler st = sampler(1.0f, 0.3f);
    bool lower_id = true;
    for (int i = 0; i < 20; ++i) lower_id &= st.sample(tie, 4) == 1;
    check(lower_id, "a tie at the top of the nucleus goes to the lower id");
    const float one = 3.0f;
    check(sampler(0.6f, 0.9f).sample(&one, 1) == 0, "single-entry vocabulary");
  }

  std::printf("[chat]\n");
  {
    // Two turns whose replies are cut off by max_new. Afterwards the KV cache has to hold exactly
    // turn1 + reply1 + turn2 + reply2 (minus reply2's last token, which is carried into turn 3):
    // a fresh engine fed that transcript must give bit-identical logits for the next token.
    const int max_new = 3;
    tf::CpuModel chat_eng(mf, 128);
    tf::ChatSession chat(chat_eng, tok, "be brief");
    std::vector<int> r1, r2;
    tf::GenStats s1 = chat.turn(greedy, "hello there", max_new, {}, [&](int t) { r1.push_back(t); });
    tf::GenStats s2 = chat.turn(greedy, "and then?", max_new, {}, [&](int t) { r2.push_back(t); });
    check(!s1.hit_stop && !s2.hit_stop && r1.size() == 3 && r2.size() == 3, "both replies ran into max_new");

    std::vector<int> transcript = chat.turn_ids("hello there", true);
    transcript.insert(transcript.end(), r1.begin(), r1.end());  // all 3 tokens the user saw
    const std::vector<int> t2 = chat.turn_ids("and then?", false);
    transcript.insert(transcript.end(), t2.begin(), t2.end());
    transcript.insert(transcript.end(), r2.begin(), r2.end() - 1);
    check(chat.pos() == static_cast<int>(transcript.size()), "KV cache length = shown transcript");
    check(chat.carry() == r2.back(), "last shown token of the cut-off reply is carried");

    tf::CpuModel fresh(mf, 128);
    transcript.push_back(r2.back());
    const int next = r2.back();  // what turn 3 would feed first
    const float* la = chat_eng.forward(&next, 1, chat.pos());
    std::vector<float> got(la, la + mf.config().vocab_size);
    const float* lb = fresh.forward(transcript.data(), static_cast<int>(transcript.size()), 0);
    std::vector<float> want(lb, lb + mf.config().vocab_size);
    check(got == want, "chat KV cache == fresh forward over the transcript (bit-identical)");
  }

  if (g_failures) {
    std::printf("%d test(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all tests passed\n");
  return 0;
}
