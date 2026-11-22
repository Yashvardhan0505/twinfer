#include "../common/cli.h"
#include "cpu_model.h"

int main(int argc, char** argv) {
  return tf::run_cli(argc, argv, /*cuda=*/false, [](const tf::ModelFiles& mf, const tf::Options& o) {
    return std::unique_ptr<tf::Engine>(new tf::CpuModel(mf, o.ctx));
  });
}
