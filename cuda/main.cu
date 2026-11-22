#include "../common/cli.h"
#include "cuda_model.h"

int main(int argc, char** argv) {
  return tf::run_cli(argc, argv, /*cuda=*/true, [](const tf::ModelFiles& mf, const tf::Options& o) {
    tf::CudaOptions co;
    co.int8 = o.int8;
    co.graph = o.cuda_graph;
    return std::unique_ptr<tf::Engine>(new tf::CudaModel(mf, o.ctx, co));
  });
}
