// Just enough cuBLAS for bench_kernels.cu to compile under the emulator (GemmEx fp16 -> fp32 only).
#pragma once
#include "cuda_fp16.h"
#include "cuda_runtime.h"

typedef enum { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_SUPPORTED = 15 } cublasStatus_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 } cublasOperation_t;
typedef enum { CUDA_R_32F = 0, CUDA_R_16F = 2 } cudaDataType_t;
typedef enum { CUBLAS_COMPUTE_32F = 68 } cublasComputeType_t;
typedef enum { CUBLAS_GEMM_DEFAULT = -1 } cublasGemmAlgo_t;
struct cublasContext {};
typedef cublasContext* cublasHandle_t;

inline cublasStatus_t cublasCreate(cublasHandle_t* h) { *h = new cublasContext; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasDestroy(cublasHandle_t h) { delete h; return CUBLAS_STATUS_SUCCESS; }

// Column-major C[m, n] = op(A)[m, k] . op(B)[k, n]
inline cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                   const void* alpha, const void* A, cudaDataType_t at, int lda, const void* B,
                                   cudaDataType_t bt, int ldb, const void* beta, void* C, cudaDataType_t ct, int ldc,
                                   cublasComputeType_t, cublasGemmAlgo_t) {
  if (at != CUDA_R_16F || bt != CUDA_R_16F || ct != CUDA_R_32F) return CUBLAS_STATUS_NOT_SUPPORTED;
  const __half* a = static_cast<const __half*>(A);
  const __half* b = static_cast<const __half*>(B);
  float* c = static_cast<float*>(C);
  const float al = *static_cast<const float*>(alpha), be = *static_cast<const float*>(beta);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      float acc = 0.0f;
      for (int p = 0; p < k; ++p) {
        const float x = __half2float(ta == CUBLAS_OP_N ? a[i + p * lda] : a[p + i * lda]);
        const float y = __half2float(tb == CUBLAS_OP_N ? b[p + j * ldb] : b[j + p * ldb]);
        acc += x * y;
      }
      c[i + j * ldc] = al * acc + (be != 0.0f ? be * c[i + j * ldc] : 0.0f);
    }
  return CUBLAS_STATUS_SUCCESS;
}
