#include "devblas/c_api/blas.h"
#include "devblas/benchmarking/bench.h"
#include "devblas/kernels/naive_gemm.h"
#include "devblas/kernels/tiled_gemm.h"
#include "devblas/types/config.h"
#include "devblas/types/layout.h"

using namespace devblas;

namespace types = internal::types;

extern "C" void naive_igemm_ijk(devblas_layout_t layout, const int *A,
                                const int *B, int *C, int M, int N, int K,
                                int lda, int ldb, int ldc) {
  internal::naive_gemm_ijk<int>(types::layout_to_cpp(layout), A, B, C, M, N, K,
                                lda, ldb, ldc);
}

extern "C" void naive_igemm_kij(devblas_layout_t layout, const int *A,
                                const int *B, int *C, int M, int N, int K,
                                int lda, int ldb, int ldc) {
  internal::naive_gemm_kij<int>(types::layout_to_cpp(layout), A, B, C, M, N, K,
                                lda, ldb, ldc);
}

extern "C" void naive_sgemm_ijk(devblas_layout_t layout, const float *A,
                                const float *B, float *C, int M, int N, int K,
                                int lda, int ldb, int ldc) {
  internal::naive_gemm_ijk<float>(types::layout_to_cpp(layout), A, B, C, M, N,
                                  K, lda, ldb, ldc);
}

extern "C" void naive_sgemm_kij(devblas_layout_t layout, const float *A,
                                const float *B, float *C, int M, int N, int K,
                                int lda, int ldb, int ldc) {
  internal::naive_gemm_kij<float>(types::layout_to_cpp(layout), A, B, C, M, N,
                                  K, lda, ldb, ldc);
}

extern "C" void tiled_sgemm(devblas_layout_t layout, const float *A, const float *B,
                 float *C, int M, int N, int K, int lda, int ldb, int ldc,
                 int tile_size) {
  internal::types::GemmConfig config =
      internal::types::GemmConfig(internal::types::layout_to_cpp(layout), M, N,
                                  K, lda, ldb, ldc, tile_size);
  internal::tiled_gemm<float>(A, B, C, config);
}

extern "C" void tiled_igemm(devblas_layout_t layout, const int *A, const int *B,
                 int *C, int M, int N, int K, int lda, int ldb, int ldc,
                 int tile_size) {
  internal::types::GemmConfig config =
      internal::types::GemmConfig(internal::types::layout_to_cpp(layout), M, N,
                                  K, lda, ldb, ldc, tile_size);
  internal::tiled_gemm<int>(A, B, C, config);
}

extern "C" void bench_igemm(devblas_igemm_fn fn, const char *name,
                            int warmup_iters, int iter, devblas_layout_t layout,
                            int M, int N, int K, int lda, int ldb, int ldc) {
  internal::bench::benchmark_gemm<int>(name ? name : "default_naive_igemm",
                                       warmup_iters, fn, layout, iter, M, N, K,
                                       lda, ldb, ldc);
}

extern "C" void bench_sgemm(devblas_sgemm_fn fn, const char *name,
                            int warmup_iters, int iter, devblas_layout_t layout,
                            int M, int N, int K, int lda, int ldb, int ldc) {
  internal::bench::benchmark_gemm<float>(name ? name : "default_naive_sgemm",
                                         warmup_iters, fn, layout, iter, M, N,
                                         K, lda, ldb, ldc);
}

