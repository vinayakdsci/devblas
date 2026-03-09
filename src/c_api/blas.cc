#include "devblas/c_api/blas.h"
#include "devblas/benchmarking/bench.h"
#include "devblas/kernels/naive_gemm.h"
#include "devblas/kernels/tiled_gemm.h"
#include "devblas/types/config.h"
#include "devblas/types/layout.h"

using namespace devblas;

namespace types = internal::types;

extern "C" void naive_igemm_ijk(const int *A, const int *B, int *C,
                                devblas_gemm_config_t *config) {
  auto cppConfig = types::gemm_config_to_cpp(config);
  internal::naive_gemm_ijk<int>(A, B, C, std::move(cppConfig));
}

extern "C" void naive_igemm_kij(const int *A, const int *B, int *C,
                                devblas_gemm_config_t *config) {
  auto cppConfig = types::gemm_config_to_cpp(config);
  internal::naive_gemm_kij<int>(A, B, C, std::move(cppConfig));
}

extern "C" void naive_sgemm_ijk(const float *A, const float *B, float *C,
                                devblas_gemm_config_t *config) {
  auto cppConfig = types::gemm_config_to_cpp(config);
  internal::naive_gemm_ijk<float>(A, B, C, std::move(cppConfig));
}

extern "C" void naive_sgemm_kij(const float *A, const float *B, float *C,
                                devblas_gemm_config_t *config) {
  auto cppConfig = types::gemm_config_to_cpp(config);
  internal::naive_gemm_kij<float>(A, B, C, std::move(cppConfig));
}

extern "C" void tiled_sgemm(const float *A, const float *B, float *C,
                            devblas_gemm_config_t *config) {
  auto cppConfig = types::gemm_config_to_cpp(config);
  internal::tiled_gemm<float>(A, B, C, std::move(cppConfig));
}

extern "C" void tiled_igemm(const int *A, const int *B, int *C,
                            devblas_gemm_config_t *config) {
  auto cppConfig = types::gemm_config_to_cpp(config);
  internal::tiled_gemm<int>(A, B, C, std::move(cppConfig));
}

extern "C" void bench_igemm(devblas_igemm_fn fn, const char *name,
                            int warmup_iters, int iter,
                            devblas_gemm_config_t *config) {
  internal::bench::benchmark_gemm<int>(name ? name : "default_naive_igemm",
                                       warmup_iters, iter, fn, config);
}

extern "C" void bench_sgemm(devblas_sgemm_fn fn, const char *name,
                            int warmup_iters, int iter,
                            devblas_gemm_config_t *config) {
  internal::bench::benchmark_gemm<float>(name ? name : "default_naive_sgemm",
                                         warmup_iters, iter, fn, config);
}
