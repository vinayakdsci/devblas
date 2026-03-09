#include "devblas/benchmarking/bench.h"
#include "devblas/benchmarking/utils.h"
#include "devblas/c_api/blas.h"
#include "devblas/types/layout.h"
#include <iostream>
#include <random>
#include <type_traits>

namespace {
template <typename T>
using uniform_distribution_selector =
    std::conditional_t<std::is_floating_point_v<T>,
                       std::uniform_real_distribution<T>,
                       std::uniform_int_distribution<T>>;
}

namespace devblas::internal {
namespace bench {

template <typename T>
void benchmark_gemm(const char *name, int warmup_iters, int iters, GemmFn<T> fn,
                    devblas_gemm_config_t *config) {

  if (!config) {
    throw std::runtime_error("Received NULL config!");
  }

  auto cppConfig = types::gemm_config_to_cpp(config);

  auto [M, N, K] = cppConfig.logicalDims();
  auto [lda, ldb, ldc] = cppConfig.leadingDims();

  std::random_device rd;
  std::mt19937 rng(rd());

  uniform_distribution_selector<T> dist(static_cast<T>(-5), static_cast<T>(5));

  std::vector<T> A;
  std::vector<T> B;
  std::vector<T> C;

  if (cppConfig.layoutIsRowMajor()) {
    A = std::vector<T>(M * lda, 0);
    B = std::vector<T>(K * ldb, 0);
    C = std::vector<T>(M * ldc, 0);
  } else {
    A = std::vector<T>(K * lda, 0);
    B = std::vector<T>(N * ldb, 0);
    C = std::vector<T>(N * ldc, 0);
  }

  // TODO (vinayakdsci): Refactor this into a reusable function.
  if (cppConfig.layoutIsRowMajor()) {
    for (int64_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < lda; ++j) {
        if (j < K) {
          A[i * lda + j] = static_cast<T>(dist(rng));
        }
      }
    }
  } else {
    for (size_t k = 0; k < K; ++k) {
      for (int64_t i = 0; i < lda; ++i) {
        if (i < M) {
          A[k * lda + i] = static_cast<T>(dist(rng));
        }
      }
    }
  }

  if (cppConfig.layoutIsRowMajor()) {
    for (int64_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < ldb; ++j) {
        if (j < N) {
          B[i * ldb + j] = static_cast<T>(dist(rng));
        }
      }
    }
  } else {
    for (size_t j = 0; j < N; ++j) {
      for (int64_t i = 0; i < ldb; ++i) {
        if (i < K) {
          B[j * ldb + i] = static_cast<T>(dist(rng));
        }
      }
    }
  }

  for (int i = 0; i < warmup_iters; ++i) {
    fn(A.data(), B.data(), C.data(), config);
  }

  double flops_per_gemm = 2.0 * M * N * K;
  double total_flops = iters * flops_per_gemm;

  Timer t = Timer();
  for (int i = 0; i < iters; ++i) {
    fn(A.data(), B.data(), C.data(), config);
  }

  double total_seconds = t.elapsed_s();
  double gflops = (total_flops / total_seconds) * (double)1e-9;

  std::cout << name << ":\n";
  std::cout << "\tAverage GFLOP/s: " << gflops << "\n";
}

template void benchmark_gemm<int>(const char *name, int warmup_iters, int iters,
                                  GemmFn<int>, devblas_gemm_config_t *config);
template void benchmark_gemm<float>(const char *name, int warmup_iters,
                                    int iters, GemmFn<float>,
                                    devblas_gemm_config_t *config);

} // namespace bench
} // namespace devblas::internal
