#include "devblas/benchmarking/bench.h"
#include "devblas/benchmarking/utils.h"
#include "devblas/c_api/blas.h"
#include "devblas/types/layout.h"
#include <iostream>
#include <random>

namespace devblas::internal {
namespace bench {

template <typename T>
void benchmark_gemm(const char *name, GemmFn<T> fn, devblas_layout_t layout,
                    int iters, int M, int N, int K, int lda, int ldb, int ldc) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<T> A;
  std::vector<T> B;
  std::vector<T> C;

  if (layout == DEVBLAS_LAYOUT_ROW_MAJOR) {
    A = std::vector<T>(M * lda);
    B = std::vector<T>(K * ldb);
    C = std::vector<T>(M * ldc);
  } else {
    A = std::vector<T>(K * lda);
    B = std::vector<T>(N * ldb);
    C = std::vector<T>(N * ldc);
  }

  // TODO (vinayakdsci): Refactor this into a reusable function.
  if (layout == DEVBLAS_LAYOUT_ROW_MAJOR) {
    for (int64_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < lda; ++j) {
        if (j >= K) {
          A[i * lda + j] = 0;
        } else {
          A[i * lda + j] = static_cast<T>(dist(rng));
        }
      }
    }
  } else {
    for (size_t k = 0; k < K; ++k) {
      for (int64_t i = 0; i < lda; ++i) {
        if (i >= M) {
          A[k * lda + i] = 0;
        } else {
          A[k * lda + i] = static_cast<T>(dist(rng));
        }
      }
    }
  }

  if (layout == DEVBLAS_LAYOUT_ROW_MAJOR) {
    for (int64_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < ldb; ++j) {
        if (j >= N) {
          B[i * ldb + j] = 0;
        } else {
          B[i * ldb + j] = static_cast<T>(dist(rng));
        }
      }
    }
  } else {
    for (size_t j = 0; j < N; ++j) {
      for (int64_t i = 0; i < ldb; ++i) {
        if (i >= K) {
          B[j * ldb + i] = 0;
        } else {
          B[j * ldb + i] = static_cast<T>(dist(rng));
        }
      }
    }
  }

  double flops_per_gemm = 2.0 * M * N * K;
  double total_flops = iters * flops_per_gemm;

  Timer t = Timer();
  for (int i = 0; i < iters; ++i) {
    fn(layout, A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
  }

  double total_seconds = t.elapsed_s();
  double gflops = (total_flops / total_seconds) * (double)1e-9;

  std::cout << name << ":\n";
  std::cout << "\tAverage GFLOP/s: " << gflops << "\n";
}

template void benchmark_gemm<int>(const char *name, GemmFn<int>,
                                  devblas_layout_t layout, int iters, int M,
                                  int N, int K, int lda, int ldb, int ldc);
template void benchmark_gemm<float>(const char *name, GemmFn<float>,
                                    devblas_layout_t layout, int iters, int M,
                                    int N, int K, int lda, int ldb, int ldc);

} // namespace bench
} // namespace devblas::internal
