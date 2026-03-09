#include "devblas/kernels/naive_gemm.h"
#include "devblas/types/layout.h"

namespace devblas {
namespace internal {
template <typename T>
  requires(std::integral<T> || std::floating_point<T>)
void naive_gemm_ijk(const T *A, const T *B, T *C, types::GemmConfig config) {
  auto [M, N, K] = config.logicalDims();
  auto [lda, ldb, ldc] = config.leadingDims();

  if (config.layoutIsRowMajor()) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] = 0;
        for (int k = 0; k < K; ++k) {
          C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
        }
      }
    }
  } else {
    // Column major.
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        C[j * ldc + i] = 0;
        for (int k = 0; k < K; ++k) {
          C[j * ldc + i] += A[k * lda + i] * B[j * ldb + k];
        }
      }
    }
  }
}

template <typename T>
  requires(std::integral<T> || std::floating_point<T>)
void naive_gemm_kij(const T *A, const T *B, T *C, types::GemmConfig config) {
  auto [M, N, K] = config.logicalDims();
  auto [lda, ldb, ldc] = config.leadingDims();

  // Zero initialize C before writing into it.
  for (int i = 0; i < (config.layoutIsRowMajor() ? N * ldc : M * ldc); ++i) {
    C[i] = 0;
  }
  if (config.layoutIsRowMajor()) {
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
        }
      }
    }
  } else {
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          C[j * ldc + i] += A[k * lda + i] * B[j * ldb + k];
        }
      }
    }
  }
}

template void naive_gemm_kij<int>(const int *, const int *, int *,
                                  types::GemmConfig config);

template void naive_gemm_kij<float>(const float *, const float *, float *,
                                    types::GemmConfig config);

template void naive_gemm_ijk<int>(const int *, const int *, int *,
                                  types::GemmConfig config);

template void naive_gemm_ijk<float>(const float *, const float *, float *,
                                    types::GemmConfig config);

} // namespace internal
} // namespace devblas
