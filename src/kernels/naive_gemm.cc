#include "devblas/kernels/naive_gemm.h"
#include "devblas/types/layout.h"

namespace devblas {
namespace internal {
template <typename T>
  requires(std::integral<T> || std::floating_point<T>)
void naive_gemm_ijk(types::Layout layout, const T *A, const T *B, T *C, int M,
                    int N, int K, int lda, int ldb, int ldc) {
  if (layout == types::Layout::ROW_MAJOR) {
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
void naive_gemm_kij(types::Layout layout, const T *A, const T *B, T *C, int M,
                    int N, int K, int lda, int ldb, int ldc) {
  bool init = false;
  if (layout == types::Layout::ROW_MAJOR) {
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          if (!init) {
            C[i * ldc + j] = 0;
            init = true;
          }
          C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
        }
      }
    }
  } else {
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          if (!init) {
            C[j * ldc + i] = 0;
            init = true;
          }
          C[j * ldc + i] += A[k * lda + i] * B[j * ldb + k];
        }
      }
    }
  }
}

template void naive_gemm_kij<int>(types::Layout, const int *, const int *,
                                  int *, int, int, int, int, int, int);

template void naive_gemm_kij<float>(types::Layout, const float *, const float *,
                                    float *, int, int, int, int, int, int);

template void naive_gemm_ijk<int>(types::Layout, const int *, const int *,
                                  int *, int, int, int, int, int, int);

template void naive_gemm_ijk<float>(types::Layout, const float *, const float *,
                                    float *, int, int, int, int, int, int);

} // namespace internal
} // namespace devblas
