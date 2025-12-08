#ifndef DEVBLAS_NAIVE_GEMM_H
#define DEVBLAS_NAIVE_GEMM_H

#include "devblas/types/layout.h"
#include <concepts>

namespace devblas {
namespace internal {

template <typename T>
  requires(std::integral<T> || std::floating_point<T>)
void naive_gemm_ijk(types::Layout layout, const T *A, const T *B, T *C, int M,
                    int N, int K, int lda, int ldb, int ldc);

template <typename T>
  requires(std::integral<T> || std::floating_point<T>)
void naive_gemm_kij(types::Layout layout, const T *A, const T *B, T *C, int M,
                    int N, int K, int lda, int ldb, int ldc);
} // namespace internal
} // namespace devblas
#endif
