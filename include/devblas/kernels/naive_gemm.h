#ifndef DEVBLAS_NAIVE_GEMM_H
#define DEVBLAS_NAIVE_GEMM_H

#include "devblas/types/config.h"
#include "devblas/types/layout.h"
#include <concepts>

namespace devblas {
namespace internal {

template <typename T>
  requires(std::integral<T> || std::floating_point<T>)
void naive_gemm_ijk(const T *A, const T *B, T *C, types::GemmConfig config);

template <typename T>
  requires(std::integral<T> || std::floating_point<T>)
void naive_gemm_kij(const T *A, const T *B, T *C, types::GemmConfig config);
} // namespace internal
} // namespace devblas
#endif
