#ifndef DEVBLAS_TILED_GEMM_H
#define DEVBLAS_TILED_GEMM_H

#include "devblas/types/config.h"
#include <concepts>
#include <stdexcept>

namespace devblas::internal {
template <typename T>
    requires(std::integral<T> || std::floating_point<T>)
void tiled_gemm(const T *const __restrict A, const T *const __restrict B, T *C,
                types::GemmConfig config);
} // namespace devblas::internal

#endif
