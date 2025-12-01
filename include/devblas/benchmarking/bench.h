#ifndef DEVBLAS_BENCH_H
#define DEVBLAS_BENCH_H

#include "devblas/c_api/blas.h"
#include "devblas/types/layout.h"

namespace devblas::internal::bench {

template <typename T>
using GemmFn = void (*)(devblas_layout_t layout, const T *, const T *, T *, int,
                        int, int, int, int, int);

template <typename T>
void benchmark_gemm(const char *name, int warmup_iters, GemmFn<T> fn,
                    devblas_layout_t layout, int iters, int M, int N, int K,
                    int lda, int ldb, int ldc);
} // namespace devblas::internal::bench

#endif
