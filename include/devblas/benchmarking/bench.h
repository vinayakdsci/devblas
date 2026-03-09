#ifndef DEVBLAS_BENCH_H
#define DEVBLAS_BENCH_H

#include "devblas/c_api/blas.h"
#include "devblas/types/config.h"
#include "devblas/types/layout.h"

namespace devblas::internal::bench {

template <typename T>
using GemmFn = void (*)(const T *, const T *, T *,
                        devblas_gemm_config_t *config);

template <typename T>
void benchmark_gemm(const char *name, int warmup_iters, int iters, GemmFn<T> fn,
                    devblas_gemm_config_t *config);
} // namespace devblas::internal::bench

#endif
