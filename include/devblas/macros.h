#pragma once

#include <stdint.h>

#define DEVBLAS_STR_MACRO(x) #x

#ifdef DEVBLAS_OPENMP_PARALLELIZE
#define DEVBLAS_OMP_FOR_COLLAPSE(x) \
    _Pragma(DEVBLAS_STR_MACRO(omp parallel for collapse(x)))
#define DEVBLAS_OMP_FOR \
    _Pragma(DEVBLAS_STR_MACRO(omp parallel for simd))
#else
#define DEVBLAS_OMP_FOR_COLLAPSE(x)
#define DEVBLAS_OMP_FOR
#endif

#define DEVBLAS_INT_T int64_t