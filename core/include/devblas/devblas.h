#pragma once

#include <devblas/api/typedecls.h>

void
devblas_sgemm(devblas_layout_t layout, devblas_transpose_t transA,
			  devblas_transpose_t transB, devblas_i32_t M,
			  devblas_i32_t N, devblas_i32_t K,
			  const devblas_f32_t *const restrict A, devblas_i32_t lda,
			  const devblas_f32_t *const restrict B, devblas_i32_t ldb,
			  devblas_f32_t *const restrict C, devblas_i32_t ldc);
