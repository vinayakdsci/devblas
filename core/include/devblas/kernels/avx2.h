#pragma once

#include <devblas/api/backend.h>

void devblas_avx2_8x8_sgemm(const devblas_f32_t *const restrict A,
							const devblas_f32_t *const restrict B,
							devblas_f32_t *const restrict C,
							devblas_i32_t K, devblas_i32_t ldc);

void devblas_avx2_8x4_sgemm(const devblas_f32_t *const restrict A,
							const devblas_f32_t *const restrict B,
							devblas_f32_t *const restrict C,
							devblas_i32_t K, devblas_i32_t ldc);

void devblas_avx2_4x8_sgemm(const devblas_f32_t *const restrict A,
							const devblas_f32_t *const restrict B,
							devblas_f32_t *const restrict C,
							devblas_i32_t K, devblas_i32_t ldc);

void devblas_register_avx2_backend(void);
