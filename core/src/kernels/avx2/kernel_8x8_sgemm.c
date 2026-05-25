#include <devblas/kernels/avx2.h>

#include <stdio.h>

void
devblas_avx2_8x8_sgemm(const devblas_f32_t *const restrict A,
					   const devblas_f32_t *const restrict B,
					   devblas_f32_t *const restrict C, devblas_i32_t K,
					   devblas_i32_t ldc)
{
	printf("Hello, World!\n");
}
