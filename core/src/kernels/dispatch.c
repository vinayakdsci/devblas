#include <devblas/api/backend.h>
#include <devblas/kernels/avx2.h>

const devblas_gemm_backend_t
		*devblas_cpu_backend_registry[DEVBLAS_ISA_CNT] = { NULL };

static bool initialized = false;
static const devblas_gemm_backend_t *active_backend = 0;

static void
devblas_register_backends_once(void)
{
	if (!initialized)
		{
			devblas_register_avx2_backend();
			active_backend =
					devblas_cpu_backend_registry[devblas_cpu_isa()];
			initialized = true;
		}
}

void
devblas_sgemm(devblas_layout_t layout, devblas_transpose_t transA,
			  devblas_transpose_t transB, devblas_i32_t M,
			  devblas_i32_t N, devblas_i32_t K,
			  const devblas_f32_t *const restrict A, devblas_i32_t lda,
			  const devblas_f32_t *const restrict B, devblas_i32_t ldb,
			  devblas_f32_t *const restrict C, devblas_i32_t ldc)
{
	devblas_register_backends_once();

	const int MR = 8, NR = 8;
	active_backend->sgemm_ukernels[MR][NR](A, B, C, K, ldc);
}
