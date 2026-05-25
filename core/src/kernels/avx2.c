#include <devblas/kernels/avx2.h>

static const devblas_gemm_backend_t avx2_backend = {
        .sgemm_ukernels = {
            // [4][8] = devblas_avx2_4x8_sgemm,
            // [8][4] = devblas_avx2_8x4_sgemm,
            [8][8] = devblas_avx2_8x8_sgemm,
        },
        .igemm_ukernels = {0},
    };

DEVBLAS_REGISTER_BACKEND(DEVBLAS_ISA_AVX2, avx2_backend)
