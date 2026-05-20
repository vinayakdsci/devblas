#include "devblas/c_api/blas.h"
#include <assert.h>
#include <stdio.h>

#define SQ (1024)
int main(void) {
    devblas_gemm_config_t config = {
        .layout = DEVBLAS_LAYOUT_ROW_MAJOR,
        .M = SQ,
        .N = SQ,
        .K = SQ,
        .lda = SQ + 8,
        .ldb = SQ + 8,
        .ldc = SQ + 8,
        .tileSize = 16,
    };

    bench_sgemm(tiled_sgemm, "my_exp_kernel", 2, 5, &config);
    return 0;
}
