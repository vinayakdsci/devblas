#include "devblas/c_api/blas.h"
#include <assert.h>
#include <stdio.h>

int main(void) {

  devblas_gemm_config_t config = {
      .M = 1024,
      .N = 1024,
      .K = 1024,
      .lda = 1032,
      .ldb = 1032,
      .ldc = 1032,
      .tileSize = 8,
  };

  bench_sgemm(tiled_sgemm, "tiled_sgemm_ijk_benchmark", 2, 5, &config);

  return 0;
}
