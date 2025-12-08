#include "devblas/c_api/blas.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
  // bench_igemm(naive_igemm_ijk, "naive_igemm_ijk_driver", 1, 3,
  //             DEVBLAS_LAYOUT_ROW_MAJOR, 1024, 1024, 1024, 1024, 1024, 1024);
  // bench_sgemm(naive_sgemm_ijk, "naive_sgemm_ijk_driver", 1, 3,
  //             DEVBLAS_LAYOUT_COLUMN_MAJOR, 1024, 1024, 1024, 1024, 1024,
  //             1024);

  float A[] = {0, 1, 2, 3};
  float B[] = {0, 1, 2, 3};
  float C[] = {0, 0, 0, 0};

  naive_sgemm_kij(DEVBLAS_LAYOUT_COLUMN_MAJOR, A, B, C, 2, 2, 2, 2, 2, 2);

  for (size_t i = 0; i < 4; ++i) {
    fprintf(stderr, "%f\n", C[i]);
  }

  return 0;
}
