#include "devblas/c_api/blas.h"
#include <assert.h>

int main(void) {
  bench_igemm(naive_igemm_ijk, 3, DEVBLAS_LAYOUT_ROW_MAJOR, 1024, 1024, 1024,
              1024, 1024, 1024);
  bench_sgemm(naive_sgemm_ijk, 3, DEVBLAS_LAYOUT_COLUMN_MAJOR, 1024, 1024, 1024,
              1024, 1024, 1024);
  return 0;
}
