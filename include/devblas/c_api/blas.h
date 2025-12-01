#ifndef DEVBLAS_C_API_H
#define DEVBLAS_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum devblas_layout_t {
  DEVBLAS_LAYOUT_ROW_MAJOR,
  DEVBLAS_LAYOUT_COLUMN_MAJOR
} devblas_layout_t;

void naive_igemm_ijk(devblas_layout_t, const int *, const int *, int *, int,
                     int, int, int, int, int);
void naive_sgemm_ijk(devblas_layout_t, const float *, const float *, float *,
                     int, int, int, int, int, int);

typedef void (*devblas_sgemm_fn)(devblas_layout_t, const float *, const float *,
                                 float *, int, int, int, int, int, int);
typedef void (*devblas_igemm_fn)(devblas_layout_t, const int *, const int *,
                                 int *, int, int, int, int, int, int);

void bench_igemm(devblas_igemm_fn fn, const char *name, int warmup_iters, int iter,
                 devblas_layout_t layout, int M, int N, int K, int lda, int ldb, int ldc);
void bench_sgemm(devblas_sgemm_fn fn, const char *name, int warmup_iters, int iter,
                 devblas_layout_t layout, int M, int N, int K, int lda, int ldb, int ldc);
#ifdef __cplusplus
}
#endif

#endif
