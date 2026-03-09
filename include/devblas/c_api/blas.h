#ifndef DEVBLAS_C_API_H
#define DEVBLAS_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum devblas_layout_t {
  DEVBLAS_LAYOUT_ROW_MAJOR,
  DEVBLAS_LAYOUT_COLUMN_MAJOR
} devblas_layout_t;

typedef struct devblas_gemm_config_t {
  devblas_layout_t layout;
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldc;
  int tileSize;
} devblas_gemm_config_t;

void naive_igemm_ijk(const int *, const int *, int *,
                     devblas_gemm_config_t *config);
void naive_sgemm_ijk(const float *, const float *, float *,
                     devblas_gemm_config_t *config);

void naive_igemm_kij(const int *, const int *, int *,
                     devblas_gemm_config_t *config);
void naive_sgemm_kij(const float *, const float *, float *,
                     devblas_gemm_config_t *config);

void tiled_sgemm(const float *, const float *, float *,
                 devblas_gemm_config_t *config);
void tiled_igemm(const int *, const int *, int *,
                 devblas_gemm_config_t *config);

typedef void (*devblas_sgemm_fn)(const float *, const float *, float *,
                                 devblas_gemm_config_t *config);
typedef void (*devblas_igemm_fn)(const int *, const int *, int *,
                                 devblas_gemm_config_t *config);

void bench_igemm(devblas_igemm_fn fn, const char *name, int warmup_iters,
                 int iter, devblas_gemm_config_t *config);
void bench_sgemm(devblas_sgemm_fn fn, const char *name, int warmup_iters,
                 int iter, devblas_gemm_config_t *config);
#ifdef __cplusplus
}
#endif

#endif
