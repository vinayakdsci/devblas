#include "devblas/kernels/tiled_gemm.h"

namespace devblas::internal {

template <typename T>
  requires(std::integral<T> || std::floating_point<T>)
void tiled_gemm(const T *A, const T *B, T *C, types::GemmConfig &config) {
  auto [M, N, K] = config.logicalDims();
  auto [lda, ldb, ldc] = config.leadingDims();
  std::optional<int> ts = config.tileSize();

  if (!ts) {
    throw std::runtime_error("TileSize should be passed in for tiled GEMM");
  }

  // Zero initialize C
  size_t cSize = (config.layoutIsRowMajor() ? M * ldc : N * ldc);
  for (size_t i = 0; i < cSize; ++i) {
    C[i] = 0;
  }

  if (config.layoutIsRowMajor()) {
    for (int i = 0; i < M; i += *ts) {
      for (int j = 0; j < N; j += *ts) {
        for (int k = 0; k < K; k += *ts) {
          // Decide the tile size based on how many rows/cols are left on the
          // edge.
          int i_end = std::min(i + *ts, M);
          int j_end = std::min(j + *ts, N);
          int k_end = std::min(k + *ts, K);

          // Perform matmul on the tiles.
          for (int ii = i; ii < i_end; ++ii) {
            for (int jj = j; jj < j_end; ++jj) {
              T acc = 0;
              for (int kk = 0; kk < k_end; ++kk) {
                acc += A[ii * lda + kk] * B[kk * ldb + jj];
              }
              C[ii * ldc + jj] = acc;
            }
          }
        }
      }
    }
  } else {
    for (int j = 0; j < N; j += *ts) {
      for (int i = 0; i < M; i += *ts) {
        for (int k = 0; k < K; k += *ts) {
          // Decide the tile size based on how many rows/cols are left on the
          // edge.
          int i_end = std::min(i + *ts, M);
          int j_end = std::min(j + *ts, N);
          int k_end = std::min(k + *ts, K);

          // Perform matmul on the tiles.
          for (int jj = j; jj < j_end; ++jj) {
            for (int ii = i; ii < i_end; ++ii) {
              T acc = 0;
              for (int kk = 0; kk < k_end; ++kk) {
                acc += A[kk * lda + ii] * B[jj * ldb + kk];
              }
              C[jj * ldc + ii] = acc;
            }
          }
        }
      }
    }
  }
}

template void tiled_gemm<float>(const float *A, const float *B, float *C, types::GemmConfig& config);
template void tiled_gemm<int>(const int *A, const int *B, int *C, types::GemmConfig& config);

} // namespace devblas::internal
