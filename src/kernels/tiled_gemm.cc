#include "devblas/kernels/tiled_gemm.h"

namespace devblas::internal {

template <typename T>
static void accumulateRowMajor(const T *const __restrict A,
                               const T *const __restrict B, T *C,
                               types::GemmConfig &config) {
    auto [M, N, K] = config.logicalDims();
    auto [lda, ldb, ldc] = config.leadingDims();
    std::optional<int> ts = config.tileSize();
    if (!ts || (*ts <= 0)) {
        throw std::runtime_error(
            "Non-negative TileSize should be passed in for tiled GEMM");
    }

    for (int i = 0; i < M; i += *ts) {
        int i_end = std::min(i + *ts, M);
        for (int j = 0; j < N; j += *ts) {
            int j_end = std::min(j + *ts, N);
            for (int k = 0; k < K; k += *ts) {
                int k_end = std::min(k + *ts, K);
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        T acc = T{0};
                        for (int kk = k; kk < k_end; ++kk) {
                            acc += A[ii * lda + kk] * B[kk * ldb + jj];
                        }
                        C[ii * ldc + jj] += acc;
                    }
                }
            }
        }
    }
}

template <typename T>
static void accumulateColMajor(const T *__restrict A, const T *__restrict B,
                               T *C, types::GemmConfig &config) {
    auto [M, N, K] = config.logicalDims();
    auto [lda, ldb, ldc] = config.leadingDims();
    std::optional<int> ts = config.tileSize();
    if (!ts || (*ts <= 0)) {
        throw std::runtime_error(
            "Non-negative TileSize should be passed in for tiled GEMM");
    }

    for (int k = 0; k < K; k += *ts) {
        int k_end = std::min(k + *ts, K);
        for (int i = 0; i < M; i += *ts) {
            int i_end = std::min(i + *ts, M);
            for (int j = 0; j < N; j += *ts) {
                int j_end = std::min(j + *ts, N);
                for (int kk = k; kk < k_end; ++kk) {
                    for (int ii = i; ii < i_end; ++ii) {
                        for (int jj = j; jj < j_end; ++jj) {
                            C[jj * ldc + ii] +=
                                A[kk * lda + ii] * B[jj * ldb + kk];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
    requires(std::integral<T> || std::floating_point<T>)
void tiled_gemm(const T *const __restrict A, const T *const __restrict B, T *C,
                types::GemmConfig config) {
    if (config.layoutIsRowMajor()) {
        accumulateRowMajor<T>(A, B, C, config);
        return;
    }
    accumulateColMajor<T>(A, B, C, config);
}

template void tiled_gemm<float>(const float *A, const float *B, float *C,
                                types::GemmConfig config);
template void tiled_gemm<int>(const int *A, const int *B, int *C,
                              types::GemmConfig config);

} // namespace devblas::internal
