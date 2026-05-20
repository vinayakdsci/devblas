#include "devblas/kernels/tiled_gemm.h"
#include "devblas/macros.h"

namespace devblas::internal {

template <typename T>
static void accumulateRowMajor(const T *const __restrict A,
                               const T *const __restrict B, T *C,
                               types::GemmConfig &config) {
    auto [MM, NN, KK] = config.logicalDims();
    auto [LDA, LDB, LDC] = config.leadingDims();
    std::optional<int> ts = config.tileSize();
    if (!ts || (*ts <= 0)) {
        throw std::runtime_error(
            "Non-negative TileSize should be passed in for tiled GEMM");
    }

    DEVBLAS_INT_T M = MM, N = NN, K = KK, lda = LDA, ldb = LDB, ldc = LDC;

    DEVBLAS_OMP_FOR
    for (DEVBLAS_INT_T i = 0; i < M; i += *ts) {
        DEVBLAS_INT_T i_end = std::min(i + *ts, M);
        for (DEVBLAS_INT_T j = 0; j < N; j += *ts) {
            DEVBLAS_INT_T j_end = std::min(j + *ts, N);
            for (DEVBLAS_INT_T k = 0; k < K; k += *ts) {
                DEVBLAS_INT_T k_end = std::min(k + *ts, K);
                for (DEVBLAS_INT_T ii = i; ii < i_end; ++ii) {
                    for (DEVBLAS_INT_T jj = j; jj < j_end; ++jj) {
                        T acc = T{0};
                        for (DEVBLAS_INT_T kk = k; kk < k_end; ++kk) {
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
    auto [MM, NN, KK] = config.logicalDims();
    auto [LDA, LDB, LDC] = config.leadingDims();
    std::optional<int> ts = config.tileSize();
    if (!ts || (*ts <= 0)) {
        throw std::runtime_error(
            "Non-negative TileSize should be passed in for tiled GEMM");
    }

    DEVBLAS_INT_T M = MM, N = NN, K = KK, lda = LDA, ldb = LDB, ldc = LDC;

    DEVBLAS_OMP_FOR
    for (DEVBLAS_INT_T k = 0; k < K; k += *ts) {
        DEVBLAS_INT_T k_end = std::min(k + *ts, K);
        for (DEVBLAS_INT_T i = 0; i < M; i += *ts) {
            DEVBLAS_INT_T i_end = std::min(i + *ts, M);
            for (DEVBLAS_INT_T j = 0; j < N; j += *ts) {
                DEVBLAS_INT_T j_end = std::min(j + *ts, N);
                for (DEVBLAS_INT_T kk = k; kk < k_end; ++kk) {
                    for (DEVBLAS_INT_T ii = i; ii < i_end; ++ii) {
                        for (DEVBLAS_INT_T jj = j; jj < j_end; ++jj) {
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
