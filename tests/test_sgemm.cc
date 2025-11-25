#include "devblas/kernels/naive_gemm.h"
#include "devblas/types/layout.h"
#include <gtest/gtest.h>

using namespace devblas::internal;

TEST(GemmTestAccuracy, IGEMMColMajor) {
  int A[6] = {1, 4, 2, 5, 3, 6};
  int B[6] = {7, 9, 11, 8, 10, 12};
  int C[4] = {0, 0, 0, 0};

  int M = 2, N = 2, K = 3;
  naive_gemm_ijk<int>(types::Layout::COLUMN_MAJOR, A, B, C, M, N, K, M, K, M);

  std::vector<int> expected = {58, 139, 64, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTestAccuracy, IGEMMRowMajor) {
  int A[6] = {1, 2, 3, 4, 5, 6};
  int B[6] = {7, 8, 9, 10, 11, 12};
  int C[4] = {0, 0, 0, 0};

  int M = 2, N = 2, K = 3;
  naive_gemm_ijk<int>(types::Layout::ROW_MAJOR, A, B, C, M, N, K, K, N, N);

  std::vector<int> expected = {58, 64, 139, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}
