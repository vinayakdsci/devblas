#include "devblas/kernels/naive_gemm.h"
#include "devblas/kernels/tiled_gemm.h"
#include "devblas/types/layout.h"
#include "devblas/types/config.h"
#include <gtest/gtest.h>

using namespace devblas::internal;

TEST(GemmTestAccuracy, IGEMMIJKColMajor) {
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

TEST(GemmTestAccuracy, IGEMMIJKRowMajor) {
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

TEST(GemmTestAccuracy, IGEMMKIJRowMajor) {
  int A[6] = {1, 2, 3, 4, 5, 6};
  int B[6] = {7, 8, 9, 10, 11, 12};
  int C[4] = {0, 0, 0, 0};

  int M = 2, N = 2, K = 3;
  naive_gemm_kij<int>(types::Layout::ROW_MAJOR, A, B, C, M, N, K, K, N, N);

  std::vector<int> expected = {58, 64, 139, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTestAccuracy, IGEMMKIJColMajor) {
  int A[6] = {1, 4, 2, 5, 3, 6};
  int B[6] = {7, 9, 11, 8, 10, 12};
  int C[4] = {0, 0, 0, 0};

  int M = 2, N = 2, K = 3;
  naive_gemm_kij<int>(types::Layout::COLUMN_MAJOR, A, B, C, M, N, K, M, K, M);

  std::vector<int> expected = {58, 139, 64, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTestAccuracy, SGEMMIJKColMajor) {
  float A[6] = {1, 4, 2, 5, 3, 6};
  float B[6] = {7, 9, 11, 8, 10, 12};
  float C[4] = {0, 0, 0, 0};

  float M = 2, N = 2, K = 3;
  naive_gemm_ijk<float>(types::Layout::COLUMN_MAJOR, A, B, C, M, N, K, M, K, M);

  std::vector<float> expected = {58, 139, 64, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTestAccuracy, SGEMMIJKRowMajor) { float A[6] = {1, 2, 3, 4, 5, 6};
  float B[6] = {7, 8, 9, 10, 11, 12};
  float C[4] = {0, 0, 0, 0};

  float M = 2, N = 2, K = 3;
  naive_gemm_ijk<float>(types::Layout::ROW_MAJOR, A, B, C, M, N, K, K, N, N);

  std::vector<float> expected = {58, 64, 139, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTestAccuracy, SGEMMKIJRowMajor) {
  float A[6] = {1, 2, 3, 4, 5, 6};
  float B[6] = {7, 8, 9, 10, 11, 12};
  float C[4] = {0, 0, 0, 0};

  float M = 2, N = 2, K = 3;
  naive_gemm_kij<float>(types::Layout::ROW_MAJOR, A, B, C, M, N, K, K, N, N);

  std::vector<float> expected = {58, 64, 139, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTestAccuracy, SGEMMKIJColMajor) {
  float A[6] = {1, 4, 2, 5, 3, 6};
  float B[6] = {7, 9, 11, 8, 10, 12};
  float C[4] = {0, 0, 0, 0};

  float M = 2, N = 2, K = 3;
  naive_gemm_kij<float>(types::Layout::COLUMN_MAJOR, A, B, C, M, N, K, M, K, M);

  std::vector<float> expected = {58, 139, 64, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTestAccuracy, SGEMMTiledRowMajor) {
  float A[6] = {1, 2, 3, 4, 5, 6};
  float B[6] = {7, 8, 9, 10, 11, 12};
  float C[4] = {0, 0, 0, 0};

  float M = 2, N = 2, K = 3;

  auto config = types::GemmConfig(types::Layout::ROW_MAJOR, M, N, K, K, N, N, 2);

  tiled_gemm<float>(A, B, C, config);

  std::vector<float> expected = {58, 64, 139, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTestAccuracy, SGEMMTiledColMajor) {
  float A[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};
  float B[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};
  float C[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  float M = 3, N = 3, K = 3;

  auto config = types::GemmConfig(types::Layout::COLUMN_MAJOR, M, N, K, M, K, M, 2);

  tiled_gemm<float>(A, B, C, config);

  std::vector<float> expected = {30, 66, 102, 36, 81, 126, 42, 96, 150};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}

TEST(GemmTestAccuracy, SGEMMTiledColMajorAsymm) {
  float A[6] = {1, 4, 2, 5, 3, 6};
  float B[6] = {7, 9, 11, 8, 10, 12};
  float C[4] = {0, 0, 0, 0};

  float M = 2, N = 2, K = 3;
  auto config = types::GemmConfig(types::Layout::COLUMN_MAJOR, M, N, K, M, K, M, 2);

  tiled_gemm<float>(A, B, C, config);

  std::vector<float> expected = {58, 139, 64, 154};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(C[i], expected[i]);
  }
}
