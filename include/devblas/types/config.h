#ifndef DEVBLAS_TYPES_CONFIG_H
#define DEVBLAS_TYPES_CONFIG_H

#include <optional>
#include <array>
#include "devblas/types/layout.h"

namespace devblas::internal::types {
struct GemmConfig {

  GemmConfig(Layout layout, int M, int N, int K, int lda, int ldb, int ldc,
             int tileSize)
      : layout_(layout), M_(M), N_(N), K_(K), lda_(lda), ldb_(ldb), ldc_(ldc),
        tileSize_(tileSize) {}

  GemmConfig(Layout layout, int M, int N, int K, int lda, int ldb, int ldc)
      : layout_(layout), M_(M), N_(N), K_(K), lda_(lda), ldb_(ldb), ldc_(ldc),
        tileSize_(std::nullopt) {}

  std::array<int, 3> logicalDims() {
    return std::array<int, 3>{M_, N_, K_};
  }

  std::array<int, 3> leadingDims() {
    return std::array<int, 3>{lda_, ldb_, ldc_};
  }

  std::optional<int> tileSize() {
    return tileSize_;
  }

  Layout layout() {
    return layout_;
  }

  bool layoutIsRowMajor() {
    return layout_ == Layout::ROW_MAJOR;
  }

  bool layoutIsColMajor() {
    return !layoutIsRowMajor();
  }

private:
  Layout layout_;
  int M_;
  int N_;
  int K_;
  int lda_;
  int ldb_;
  int ldc_;
  std::optional<int> tileSize_;
};
} // namespace devblas::internal::types

#endif
