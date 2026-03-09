#ifndef DEVBLAS_TYPES_CONFIG_H
#define DEVBLAS_TYPES_CONFIG_H

#include "devblas/types/layout.h"
#include <array>
#include <optional>
#include <stdexcept>

namespace devblas::internal::types {
struct GemmConfig {

  GemmConfig(Layout layout, int M, int N, int K, int lda, int ldb, int ldc,
             int tileSize)
      : layout_(layout), M_(M), N_(N), K_(K), lda_(lda), ldb_(ldb), ldc_(ldc),
        tileSize_(tileSize) {}

  GemmConfig(Layout layout, int M, int N, int K, int lda, int ldb, int ldc)
      : layout_(layout), M_(M), N_(N), K_(K), lda_(lda), ldb_(ldb), ldc_(ldc),
        tileSize_(std::nullopt) {}

  std::array<int, 3> logicalDims() { return std::array<int, 3>{M_, N_, K_}; }

  std::array<int, 3> leadingDims() {
    return std::array<int, 3>{lda_, ldb_, ldc_};
  }

  std::optional<int> tileSize() { return tileSize_; }

  Layout layout() { return layout_; }

  bool layoutIsRowMajor() { return layout_ == Layout::ROW_MAJOR; }

  bool layoutIsColMajor() { return !layoutIsRowMajor(); }

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

inline GemmConfig gemm_config_to_cpp(devblas_gemm_config_t *config) {
  if (!config) {
    throw std::runtime_error("Received NULL GEMM config!");
  }

  if (config->tileSize < 0) {
    return GemmConfig(layout_to_cpp(config->layout), config->M, config->N,
                      config->K, config->lda, config->ldb, config->ldc);
  }

  return GemmConfig(layout_to_cpp(config->layout), config->M, config->N,
                    config->K, config->lda, config->ldb, config->ldc,
                    config->tileSize);
}

} // namespace devblas::internal::types

#endif
