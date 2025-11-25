#ifndef DEVBLAS_LAYOUT_H
#define DEVBLAS_LAYOUT_H

#include "devblas/c_api/blas.h"

namespace devblas::internal {
namespace types {
enum class Layout {
  ROW_MAJOR,
  COLUMN_MAJOR,
};

inline Layout layout_to_cpp(devblas_layout_t layout) {
  switch (layout) {
  case DEVBLAS_LAYOUT_ROW_MAJOR:
    return internal::types::Layout::ROW_MAJOR;
  case DEVBLAS_LAYOUT_COLUMN_MAJOR:
    return internal::types::Layout::COLUMN_MAJOR;
  default:
    return internal::types::Layout::ROW_MAJOR;
  }
}
}
} // namespace devblas::internal

#endif