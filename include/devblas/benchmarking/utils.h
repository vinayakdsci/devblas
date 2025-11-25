#ifndef DEVBLAS_BENCHMARK_UTILS_H
#define DEVBLAS_BENCHMARK_UTILS_H

#include <chrono>

namespace devblas::internal {
namespace bench {
struct Timer {
  Timer() : start_(std::chrono::steady_clock::now()) {}
  double elapsed_ms() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_)
        .count();
  }
  double elapsed_s() { return elapsed_ms() * (double)1e-3; }
  void reset() { start_ = std::chrono::steady_clock::now(); }

private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
};

} // namespace bench
} // namespace devblas::internal

#endif
