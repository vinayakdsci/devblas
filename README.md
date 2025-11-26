## devblas: A CPU BLAS implementation

**devblas** is a simple CPU-only educational BLAS framework written in C++ and exposed through C to learn about how high performance linear algebra operations are written.
The library supports both column-major and row-major layouts and also provides benchmarking harness that can be used for any IGEMM/SGEMM implementation as long as the function
signatures are satisfied.

>[!NOTE]
>The API is intentionally exposed through C to maintain ABI stability.
>Only the internal implementation details are written in C++ and avoid use of classes and other OOP features unless absolutely necessary.
>We make use of templates to write re-usable code without introducing additional runtime costs.

### Building

**devblas** build produces a `.so` shared library. It is intended that the user will use the exported API by linking against the shared object file produced after compilation.
To build (and install), run

```sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=devblas_out; cmake --build build && cmake --install build
```
in the root of the repository.

This will install `libdevblas.so` in the `devblas_out/devblas` directory. You can now link your code against the `.so` file produced to use the provided BLAS implementation.

A driver implementation named `bench_driver.c` is also provided to experiment with benchmarking and will automatically be linked with `libdevblas.so`. To use it, add `-DDEVBLAS_BUILD_BENCH=ON` in the cmake command.

### Benchmarking
**devblas** is a BLAS library, and hence benchmarking is an important part of it. Benchmarking code is implemented in `src/benchmarking/` with headers in `include/devblas/benchmarking`.
Currently **devblas** exposes two benchmarking functions, `bench_igemm(...)` and `bench_sgemm(...)`. Each of them takes in a function pointer to a GEMM implementation (IGEMM or SGEMM) and its argumments.

The signatures of the function pointers (for both IGEMM and SGEMM) are as below:
```cpp
// sgemm
typedef void (*devblas_sgemm_fn)(devblas_layout_t, const float *, const float *,
                                 float *, int, int, int, int, int, int);
// igemm
typedef void (*devblas_igemm_fn)(devblas_layout_t, const int *, const int *,
                                 int *, int, int, int, int, int, int);
```

`bench_driver.c` contains example benchmarking code that benchmarks both `igemm` and `sgemm` over 3 iterations and prints the average GFLOP/s value to `stdout`.

As long as the function pointer signature is satisfied, the user can easily plugin their own implementation of a GEMM (provided it is either IGEMM or SGEMM) into the respective benchmarking function to benchmark it.

### Tests

It is easy to write code in a BLAS implementation that fails numerical accuracy or slows down performance. To ensure implementation correctness and performance, unit tests (which assess only correctness for now) are implemented in the `tests/` directory.
To run the tests, build the code and, after entering the build directory, run:
```sh
$ ninja test
```
This will run the whole test harness.
