## devblas: A CPU BLAS implementation

**devblas** is a simple CPU-only educational BLAS framework written in C++, with a stable C API.
It is designed as a learning tool for understanding how high-performance linear algebra kernels
(such as GEMM) are implemented on modern CPUs.

The library supports both row-major and column-major matrix layouts and includes a flexible
benchmarking harness that can benchmark any IGEMM/SGEMM implementation that matches the
required function signatures.

> [!NOTE]
> The public API is intentionally exposed through C to maintain ABI stability.
> Internal implementation is written in C++ but avoids classes and heavy OOP.
> Templates are used only to remove duplicate logic without adding runtime cost.

### Building

**devblas** build produces a `.so` shared library. It is intended that the user will use the exported API by linking against the shared object file produced after compilation.
To build (and install), run

```sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=devblas_out; cmake --build build && cmake --install build
```
in the root of the repository.

This generates `libdevblas.so` inside `devblas_out/devblas`. You can link your own
applications against this shared library.

Each of them takes a function pointer to a user-provided IGEMM/SGEMM kernel along
with the matrix dimensions and leading dimensions needed for benchmarking.

A driver implementation named `bench_driver.c` is also provided to experiment with benchmarking and will automatically be linked with `libdevblas.so`. To use it, add `-DDEVBLAS_BUILD_BENCH=ON` in the cmake command.

### Benchmarking
**devblas** is a BLAS library, and hence benchmarking is an important part of it. Benchmarking code is implemented in `src/benchmarking/` with headers in `include/devblas/benchmarking`.
Currently **devblas** exposes two benchmarking functions, `bench_igemm(...)` and `bench_sgemm(...)`.

Each of them takes a function pointer to a user-provided IGEMM/SGEMM kernel along
with the matrix dimensions and leading dimensions needed for benchmarking.

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

Writing BLAS kernels is error-prone and it is easy to introduce numerical mistakes
or degrade performance unintentionally.
To ensure correctness, unit tests are implemented in the `tests/` directory.
Performance tests will be added later.

To run the tests, build the code and, after entering the build directory, run:
```sh
$ ninja test
```
This will run the whole test harness.
