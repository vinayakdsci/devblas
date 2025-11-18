## devblas: A CPU BLAS implementation

**devblas** is a simple CPU-only educational BLAS framework written in C to learn about how high performance linear algebra operations are written.

### Building

**devblas** build produces a `.so` shared library. It is intended that the user will use the exported API by linking against the shared object file produced after compilation.
To build (and install), run

```sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=devblas_out; cmake --build build && cmake --install build
```
in the root of the repository.

This will install `libdevblas.so` in the `devblas_out/devblas` directory. You can now link your code against the `.so` file produced to use the provided BLAS implementation. 
