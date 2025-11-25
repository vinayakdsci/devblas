add_library(devblas SHARED
	kernels/naive_gemm.cc
	c_api/blas.cc
	benchmarking/bench.cc
)

target_include_directories(devblas PUBLIC ../include/)


set_target_properties(devblas PROPERTIES
    VISIBILITY_INLINES_HIDDEN true
)

install(TARGETS devblas DESTINATION devblas)

