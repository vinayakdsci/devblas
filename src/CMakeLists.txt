add_library(devblas SHARED
	kernels/naive_gemm.cc
	kernels/tiled_gemm.cc
	c_api/blas.cc
	benchmarking/bench.cc
)

target_include_directories(devblas PUBLIC ../include/)
target_compile_options(
	devblas
	PRIVATE
		$<$<CONFIG:Release>:-O3 -march=native -pedantic>
)

if(DEVBLAS_OPENMP_PARALLELIZE)
	target_compile_definitions(devblas PRIVATE DEVBLAS_OPENMP_PARALLELIZE)
	target_compile_options(
		devblas
		PRIVATE
			-fopenmp
	)
endif()

set_target_properties(devblas PROPERTIES
    VISIBILITY_INLINES_HIDDEN true
)

install(TARGETS devblas DESTINATION devblas)

