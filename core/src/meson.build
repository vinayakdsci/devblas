subdir('kernels')

libdevblas_core = static_library(
	'devblas-core',

    files(
      'utils.c',
    ),

	include_directories : inc,
	c_args : opt,
	dependencies : [
		devblas_kernels,
	],
)

devblas_core = declare_dependency(
	link_with : libdevblas_core,

	include_directories : inc,
	dependencies : [
		devblas_kernels,
	],
)
