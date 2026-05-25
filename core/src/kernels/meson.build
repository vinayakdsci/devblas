subdir('avx2')

libdevblas_kernels = static_library(
	'devblas-kernels',

	[
		'dispatch.c',
	],

	include_directories : inc,
	c_args : opt,

	dependencies : [
		devblas_avx2,
	],
)

devblas_kernels = declare_dependency(
	link_with : libdevblas_kernels,

	dependencies : [
		devblas_avx2,
	],
)
