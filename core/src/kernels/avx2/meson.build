libdevblas_avx2 = static_library(
	'devblas-avx2',

	files(
		'kernel_8x8_sgemm.c',
	),

	include_directories : inc,

	c_args : opt + [
		'-mavx2',
		'-mfma',
	],
)

devblas_avx2 = declare_dependency(
	link_with : libdevblas_avx2,
)
