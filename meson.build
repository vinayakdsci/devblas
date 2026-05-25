project(
	'devblas',
	'c',

	default_options : [
		'buildtype=release',
		'warning_level=3',
		'c_std=c11',
	],
)

opt = [
	'-O3',
	'-ffast-math',
	'-fno-math-errno',
	'-fno-trapping-math',
	'-fstrict-aliasing',
	'-fomit-frame-pointer',
	'-funroll-loops',
	'-fvisibility=hidden',
]

inc = include_directories(
	'core/include',
)

subdir('core')

executable(
	'bench_driver',
	'bench_driver.c',
	dependencies : devblas_core,
)
