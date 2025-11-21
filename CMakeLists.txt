cmake_minimum_required(VERSION 3.28)
project(devBlas LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_subdirectory(src)

option(DEVBLAS_BUILD_BENCH OFF)

if (DEVBLAS_BUILD_BENCH)
	add_executable(bench_driver bench_driver.cc)
	# target_link_libraries(bench_driver devblas)
	target_link_libraries(bench_driver devblas)
endif()


enable_testing()
add_subdirectory(tests)
