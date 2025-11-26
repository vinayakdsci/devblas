cmake_minimum_required(VERSION 3.28)
project(devBlas)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(src)

option(DEVBLAS_BUILD_BENCH OFF)
option(DEVBLAS_FAST_MATH "Build with -ffast-math to more aggressively optimize floating point operations. Might affect numerical stability and standards compliance" OFF)
if(DEVBLAS_FAST_MATH)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3")

if (DEVBLAS_BUILD_BENCH)
	add_executable(bench_driver bench_driver.c)
	# target_link_libraries(bench_driver devblas)
	target_link_libraries(bench_driver devblas)
endif()


enable_testing()
add_subdirectory(tests)
