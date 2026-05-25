#pragma once

#include <stdint.h>
#include <stdbool.h>

typedef enum
{
	DEVBLAS_LAYOUT_ROW_MAJOR,
	DEVBLAS_LAYOUT_COL_MAJOR,
} devblas_layout_t;

typedef enum
{
	DEVBLAS_ISA_AVX2,
	DEVBLAS_ISA_NEON,
	DEVBLAS_ISA_GENERIC,

	// This is a sentinel enum token
	// meant to track supported ISA count.
	// It should ALWAYS be the last enum
	// element for this reason.
	DEVBLAS_ISA_CNT,
} devblas_isa_t;

typedef enum
{
	DEVBLAS_STATUS_SUCCESS,
	DEVBLAS_STATUS_INVALID_PTR,
	DEVBLAS_STATUS_INVALID_VALUE,
	DEVBLAS_STATUS_UNSUPPORTED,
} devblas_status_t;

typedef bool devblas_transpose_t;

typedef int64_t devblas_i64_t;
typedef int32_t devblas_i32_t;
typedef float devblas_f32_t;
typedef double devblas_f64_t;
