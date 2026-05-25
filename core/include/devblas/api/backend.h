#pragma once

#include <string.h>
#include <devblas/api/typedecls.h>

#define DEVBLAS_REGISTER_DISPATCH_FPTR(T, prefix)                \
	typedef void (*devblas_internal_##prefix##gemm_dispatch_fn)( \
			devblas_layout_t layout, devblas_transpose_t transA, \
			devblas_transpose_t transB, devblas_i32_t M,         \
			devblas_i32_t N, devblas_i32_t K,                    \
			const T *const restrict A, devblas_i32_t lda,        \
			const T *const restrict B, devblas_i32_t ldb,        \
			T *const restrict C, devblas_i32_t ldc);

#define DEVBLAS_REGISTER_GEMM_DISPATCHES(X) \
	X(devblas_f32_t, s)                     \
	X(devblas_i32_t, i)

DEVBLAS_REGISTER_GEMM_DISPATCHES(DEVBLAS_REGISTER_DISPATCH_FPTR)

#define DEVBLAS_REGISTER_UKERNEL_FPTR(T, prefix)                  \
	typedef void (*devblas_internal_##prefix##gemm_ukernel_fn)(   \
			const T *const restrict A, const T *const restrict B, \
			T *const restrict C, devblas_i32_t K, devblas_i32_t ldc);

#define DEVBLAS_REGISTER_UKERNELS(X) \
	X(devblas_f32_t, s)              \
	X(devblas_i32_t, i)

DEVBLAS_REGISTER_UKERNELS(DEVBLAS_REGISTER_UKERNEL_FPTR)

inline devblas_isa_t
devblas_cpu_isa(void)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
	// TODO: We fallback to generic if anything except avx2 is
	// supported. This should be handled as hardware becomes available.
	return __builtin_cpu_supports("avx2") ? DEVBLAS_ISA_AVX2
										  : DEVBLAS_ISA_GENERIC;
#elif defined(__aarch64__) || defined(_M_ARM64)
	return DEVBLAS_ISA_NEON;
#else
	return DEVBLAS_ISA_GENERIC;
#endif
}

#define MAX_MR 16
#define MAX_NR 16

typedef struct
{
	devblas_internal_sgemm_ukernel_fn sgemm_ukernels[MAX_MR][MAX_NR];
	devblas_internal_igemm_ukernel_fn igemm_ukernels[MAX_MR][MAX_NR];
} devblas_gemm_backend_t;

extern const devblas_gemm_backend_t
		*devblas_cpu_backend_registry[DEVBLAS_ISA_CNT];

static inline void
devblas_backend_vtable_init(devblas_gemm_backend_t *backend,
							devblas_isa_t isa)
{
	devblas_cpu_backend_registry[isa] = backend;
}

static inline const devblas_gemm_backend_t *
devblas_get_active_backend(devblas_isa_t isa)
{
	return devblas_cpu_backend_registry[isa];
}

#define DEVBLAS_REGISTER_BACKEND(isa, backend)          \
	void devblas_register_##backend(void)               \
	{                                                   \
		devblas_cpu_backend_registry[isa] = &(backend); \
	}
