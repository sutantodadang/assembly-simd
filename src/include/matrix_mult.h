/**
 * @file matrix_mult.h
 * @brief SIMD Matrix Multiplication Public API
 *
 * This header provides the C interface for high-performance matrix
 * multiplication using AVX2 and AVX-512 SIMD instructions.
 *
 * @section Usage
 * @code
 * #include "matrix_mult.h"
 *
 * // Check CPU support
 * if (check_avx512_support()) {
 *     matrix_mult_avx512(A, B, C, M, N, K);
 * } else if (check_avx2_support()) {
 *     matrix_mult_avx2(A, B, C, M, N, K);
 * } else {
 *     matrix_mult_scalar(A, B, C, M, N, K);
 * }
 * @endcode
 *
 * @section Memory Requirements
 * - For AVX2: 32-byte alignment recommended
 * - For AVX-512: 64-byte alignment recommended
 * - Use aligned_alloc() or _aligned_malloc() for best performance
 *
 * @author SIMD Matrix Multiplication Project
 * @license MIT
 */

#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include <stddef.h> /* For size_t */
#include <stdint.h> /* For uint64_t */
#include <stdlib.h> /* For malloc, free, aligned_alloc */

#if defined(_WIN32)
#include <malloc.h> /* For _aligned_malloc, _aligned_free on Windows */
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * ALIGNMENT MACROS
 *============================================================================*/

/**
 * @def ALIGN_AVX2
 * @brief Alignment requirement for AVX2 operations (32 bytes)
 */
#define ALIGN_AVX2 32

/**
 * @def ALIGN_AVX512
 * @brief Alignment requirement for AVX-512 operations (64 bytes)
 */
#define ALIGN_AVX512 64

/**
 * @def SIMD_ALIGNED(alignment)
 * @brief Compiler-specific aligned attribute
 *
 * Usage: float SIMD_ALIGNED(32) matrix[64];
 */
#if defined(_MSC_VER)
#define SIMD_ALIGNED(x) __declspec(align(x))
#elif defined(__GNUC__) || defined(__clang__)
#define SIMD_ALIGNED(x) __attribute__((aligned(x)))
#else
#define SIMD_ALIGNED(x)
#endif

/**
 * @def ASSUME_ALIGNED(ptr, alignment)
 * @brief Compiler hint that pointer is aligned
 *
 * Helps the compiler generate better code when it knows alignment.
 */
#if defined(__GNUC__) || defined(__clang__)
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#else
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

/*============================================================================
 * MEMORY ALLOCATION HELPERS
 *============================================================================*/

/**
 * @brief Allocate aligned memory for matrix operations
 *
 * Cross-platform function to allocate memory with the required alignment
 * for SIMD operations.
 *
 * @param alignment Alignment in bytes (use ALIGN_AVX2 or ALIGN_AVX512)
 * @param size      Size in bytes to allocate
 * @return Pointer to aligned memory, or NULL on failure
 *
 * @note Must be freed with simd_free()
 */
static inline void *simd_alloc(size_t alignment, size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#elif defined(_POSIX_VERSION) && _POSIX_VERSION >= 200112L
  void *ptr = NULL;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return NULL;
  }
  return ptr;
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  return aligned_alloc(alignment, size);
#else
  /* Fallback: over-allocate and align manually */
  void *raw = malloc(size + alignment + sizeof(void *));
  if (!raw)
    return NULL;
  void **aligned = (void **)(((uintptr_t)raw + alignment + sizeof(void *)) &
                             ~(alignment - 1));
  aligned[-1] = raw;
  return aligned;
#endif
}

/**
 * @brief Free memory allocated with simd_alloc()
 *
 * @param ptr Pointer previously returned by simd_alloc()
 */
static inline void simd_free(void *ptr) {
  if (!ptr)
    return;
#if defined(_WIN32)
  _aligned_free(ptr);
#elif defined(_POSIX_VERSION) && _POSIX_VERSION >= 200112L ||                  \
    (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
  free(ptr);
#else
  free(((void **)ptr)[-1]);
#endif
}

/**
 * @brief Allocate a float matrix with proper alignment
 *
 * @param rows      Number of rows
 * @param cols      Number of columns
 * @param alignment Alignment (use ALIGN_AVX2 or ALIGN_AVX512)
 * @return Pointer to aligned float array, or NULL on failure
 */
static inline float *matrix_alloc(size_t rows, size_t cols, size_t alignment) {
  return (float *)simd_alloc(alignment, rows * cols * sizeof(float));
}

/**
 * @brief Free a matrix allocated with matrix_alloc()
 */
static inline void matrix_free(float *matrix) { simd_free(matrix); }

/*============================================================================
 * CPU FEATURE DETECTION
 *============================================================================*/

/**
 * @brief Check if CPU supports AVX2 and FMA instructions
 *
 * This function checks both the CPU capability via CPUID and whether
 * the operating system has enabled AVX support (XCR0).
 *
 * @return Non-zero if AVX2+FMA is supported, zero otherwise
 */
extern int check_avx2_support(void);

/**
 * @brief Check if CPU supports AVX-512F (Foundation) instructions
 *
 * This function checks for AVX-512F support and whether the OS has
 * enabled the AVX-512 state saving (which requires more context space).
 *
 * @return Non-zero if AVX-512F is supported, zero otherwise
 */
extern int check_avx512_support(void);

/*============================================================================
 * MATRIX MULTIPLICATION FUNCTIONS
 *============================================================================*/

/**
 * @brief Matrix multiplication using AVX2 SIMD instructions
 *
 * Computes C = A × B where:
 *   - A is an M×K matrix
 *   - B is a K×N matrix
 *   - C is an M×N matrix (output)
 *
 * This function uses 256-bit YMM registers to process 8 floats
 * simultaneously, with FMA (Fused Multiply-Add) instructions.
 *
 * @param A Pointer to matrix A, row-major order, 32-byte alignment recommended
 * @param B Pointer to matrix B, row-major order, 32-byte alignment recommended
 * @param C Pointer to result matrix C, row-major order, 32-byte alignment
 * recommended
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / Number of rows in B
 *
 * @pre All pointers must be non-NULL
 * @pre M, N, K must be greater than 0
 * @post Matrix C contains the result A × B
 *
 * @note For matrices with N that is a multiple of 8 and aligned to 32 bytes,
 *       use matrix_mult_avx2_aligned() for better performance.
 */
extern void matrix_mult_avx2(const float *A, const float *B, float *C, size_t M,
                             size_t N, size_t K);

/**
 * @brief Matrix multiplication using AVX2 (aligned, optimized version)
 *
 * Optimized version that assumes:
 *   - All matrices are 32-byte aligned
 *   - N is a multiple of 8
 *
 * Skips edge handling for maximum performance.
 *
 * @see matrix_mult_avx2
 */
extern void matrix_mult_avx2_aligned(const float *A, const float *B, float *C,
                                     size_t M, size_t N, size_t K);

/**
 * @brief Matrix multiplication using AVX-512 SIMD instructions
 *
 * Computes C = A × B using 512-bit ZMM registers to process 16 floats
 * simultaneously.
 *
 * Features:
 *   - Embedded broadcast (fewer instructions)
 *   - Mask registers for elegant edge handling
 *   - 32 vector registers for extensive blocking
 *
 * @param A Pointer to matrix A, row-major order, 64-byte alignment recommended
 * @param B Pointer to matrix B, row-major order, 64-byte alignment recommended
 * @param C Pointer to result matrix C, row-major order, 64-byte alignment
 * recommended
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / Number of rows in B
 *
 * @pre CPU must support AVX-512F (check with check_avx512_support())
 * @post Matrix C contains the result A × B
 *
 * @note For matrices with N that is a multiple of 16 and aligned to 64 bytes,
 *       use matrix_mult_avx512_aligned() for better performance.
 */
extern void matrix_mult_avx512(const float *A, const float *B, float *C,
                               size_t M, size_t N, size_t K);

/**
 * @brief Matrix multiplication using AVX-512 (aligned, optimized version)
 *
 * Optimized version that assumes:
 *   - All matrices are 64-byte aligned
 *   - N is a multiple of 16
 *
 * @see matrix_mult_avx512
 */
extern void matrix_mult_avx512_aligned(const float *A, const float *B, float *C,
                                       size_t M, size_t N, size_t K);

/*============================================================================
 * SCALAR REFERENCE IMPLEMENTATION
 *============================================================================*/

/**
 * @brief Scalar (non-SIMD) matrix multiplication for reference/fallback
 *
 * This is a simple triple-nested loop implementation used for:
 *   - Verification of SIMD results
 *   - Fallback when SIMD is not available
 *   - Performance baseline for benchmarking
 *
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / Number of rows in B
 */
static inline void matrix_mult_scalar(const float *A, const float *B, float *C,
                                      size_t M, size_t N, size_t K) {
  /* Triple-nested loop for matrix multiplication
   *
   * Algorithm: C[i][j] = Σ(k=0 to K-1) A[i][k] × B[k][j]
   *
   * Note: This is a cache-friendly ordering (i, k, j) where:
   *   - A[i][k] is accessed sequentially along rows
   *   - B[k][j] is accessed sequentially along rows
   *   - C[i][j] is accessed sequentially (accumulated)
   */
  for (size_t i = 0; i < M; i++) {
    /* Initialize row of C to zero */
    for (size_t j = 0; j < N; j++) {
      C[i * N + j] = 0.0f;
    }

    /* Accumulate products */
    for (size_t k = 0; k < K; k++) {
      float a_ik = A[i * K + k]; /* Cache A[i][k] */
      for (size_t j = 0; j < N; j++) {
        /* C[i][j] += A[i][k] × B[k][j] */
        C[i * N + j] += a_ik * B[k * N + j];
      }
    }
  }
}

/*============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

/**
 * @brief Compare two matrices for approximate equality
 *
 * Compares matrices element-by-element with a tolerance for floating-point
 * rounding differences.
 *
 * @param A First matrix
 * @param B Second matrix
 * @param M Number of rows
 * @param N Number of columns
 * @param tolerance Maximum allowed difference per element
 * @return 1 if matrices are approximately equal, 0 otherwise
 */
static inline int matrix_compare(const float *A, const float *B, size_t M,
                                 size_t N, float tolerance) {
  for (size_t i = 0; i < M * N; i++) {
    float diff = A[i] - B[i];
    if (diff < 0)
      diff = -diff; /* Absolute value */
    if (diff > tolerance) {
      return 0; /* Difference too large */
    }
  }
  return 1; /* All elements within tolerance */
}

/**
 * @brief Find the maximum difference between two matrices
 *
 * @param A First matrix
 * @param B Second matrix
 * @param M Number of rows
 * @param N Number of columns
 * @return Maximum absolute difference between corresponding elements
 */
static inline float matrix_max_diff(const float *A, const float *B, size_t M,
                                    size_t N) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < M * N; i++) {
    float diff = A[i] - B[i];
    if (diff < 0)
      diff = -diff;
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

/**
 * @brief Initialize matrix with random values between 0 and 1
 *
 * @param matrix Pointer to matrix
 * @param M Number of rows
 * @param N Number of columns
 * @param seed Random seed (for reproducibility)
 */
static inline void matrix_init_random(float *matrix, size_t M, size_t N,
                                      unsigned int seed) {
  /* Simple LCG (Linear Congruential Generator) for reproducible random values
   */
  uint64_t state = seed;
  for (size_t i = 0; i < M * N; i++) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t value = (uint32_t)(state >> 33);
    matrix[i] = (float)value / (float)UINT32_MAX; /* Normalize to [0, 1] */
  }
}

/**
 * @brief Check if a pointer is aligned to the specified boundary
 *
 * @param ptr Pointer to check
 * @param alignment Required alignment (must be power of 2)
 * @return 1 if aligned, 0 otherwise
 */
static inline int is_aligned(const void *ptr, size_t alignment) {
  return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

#ifdef __cplusplus
}
#endif

#endif /* MATRIX_MULT_H */
