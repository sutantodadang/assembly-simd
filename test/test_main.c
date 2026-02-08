/**
 * @file test_main.c
 * @brief Test harness for SIMD matrix multiplication
 *
 * This test program verifies the correctness of AVX2 and AVX-512
 * matrix multiplication implementations by comparing their results
 * against a scalar reference implementation.
 *
 * Tests include:
 *   - Various matrix sizes (aligned and unaligned to SIMD width)
 *   - Correctness verification with tolerance for FP rounding
 *   - Performance benchmarking
 *
 * @author SIMD Matrix Multiplication Project
 * @license MIT
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

/* Include our header from the include directory */
#include "../src/include/matrix_mult.h"

/*============================================================================
 * TEST CONFIGURATION
 *============================================================================*/

/* Maximum acceptable difference between SIMD and scalar results */
#define TOLERANCE 1e-4f

/* Number of benchmark iterations for timing */
#define BENCHMARK_ITERS 10

/* Test matrix sizes */
typedef struct {
  size_t M;
  size_t N;
  size_t K;
  const char *description;
} TestCase;

static TestCase test_cases[] = {
    /* Small matrices */
    {4, 4, 4, "4×4 (tiny)"},
    {8, 8, 8, "8×8 (one AVX2 vector)"},
    {16, 16, 16, "16×16 (one AVX-512 vector)"},

    /* Medium matrices (power of 2) */
    {32, 32, 32, "32×32"},
    {64, 64, 64, "64×64"},
    {128, 128, 128, "128×128"},
    {256, 256, 256, "256×256"},

    /* Non-power-of-2 (edge case handling) */
    {13, 7, 9, "13×7×9 (prime dimensions)"},
    {17, 23, 19, "17×23×19 (odd primes)"},
    {100, 100, 100, "100×100"},
    {127, 127, 127, "127×127 (2^7 - 1)"},

    /* Rectangular matrices */
    {32, 64, 48, "32×64×48 (tall result)"},
    {64, 32, 48, "64×32×48 (wide input A)"},
    {64, 48, 32, "64×48×32 (narrow inner)"},

    /* Large matrices (for benchmarking) */
    {512, 512, 512, "512×512"},
    /* Uncomment for larger benchmarks: */
    /* {1024, 1024, 1024, "1024×1024"}, */
};

static const size_t NUM_TEST_CASES = sizeof(test_cases) / sizeof(test_cases[0]);

/*============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

/**
 * Get current time in seconds (high resolution)
 */
static double get_time(void) {
#if defined(_WIN32)
  LARGE_INTEGER freq, count;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&count);
  return (double)count.QuadPart / (double)freq.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/**
 * Calculate GFLOPS for matrix multiplication
 * matmul has 2*M*N*K floating point operations (multiply + add)
 */
static double calc_gflops(size_t M, size_t N, size_t K, double seconds) {
  double flops = 2.0 * (double)M * (double)N * (double)K;
  return (flops / 1e9) / seconds;
}

/**
 * Print a small matrix (for debugging)
 */
static void print_matrix(const char *name, const float *matrix, size_t M,
                         size_t N, size_t max_display) {
  printf("%s (%zu×%zu):\n", name, M, N);
  size_t rows = (M < max_display) ? M : max_display;
  size_t cols = (N < max_display) ? N : max_display;

  for (size_t i = 0; i < rows; i++) {
    printf("  [");
    for (size_t j = 0; j < cols; j++) {
      printf("%8.4f", matrix[i * N + j]);
      if (j < cols - 1)
        printf(", ");
    }
    if (cols < N)
      printf(", ...");
    printf("]\n");
  }
  if (rows < M)
    printf("  ...\n");
  printf("\n");
}

/*============================================================================
 * TEST FUNCTIONS
 *============================================================================*/

/**
 * Run a single test case
 */
static int run_test(const TestCase *tc, int have_avx2, int have_avx512) {
  int passed = 1;
  size_t M = tc->M;
  size_t N = tc->N;
  size_t K = tc->K;

  printf("Testing %s matrix multiplication...\n", tc->description);

  /* Allocate matrices with maximum alignment (64-byte for AVX-512) */
  float *A = matrix_alloc(M, K, ALIGN_AVX512);
  float *B = matrix_alloc(K, N, ALIGN_AVX512);
  float *C_scalar = matrix_alloc(M, N, ALIGN_AVX512);
  float *C_avx2 = matrix_alloc(M, N, ALIGN_AVX512);
  float *C_avx512 = matrix_alloc(M, N, ALIGN_AVX512);

  if (!A || !B || !C_scalar || !C_avx2 || !C_avx512) {
    printf("  ERROR: Memory allocation failed!\n");
    passed = 0;
    goto cleanup;
  }

  /* Initialize matrices with reproducible random values */
  matrix_init_random(A, M, K, 42);
  matrix_init_random(B, K, N, 123);

  /* Compute reference result with scalar implementation */
  memset(C_scalar, 0, M * N * sizeof(float));
  matrix_mult_scalar(A, B, C_scalar, M, N, K);

  /* Test AVX2 implementation */
  if (have_avx2) {
    memset(C_avx2, 0, M * N * sizeof(float));
    matrix_mult_avx2(A, B, C_avx2, M, N, K);

    float max_diff = matrix_max_diff(C_scalar, C_avx2, M, N);
    if (max_diff <= TOLERANCE) {
      printf("  [PASS] AVX2:    PASSED (max error: %.6e)\n", max_diff);
    } else {
      printf("  [FAIL] AVX2:    FAILED (max error: %.6e > tolerance %.6e)\n",
             max_diff, TOLERANCE);
      passed = 0;

      /* For debugging small matrices */
      if (M <= 8 && N <= 8) {
        print_matrix("Expected (scalar)", C_scalar, M, N, 8);
        print_matrix("Got (AVX2)", C_avx2, M, N, 8);
      }
    }
  } else {
    printf("  - AVX2:    SKIPPED (not supported)\n");
  }

  /* Test AVX-512 implementation */
  if (have_avx512) {
    memset(C_avx512, 0, M * N * sizeof(float));
    matrix_mult_avx512(A, B, C_avx512, M, N, K);

    float max_diff = matrix_max_diff(C_scalar, C_avx512, M, N);
    if (max_diff <= TOLERANCE) {
      printf("  [PASS] AVX-512: PASSED (max error: %.6e)\n", max_diff);
    } else {
      printf("  [FAIL] AVX-512: FAILED (max error: %.6e > tolerance %.6e)\n",
             max_diff, TOLERANCE);
      passed = 0;
    }
  } else {
    printf("  - AVX-512: SKIPPED (not supported)\n");
  }

cleanup:
  matrix_free(A);
  matrix_free(B);
  matrix_free(C_scalar);
  matrix_free(C_avx2);
  matrix_free(C_avx512);

  return passed;
}

/**
 * Run benchmark for a test case
 */
static void run_benchmark(const TestCase *tc, int have_avx2, int have_avx512) {
  size_t M = tc->M;
  size_t N = tc->N;
  size_t K = tc->K;
  double t_start, t_end, t_elapsed;

  /* Allocate aligned matrices */
  float *A = matrix_alloc(M, K, ALIGN_AVX512);
  float *B = matrix_alloc(K, N, ALIGN_AVX512);
  float *C = matrix_alloc(M, N, ALIGN_AVX512);

  if (!A || !B || !C) {
    printf("  ERROR: Memory allocation failed!\n");
    goto cleanup;
  }

  matrix_init_random(A, M, K, 42);
  matrix_init_random(B, K, N, 123);

  printf("\nBenchmark: %s\n", tc->description);
  printf("%-12s %12s %12s\n", "Method", "Time (ms)", "GFLOPS");
  printf("-------------------------------------\n");

  /* Benchmark scalar */
  t_start = get_time();
  for (int i = 0; i < BENCHMARK_ITERS; i++) {
    matrix_mult_scalar(A, B, C, M, N, K);
  }
  t_end = get_time();
  t_elapsed = (t_end - t_start) / BENCHMARK_ITERS;
  printf("%-12s %12.3f %12.2f\n", "Scalar", t_elapsed * 1000,
         calc_gflops(M, N, K, t_elapsed));

  /* Benchmark AVX2 */
  if (have_avx2) {
    t_start = get_time();
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
      matrix_mult_avx2(A, B, C, M, N, K);
    }
    t_end = get_time();
    t_elapsed = (t_end - t_start) / BENCHMARK_ITERS;
    printf("%-12s %12.3f %12.2f\n", "AVX2", t_elapsed * 1000,
           calc_gflops(M, N, K, t_elapsed));
  }

  /* Benchmark AVX-512 */
  if (have_avx512) {
    t_start = get_time();
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
      matrix_mult_avx512(A, B, C, M, N, K);
    }
    t_end = get_time();
    t_elapsed = (t_end - t_start) / BENCHMARK_ITERS;
    printf("%-12s %12.3f %12.2f\n", "AVX-512", t_elapsed * 1000,
           calc_gflops(M, N, K, t_elapsed));
  }

cleanup:
  matrix_free(A);
  matrix_free(B);
  matrix_free(C);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(int argc, char *argv[]) {
  int benchmark_mode = 0;

  /* Parse arguments */
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--benchmark") == 0 || strcmp(argv[i], "-b") == 0) {
      benchmark_mode = 1;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("Usage: %s [OPTIONS]\n", argv[0]);
      printf("Options:\n");
      printf("  -b, --benchmark    Run benchmarks after tests\n");
      printf("  -h, --help         Show this help message\n");
      return 0;
    }
  }

  printf(
      "+================================================================+\n");
  printf("|       SIMD Matrix Multiplication Test Suite                  |\n");
  printf(
      "+================================================================+\n\n");

  /* Check CPU features */
  int have_avx2 = check_avx2_support();
  int have_avx512 = check_avx512_support();

  printf("CPU Feature Detection:\n");
  printf("  AVX2 + FMA:   %s\n",
         have_avx2 ? "[OK] Supported" : "[--] Not supported");
  printf("  AVX-512F:     %s\n",
         have_avx512 ? "[OK] Supported" : "[--] Not supported");
  printf("\n");

  if (!have_avx2 && !have_avx512) {
    printf(
        "WARNING: No SIMD support detected. Only scalar tests will run.\n\n");
  }

  /* Run tests */
  printf("================================================================\n");
  printf("                      CORRECTNESS TESTS                        \n");
  printf(
      "================================================================\n\n");

  int all_passed = 1;
  int tests_run = 0;
  int tests_passed = 0;

  for (size_t i = 0; i < NUM_TEST_CASES; i++) {
    tests_run++;
    if (run_test(&test_cases[i], have_avx2, have_avx512)) {
      tests_passed++;
    } else {
      all_passed = 0;
    }
    printf("\n");
  }

  /* Summary */
  printf("================================================================\n");
  printf("                         SUMMARY                               \n");
  printf(
      "================================================================\n\n");

  printf("Tests passed: %d / %d\n", tests_passed, tests_run);

  if (all_passed) {
    printf("\n[PASS] All tests PASSED!\n");
  } else {
    printf("\n[FAIL] Some tests FAILED!\n");
  }

  /* Run benchmarks if requested */
  if (benchmark_mode) {
    printf(
        "\n================================================================\n");
    printf("                        BENCHMARKS                             \n");
    printf(
        "================================================================\n");

    /* Only benchmark larger matrices */
    for (size_t i = 0; i < NUM_TEST_CASES; i++) {
      if (test_cases[i].M >= 64 && test_cases[i].N >= 64) {
        run_benchmark(&test_cases[i], have_avx2, have_avx512);
      }
    }
  }

  return all_passed ? 0 : 1;
}
