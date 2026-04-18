#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/matmul.h"
#include "nemequ/munit.h"
#include "test_matmul_simd.h"

static void ref_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C) {
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) sum += (int)A[i * n + k] * (int)B[k * p + j];
      if (sum > 255) sum = 255;
      if (sum < 0) sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
}

static MunitResult test_u8_i8_u8_small(const char *name,
                                       int (*matmul_fn)(size_t, size_t, size_t, const uint8_t *, const int8_t *,
                                                        uint8_t *, double),
                                       double epsilon) {
  uint8_t A[] = {1, 2, 3, 4, 5, 6};
  int8_t  B[] = {1, 0, 0, 1, 0, 0};
  uint8_t C[4], E[4];

  ref_u8_i8_u8(2, 3, 2, A, B, E);
  matmul_fn(2, 3, 2, A, B, C, 0.0);

  for (int i = 0; i < 4; i++) {
    int d = E[i] - C[i];
    if (d < 0) d = -d;
    if (d > (int)epsilon) return MUNIT_FAIL;
  }
  return MUNIT_OK;
}

static MunitResult test_u8_i8_u8_medium(const char *name,
                                        int (*matmul_fn)(size_t, size_t, size_t, const uint8_t *, const int8_t *,
                                                         uint8_t *, double),
                                        double epsilon) {
  const size_t m = 64, n = 64, p = 64;
  uint8_t     *A = malloc(m * n);
  int8_t      *B = malloc(n * p);
  uint8_t     *C = malloc(m * p);
  uint8_t     *E = malloc(m * p);
  if (!A || !B || !C || !E) {
    free(A);
    free(B);
    free(C);
    free(E);
    return MUNIT_SKIP;
  }

  // Deterministic pseudo-random values
  for (size_t i = 0; i < m * n; i++) A[i] = (uint8_t)((i * 7 + 13) % 251);
  for (size_t i = 0; i < n * p; i++) B[i] = (int8_t)(((i * 11 + 17) % 211) - 105);
  memset(C, 0, m * p);
  memset(E, 0, m * p);

  ref_u8_i8_u8(m, n, p, A, B, E);
  matmul_fn(m, n, p, A, B, C, 0.0);

  for (size_t i = 0; i < m * p; i++) {
    int d = (int)E[i] - (int)C[i];
    if (d < 0) d = -d;
    if (d > (int)epsilon) {
      free(A);
      free(B);
      free(C);
      free(E);
      return MUNIT_FAIL;
    }
  }

  free(A);
  free(B);
  free(C);
  free(E);
  return MUNIT_OK;
}

static MunitResult test_u8_i8_u8(const char *name,
                                 int (*matmul_fn)(size_t, size_t, size_t, const uint8_t *, const int8_t *, uint8_t *,
                                                  double),
                                 double epsilon) {
  return test_u8_i8_u8_small(name, matmul_fn, epsilon);
}

static MunitResult test_scalar_u8_i8_u8(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8("scalar", matmul_scalar_u8_i8_u8, 0);
}

static MunitResult test_scalar_u8_i8_u8_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8_medium("scalar", matmul_scalar_u8_i8_u8, 0);
}

#ifdef __AVX512VNNI__
static MunitResult test_avx512vnni_u8_i8_u8(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8("avx512vnni", matmul_avx512vnni_u8_i8_u8, 0);
}

static MunitResult test_avx512vnni_u8_i8_u8_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8_medium("avx512vnni", matmul_avx512vnni_u8_i8_u8, 0);
}
#endif

static MunitResult test_dispatched_u8_i8_u8(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8("dispatched", matmul_u8_i8_u8, 0);
}

static MunitResult test_dispatched_u8_i8_u8_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8_medium("dispatched", matmul_u8_i8_u8, 0);
}

static MunitTest tests[] = {
    {"/scalar-u8-i8-u8", test_scalar_u8_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-i8-u8-medium", test_scalar_u8_i8_u8_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
#ifdef __AVX512VNNI__
    {"/avx512vnni-u8-i8-u8", test_avx512vnni_u8_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512vnni-u8-i8-u8-medium", test_avx512vnni_u8_i8_u8_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
#endif
    {"/dispatched-u8-i8-u8", test_dispatched_u8_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-u8-i8-u8-medium", test_dispatched_u8_i8_u8_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL}};

static const MunitSuite suite = {"/matmul", tests, NULL, 1, MUNIT_SUITE_OPTION_NONE};

int main(int argc, char *argv[MUNIT_ARRAY_PARAM(argc)]) {
  return munit_suite_main(&suite, NULL, argc, argv);
}
