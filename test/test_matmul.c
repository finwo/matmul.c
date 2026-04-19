#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/matmul.h"
#include "nemequ/munit.h"
#include "test_matmul_simd.h"

static void ref_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale) {
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) sum += (int)A[i * n + k] * (int)B[k * p + j];
      if (scale > 1.0) sum = (int)(sum / scale);
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

  ref_u8_i8_u8(2, 3, 2, A, B, E, 0.0);
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

  ref_u8_i8_u8(m, n, p, A, B, E, 0.0);
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

static MunitResult test_u8_i8_u8_scaled_small(const char *name,
                                              int (*matmul_fn)(size_t, size_t, size_t, const uint8_t *, const int8_t *,
                                                               uint8_t *, double),
                                              double epsilon) {
  uint8_t A[] = {8, 16, 24, 32, 40, 48};
  int8_t  B[] = {2, 0, 0, 2, 0, 0};
  uint8_t C[4], E[4];

  ref_u8_i8_u8(2, 3, 2, A, B, E, 4.0);
  matmul_fn(2, 3, 2, A, B, C, 4.0);

  for (int i = 0; i < 4; i++) {
    int d = E[i] - C[i];
    if (d < 0) d = -d;
    if (d > (int)epsilon) return MUNIT_FAIL;
  }
  return MUNIT_OK;
}

static MunitResult test_u8_i8_u8_scaled_medium(const char *name,
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

  for (size_t i = 0; i < m * n; i++) A[i] = (uint8_t)((i * 7 + 13) % 251);
  for (size_t i = 0; i < n * p; i++) B[i] = (int8_t)(((i * 11 + 17) % 211) - 105);
  memset(C, 0, m * p);
  memset(E, 0, m * p);

  ref_u8_i8_u8(m, n, p, A, B, E, 8.0);
  matmul_fn(m, n, p, A, B, C, 8.0);

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

static MunitResult test_scalar_u8_i8_u8_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8_scaled_small("scalar", matmul_scalar_u8_i8_u8, 0);
}

static MunitResult test_scalar_u8_i8_u8_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8_scaled_medium("scalar", matmul_scalar_u8_i8_u8, 0);
}

#ifdef __AVX512VNNI__
static MunitResult test_avx512vnni_u8_i8_u8_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8_scaled_small("avx512vnni", matmul_avx512vnni_u8_i8_u8, 0);
}

static MunitResult test_avx512vnni_u8_i8_u8_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8_scaled_medium("avx512vnni", matmul_avx512vnni_u8_i8_u8, 0);
}
#endif

static MunitResult test_dispatched_u8_i8_u8_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8_scaled_small("dispatched", matmul_u8_i8_u8, 0);
}

static MunitResult test_dispatched_u8_i8_u8_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_u8_i8_u8_scaled_medium("dispatched", matmul_u8_i8_u8, 0);
}

/* ========================================================================== */
/* f32_f32_f32 tests                                                          */
/* ========================================================================== */

static void ref_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < p; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) sum += (double)A[i * n + k] * (double)B[k * p + j];
      if (scale > 1.0) sum /= scale;
      C[i * p + j] = (float)sum;
    }
}

static MunitResult test_f32_f32_f32_small(const char *name,
                                          int (*matmul_fn)(size_t, size_t, size_t, const float *, const float *,
                                                           float *, double),
                                          double epsilon) {
  float A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float B[] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  float C[4], E[4];

  ref_f32_f32_f32(2, 3, 2, A, B, E, 0.0);
  matmul_fn(2, 3, 2, A, B, C, 0.0);

  for (int i = 0; i < 4; i++) {
    double d = (double)E[i] - (double)C[i];
    if (d < 0) d = -d;
    if (d > epsilon) return MUNIT_FAIL;
  }
  return MUNIT_OK;
}

static MunitResult test_f32_f32_f32_medium(const char *name,
                                           int (*matmul_fn)(size_t, size_t, size_t, const float *, const float *,
                                                            float *, double),
                                           double epsilon) {
  const size_t m = 64, n = 64, p = 64;
  float       *A = malloc(m * n * sizeof(float));
  float       *B = malloc(n * p * sizeof(float));
  float       *C = malloc(m * p * sizeof(float));
  float       *E = malloc(m * p * sizeof(float));
  if (!A || !B || !C || !E) {
    free(A);
    free(B);
    free(C);
    free(E);
    return MUNIT_SKIP;
  }

  for (size_t i = 0; i < m * n; i++) A[i] = (float)((i * 7 + 13) % 251);
  for (size_t i = 0; i < n * p; i++) B[i] = (float)(((i * 11 + 17) % 211) - 105);
  memset(C, 0, m * p * sizeof(float));
  memset(E, 0, m * p * sizeof(float));

  ref_f32_f32_f32(m, n, p, A, B, E, 0.0);
  matmul_fn(m, n, p, A, B, C, 0.0);

  for (size_t i = 0; i < m * p; i++) {
    double d = (double)E[i] - (double)C[i];
    if (d < 0) d = -d;
    if (d > epsilon) {
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

static MunitResult test_f32_f32_f32_scaled_small(const char *name,
                                                 int (*matmul_fn)(size_t, size_t, size_t, const float *, const float *,
                                                                  float *, double),
                                                 double epsilon) {
  float A[] = {8.0f, 16.0f, 24.0f, 32.0f, 40.0f, 48.0f};
  float B[] = {2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f};
  float C[4], E[4];

  ref_f32_f32_f32(2, 3, 2, A, B, E, 4.0);
  matmul_fn(2, 3, 2, A, B, C, 4.0);

  for (int i = 0; i < 4; i++) {
    double d = (double)E[i] - (double)C[i];
    if (d < 0) d = -d;
    if (d > epsilon) return MUNIT_FAIL;
  }
  return MUNIT_OK;
}

static MunitResult test_f32_f32_f32_scaled_medium(const char *name,
                                                  int (*matmul_fn)(size_t, size_t, size_t, const float *, const float *,
                                                                   float *, double),
                                                  double epsilon) {
  const size_t m = 64, n = 64, p = 64;
  float       *A = malloc(m * n * sizeof(float));
  float       *B = malloc(n * p * sizeof(float));
  float       *C = malloc(m * p * sizeof(float));
  float       *E = malloc(m * p * sizeof(float));
  if (!A || !B || !C || !E) {
    free(A);
    free(B);
    free(C);
    free(E);
    return MUNIT_SKIP;
  }

  for (size_t i = 0; i < m * n; i++) A[i] = (float)((i * 7 + 13) % 251);
  for (size_t i = 0; i < n * p; i++) B[i] = (float)(((i * 11 + 17) % 211) - 105);
  memset(C, 0, m * p * sizeof(float));
  memset(E, 0, m * p * sizeof(float));

  ref_f32_f32_f32(m, n, p, A, B, E, 8.0);
  matmul_fn(m, n, p, A, B, C, 8.0);

  for (size_t i = 0; i < m * p; i++) {
    double d = (double)E[i] - (double)C[i];
    if (d < 0) d = -d;
    if (d > epsilon) {
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

static MunitResult test_scalar_f32_f32_f32(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_small("scalar", matmul_scalar_f32_f32_f32, 1e-5);
}

static MunitResult test_scalar_f32_f32_f32_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_medium("scalar", matmul_scalar_f32_f32_f32, 1e-3);
}

static MunitResult test_scalar_f32_f32_f32_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_scaled_small("scalar", matmul_scalar_f32_f32_f32, 1e-5);
}

static MunitResult test_scalar_f32_f32_f32_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_scaled_medium("scalar", matmul_scalar_f32_f32_f32, 1e-3);
}

#ifdef __AVX2__
static MunitResult test_avx2_f32_f32_f32(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_small("avx2", matmul_avx2_f32_f32_f32, 1e-5);
}

static MunitResult test_avx2_f32_f32_f32_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_medium("avx2", matmul_avx2_f32_f32_f32, 1e-3);
}

static MunitResult test_avx2_f32_f32_f32_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_scaled_small("avx2", matmul_avx2_f32_f32_f32, 1e-5);
}

static MunitResult test_avx2_f32_f32_f32_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_scaled_medium("avx2", matmul_avx2_f32_f32_f32, 1e-3);
}
#endif

#ifdef __AVX512F__
static MunitResult test_avx512_f32_f32_f32(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_small("avx512", matmul_avx512_f32_f32_f32, 1e-5);
}

static MunitResult test_avx512_f32_f32_f32_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_medium("avx512", matmul_avx512_f32_f32_f32, 1e-3);
}

static MunitResult test_avx512_f32_f32_f32_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_scaled_small("avx512", matmul_avx512_f32_f32_f32, 1e-5);
}

static MunitResult test_avx512_f32_f32_f32_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_scaled_medium("avx512", matmul_avx512_f32_f32_f32, 1e-3);
}
#endif

static MunitResult test_dispatched_f32_f32_f32(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_small("dispatched", matmul_f32_f32_f32, 1e-5);
}

static MunitResult test_dispatched_f32_f32_f32_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_medium("dispatched", matmul_f32_f32_f32, 1e-3);
}

static MunitResult test_dispatched_f32_f32_f32_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_scaled_small("dispatched", matmul_f32_f32_f32, 1e-5);
}

static MunitResult test_dispatched_f32_f32_f32_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f32_f32_f32_scaled_medium("dispatched", matmul_f32_f32_f32, 1e-3);
}

/* ========================================================================== */
/* f64_f64_f64 tests                                                          */
/* ========================================================================== */

static void ref_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale) {
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < p; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) sum += A[i * n + k] * B[k * p + j];
      if (scale > 1.0) sum /= scale;
      C[i * p + j] = sum;
    }
}

static MunitResult test_f64_f64_f64_small(const char *name,
                                          int (*matmul_fn)(size_t, size_t, size_t, const double *, const double *,
                                                           double *, double),
                                          double epsilon) {
  double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double B[] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  double C[4], E[4];

  ref_f64_f64_f64(2, 3, 2, A, B, E, 0.0);
  matmul_fn(2, 3, 2, A, B, C, 0.0);

  for (int i = 0; i < 4; i++) {
    double d = E[i] - C[i];
    if (d < 0) d = -d;
    if (d > epsilon) return MUNIT_FAIL;
  }
  return MUNIT_OK;
}

static MunitResult test_f64_f64_f64_medium(const char *name,
                                           int (*matmul_fn)(size_t, size_t, size_t, const double *, const double *,
                                                            double *, double),
                                           double epsilon) {
  const size_t m = 64, n = 64, p = 64;
  double      *A = malloc(m * n * sizeof(double));
  double      *B = malloc(n * p * sizeof(double));
  double      *C = malloc(m * p * sizeof(double));
  double      *E = malloc(m * p * sizeof(double));
  if (!A || !B || !C || !E) {
    free(A);
    free(B);
    free(C);
    free(E);
    return MUNIT_SKIP;
  }

  for (size_t i = 0; i < m * n; i++) A[i] = (double)((i * 7 + 13) % 251);
  for (size_t i = 0; i < n * p; i++) B[i] = (double)(((i * 11 + 17) % 211) - 105);
  memset(C, 0, m * p * sizeof(double));
  memset(E, 0, m * p * sizeof(double));

  ref_f64_f64_f64(m, n, p, A, B, E, 0.0);
  matmul_fn(m, n, p, A, B, C, 0.0);

  for (size_t i = 0; i < m * p; i++) {
    double d = E[i] - C[i];
    if (d < 0) d = -d;
    if (d > epsilon) {
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

static MunitResult test_f64_f64_f64_scaled_small(const char *name,
                                                 int (*matmul_fn)(size_t, size_t, size_t, const double *,
                                                                  const double *, double *, double),
                                                 double epsilon) {
  double A[] = {8.0, 16.0, 24.0, 32.0, 40.0, 48.0};
  double B[] = {2.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  double C[4], E[4];

  ref_f64_f64_f64(2, 3, 2, A, B, E, 4.0);
  matmul_fn(2, 3, 2, A, B, C, 4.0);

  for (int i = 0; i < 4; i++) {
    double d = E[i] - C[i];
    if (d < 0) d = -d;
    if (d > epsilon) return MUNIT_FAIL;
  }
  return MUNIT_OK;
}

static MunitResult test_f64_f64_f64_scaled_medium(const char *name,
                                                  int (*matmul_fn)(size_t, size_t, size_t, const double *,
                                                                   const double *, double *, double),
                                                  double epsilon) {
  const size_t m = 64, n = 64, p = 64;
  double      *A = malloc(m * n * sizeof(double));
  double      *B = malloc(n * p * sizeof(double));
  double      *C = malloc(m * p * sizeof(double));
  double      *E = malloc(m * p * sizeof(double));
  if (!A || !B || !C || !E) {
    free(A);
    free(B);
    free(C);
    free(E);
    return MUNIT_SKIP;
  }

  for (size_t i = 0; i < m * n; i++) A[i] = (double)((i * 7 + 13) % 251);
  for (size_t i = 0; i < n * p; i++) B[i] = (double)(((i * 11 + 17) % 211) - 105);
  memset(C, 0, m * p * sizeof(double));
  memset(E, 0, m * p * sizeof(double));

  ref_f64_f64_f64(m, n, p, A, B, E, 8.0);
  matmul_fn(m, n, p, A, B, C, 8.0);

  for (size_t i = 0; i < m * p; i++) {
    double d = E[i] - C[i];
    if (d < 0) d = -d;
    if (d > epsilon) {
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

static MunitResult test_scalar_f64_f64_f64(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_small("scalar", matmul_scalar_f64_f64_f64, 1e-12);
}

static MunitResult test_scalar_f64_f64_f64_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_medium("scalar", matmul_scalar_f64_f64_f64, 1e-9);
}

static MunitResult test_scalar_f64_f64_f64_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_scaled_small("scalar", matmul_scalar_f64_f64_f64, 1e-12);
}

static MunitResult test_scalar_f64_f64_f64_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_scaled_medium("scalar", matmul_scalar_f64_f64_f64, 1e-9);
}

#ifdef __AVX2__
static MunitResult test_avx2_f64_f64_f64(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_small("avx2", matmul_avx2_f64_f64_f64, 1e-12);
}

static MunitResult test_avx2_f64_f64_f64_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_medium("avx2", matmul_avx2_f64_f64_f64, 1e-9);
}

static MunitResult test_avx2_f64_f64_f64_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_scaled_small("avx2", matmul_avx2_f64_f64_f64, 1e-12);
}

static MunitResult test_avx2_f64_f64_f64_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_scaled_medium("avx2", matmul_avx2_f64_f64_f64, 1e-9);
}
#endif

#ifdef __AVX512F__
static MunitResult test_avx512_f64_f64_f64(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_small("avx512", matmul_avx512_f64_f64_f64, 1e-12);
}

static MunitResult test_avx512_f64_f64_f64_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_medium("avx512", matmul_avx512_f64_f64_f64, 1e-9);
}

static MunitResult test_avx512_f64_f64_f64_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_scaled_small("avx512", matmul_avx512_f64_f64_f64, 1e-12);
}

static MunitResult test_avx512_f64_f64_f64_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_scaled_medium("avx512", matmul_avx512_f64_f64_f64, 1e-9);
}
#endif

static MunitResult test_dispatched_f64_f64_f64(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_small("dispatched", matmul_f64_f64_f64, 1e-12);
}

static MunitResult test_dispatched_f64_f64_f64_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_medium("dispatched", matmul_f64_f64_f64, 1e-9);
}

static MunitResult test_dispatched_f64_f64_f64_scaled(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_scaled_small("dispatched", matmul_f64_f64_f64, 1e-12);
}

static MunitResult test_dispatched_f64_f64_f64_scaled_medium(const MunitParameter *params, void *data) {
  (void)params;
  (void)data;
  return test_f64_f64_f64_scaled_medium("dispatched", matmul_f64_f64_f64, 1e-9);
}

static MunitTest tests[] = {
    {"/scalar-u8-i8-u8", test_scalar_u8_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-i8-u8-medium", test_scalar_u8_i8_u8_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-i8-u8-scaled", test_scalar_u8_i8_u8_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-i8-u8-scaled-medium", test_scalar_u8_i8_u8_scaled_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
#ifdef __AVX512VNNI__
    {"/avx512vnni-u8-i8-u8", test_avx512vnni_u8_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512vnni-u8-i8-u8-medium", test_avx512vnni_u8_i8_u8_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512vnni-u8-i8-u8-scaled", test_avx512vnni_u8_i8_u8_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512vnni-u8-i8-u8-scaled-medium", test_avx512vnni_u8_i8_u8_scaled_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE,
     NULL},
#endif
    {"/dispatched-u8-i8-u8", test_dispatched_u8_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-u8-i8-u8-medium", test_dispatched_u8_i8_u8_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-u8-i8-u8-scaled", test_dispatched_u8_i8_u8_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-u8-i8-u8-scaled-medium", test_dispatched_u8_i8_u8_scaled_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE,
     NULL},
    {"/scalar-f32-f32-f32", test_scalar_f32_f32_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f32-f32-medium", test_scalar_f32_f32_f32_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f32-f32-scaled", test_scalar_f32_f32_f32_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f32-f32-scaled-medium", test_scalar_f32_f32_f32_scaled_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE,
     NULL},
#ifdef __AVX2__
    {"/avx2-f32-f32-f32", test_avx2_f32_f32_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx2-f32-f32-f32-medium", test_avx2_f32_f32_f32_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx2-f32-f32-f32-scaled", test_avx2_f32_f32_f32_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx2-f32-f32-f32-scaled-medium", test_avx2_f32_f32_f32_scaled_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
#endif
#ifdef __AVX512F__
    {"/avx512-f32-f32-f32", test_avx512_f32_f32_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512-f32-f32-f32-medium", test_avx512_f32_f32_f32_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512-f32-f32-f32-scaled", test_avx512_f32_f32_f32_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512-f32-f32-f32-scaled-medium", test_avx512_f32_f32_f32_scaled_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE,
     NULL},
#endif
    {"/dispatched-f32-f32-f32", test_dispatched_f32_f32_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-f32-f32-f32-medium", test_dispatched_f32_f32_f32_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-f32-f32-f32-scaled", test_dispatched_f32_f32_f32_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-f32-f32-f32-scaled-medium", test_dispatched_f32_f32_f32_scaled_medium, NULL, NULL,
     MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f64-f64", test_scalar_f64_f64_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f64-f64-medium", test_scalar_f64_f64_f64_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f64-f64-scaled", test_scalar_f64_f64_f64_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f64-f64-scaled-medium", test_scalar_f64_f64_f64_scaled_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE,
     NULL},
#ifdef __AVX2__
    {"/avx2-f64-f64-f64", test_avx2_f64_f64_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx2-f64-f64-f64-medium", test_avx2_f64_f64_f64_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx2-f64-f64-f64-scaled", test_avx2_f64_f64_f64_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx2-f64-f64-f64-scaled-medium", test_avx2_f64_f64_f64_scaled_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
#endif
#ifdef __AVX512F__
    {"/avx512-f64-f64-f64", test_avx512_f64_f64_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512-f64-f64-f64-medium", test_avx512_f64_f64_f64_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512-f64-f64-f64-scaled", test_avx512_f64_f64_f64_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/avx512-f64-f64-f64-scaled-medium", test_avx512_f64_f64_f64_scaled_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE,
     NULL},
#endif
    {"/dispatched-f64-f64-f64", test_dispatched_f64_f64_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-f64-f64-f64-medium", test_dispatched_f64_f64_f64_medium, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-f64-f64-f64-scaled", test_dispatched_f64_f64_f64_scaled, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/dispatched-f64-f64-f64-scaled-medium", test_dispatched_f64_f64_f64_scaled_medium, NULL, NULL,
     MUNIT_TEST_OPTION_NONE, NULL},
    {NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL}};

static const MunitSuite suite = {"/matmul", tests, NULL, 1, MUNIT_SUITE_OPTION_NONE};

int main(int argc, char *argv[MUNIT_ARRAY_PARAM(argc)]) {
  return munit_suite_main(&suite, NULL, argc, argv);
}
