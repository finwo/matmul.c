#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/matmul.h"

#define RUNS 100

static struct timespec timespec_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts;
}

static double timespec_diff_ms(struct timespec start, struct timespec end) {
  return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
}

static int compare_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static double percentile(double *sorted, double p, int n) {
  double idx  = p / 100.0 * (n - 1);
  int    lo   = (int)idx;
  int    hi   = lo + 1;
  double frac = idx - lo;
  if (hi >= n) hi = n - 1;
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static void bench_u8_i8_u8(size_t m, size_t n, size_t p, int runs) {
  uint8_t *A     = malloc(m * n);
  int8_t  *B     = malloc(n * p);
  uint8_t *C     = malloc(m * p);
  uint8_t *Cwarm = malloc(m * p);
  double   times[RUNS];

  if (!A || !B || !C || !Cwarm) {
    fprintf(stderr, "OOM for %zu x %zu\n", m, n);
    free(A);
    free(B);
    free(C);
    free(Cwarm);
    return;
  }

  for (size_t i = 0; i < m * n; i++) A[i] = (uint8_t)(rand() % 256);
  for (size_t i = 0; i < n * p; i++) B[i] = (int8_t)(rand() % 256);
  memset(C, 0, m * p);
  memset(Cwarm, 0, m * p);

  matmul_u8_i8_u8(m, n, p, A, B, Cwarm, 0.0);

  int actual_runs = runs;
  if (m >= 4096) actual_runs = 3;

  for (int r = 0; r < actual_runs; r++) {
    memset(C, 0, m * p);
    struct timespec start = timespec_now();
    matmul_u8_i8_u8(m, n, p, A, B, C, 0.0);
    struct timespec end = timespec_now();
    times[r]            = timespec_diff_ms(start, end);
  }

  qsort(times, actual_runs, sizeof(double), compare_double);

  double gflops = 2.0 * m * n * p / (percentile(times, 50, actual_runs) * 1e6);

  printf("%8zu x %8zu | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f | %8.1f\n", m, n, percentile(times, 1, actual_runs),
         percentile(times, 5, actual_runs), percentile(times, 50, actual_runs), percentile(times, 95, actual_runs),
         percentile(times, 99, actual_runs), gflops);

  free(A);
  free(B);
  free(C);
  free(Cwarm);
}

static void bench_f32_f32_f32(size_t m, size_t n, size_t p, int runs) {
  float *A     = malloc(m * n * sizeof(float));
  float *B     = malloc(n * p * sizeof(float));
  float *C     = malloc(m * p * sizeof(float));
  float *Cwarm = malloc(m * p * sizeof(float));
  double times[RUNS];

  if (!A || !B || !C || !Cwarm) {
    fprintf(stderr, "OOM for %zu x %zu\n", m, n);
    free(A);
    free(B);
    free(C);
    free(Cwarm);
    return;
  }

  for (size_t i = 0; i < m * n; i++) A[i] = (float)(rand() % 256);
  for (size_t i = 0; i < n * p; i++) B[i] = (float)(rand() % 256);
  memset(C, 0, m * p * sizeof(float));
  memset(Cwarm, 0, m * p * sizeof(float));

  matmul_f32_f32_f32(m, n, p, A, B, Cwarm, 0.0);

  int actual_runs = runs;
  if (m >= 4096) actual_runs = 3;

  for (int r = 0; r < actual_runs; r++) {
    memset(C, 0, m * p * sizeof(float));
    struct timespec start = timespec_now();
    matmul_f32_f32_f32(m, n, p, A, B, C, 0.0);
    struct timespec end = timespec_now();
    times[r]            = timespec_diff_ms(start, end);
  }

  qsort(times, actual_runs, sizeof(double), compare_double);

  double gflops = 2.0 * m * n * p / (percentile(times, 50, actual_runs) * 1e6);

  printf("%8zu x %8zu | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f | %8.1f\n", m, n, percentile(times, 1, actual_runs),
         percentile(times, 5, actual_runs), percentile(times, 50, actual_runs), percentile(times, 95, actual_runs),
         percentile(times, 99, actual_runs), gflops);

  free(A);
  free(B);
  free(C);
  free(Cwarm);
}

static void bench_f64_f64_f64(size_t m, size_t n, size_t p, int runs) {
  double *A     = malloc(m * n * sizeof(double));
  double *B     = malloc(n * p * sizeof(double));
  double *C     = malloc(m * p * sizeof(double));
  double *Cwarm = malloc(m * p * sizeof(double));
  double  times[RUNS];

  if (!A || !B || !C || !Cwarm) {
    fprintf(stderr, "OOM for %zu x %zu\n", m, n);
    free(A);
    free(B);
    free(C);
    free(Cwarm);
    return;
  }

  for (size_t i = 0; i < m * n; i++) A[i] = (double)(rand() % 256);
  for (size_t i = 0; i < n * p; i++) B[i] = (double)(rand() % 256);
  memset(C, 0, m * p * sizeof(double));
  memset(Cwarm, 0, m * p * sizeof(double));

  matmul_f64_f64_f64(m, n, p, A, B, Cwarm, 0.0);

  int actual_runs = runs;
  if (m >= 4096) actual_runs = 3;

  for (int r = 0; r < actual_runs; r++) {
    memset(C, 0, m * p * sizeof(double));
    struct timespec start = timespec_now();
    matmul_f64_f64_f64(m, n, p, A, B, C, 0.0);
    struct timespec end = timespec_now();
    times[r]            = timespec_diff_ms(start, end);
  }

  qsort(times, actual_runs, sizeof(double), compare_double);

  double gflops = 2.0 * m * n * p / (percentile(times, 50, actual_runs) * 1e6);

  printf("%8zu x %8zu | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f | %8.1f\n", m, n, percentile(times, 1, actual_runs),
         percentile(times, 5, actual_runs), percentile(times, 50, actual_runs), percentile(times, 95, actual_runs),
         percentile(times, 99, actual_runs), gflops);

  free(A);
  free(B);
  free(C);
  free(Cwarm);
}

int main(void) {
  srand(42);

  const size_t sizes[][3] = {
      {16, 16, 16}, {64, 64, 64}, {256, 256, 256}, {1024, 1024, 1024}, {4096, 4096, 4096},
  };
  const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

  printf("Benchmark: u8_i8_u8 matmul, %d runs per size\n", RUNS);
  printf("--------------------------------------------------------------------------------\n");
  printf("%8s | %8s | %8s | %8s | %8s | %8s | %8s\n", "M x N", "1% (ms)", "5% (ms)", "50% (ms)", "95% (ms)", "99% (ms)",
         "GFLOPS");
  printf("--------------------------------------------------------------------------------\n");
  for (int i = 0; i < num_sizes; i++) {
    bench_u8_i8_u8(sizes[i][0], sizes[i][1], sizes[i][2], RUNS);
  }
  printf("--------------------------------------------------------------------------------\n");
  printf("\n");

  printf("Benchmark: f32_f32_f32 matmul, %d runs per size\n", RUNS);
  printf("--------------------------------------------------------------------------------\n");
  printf("%8s | %8s | %8s | %8s | %8s | %8s | %8s\n", "M x N", "1% (ms)", "5% (ms)", "50% (ms)", "95% (ms)", "99% (ms)",
         "GFLOPS");
  printf("--------------------------------------------------------------------------------\n");
  for (int i = 0; i < num_sizes; i++) {
    bench_f32_f32_f32(sizes[i][0], sizes[i][1], sizes[i][2], RUNS);
  }
  printf("--------------------------------------------------------------------------------\n");
  printf("\n");

  printf("Benchmark: f64_f64_f64 matmul, %d runs per size\n", RUNS);
  printf("--------------------------------------------------------------------------------\n");
  printf("%8s | %8s | %8s | %8s | %8s | %8s | %8s\n", "M x N", "1% (ms)", "5% (ms)", "50% (ms)", "95% (ms)", "99% (ms)",
         "GFLOPS");
  printf("--------------------------------------------------------------------------------\n");
  for (int i = 0; i < num_sizes; i++) {
    bench_f64_f64_f64(sizes[i][0], sizes[i][1], sizes[i][2], RUNS);
  }
  printf("--------------------------------------------------------------------------------\n");

  return 0;
}
