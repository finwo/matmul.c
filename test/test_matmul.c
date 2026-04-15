#include "nemequ/munit.h"
#include "../src/matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define f32 float
#define f64 double
#define i8 int8_t
#define u8 uint8_t

static void ref_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += A[i*n+k] * B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f32_f32_f64(size_t m, size_t n, size_t p, const float *A, const float *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f32_f32_i8(size_t m, size_t n, size_t p, const float *A, const float *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_f32_f32_u8(size_t m, size_t n, size_t p, const float *A, const float *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_f32_f64_f32(size_t m, size_t n, size_t p, const float *A, const double *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f32_f64_f64(size_t m, size_t n, size_t p, const float *A, const double *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f32_f64_i8(size_t m, size_t n, size_t p, const float *A, const double *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_f32_f64_u8(size_t m, size_t n, size_t p, const float *A, const double *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_f32_i8_f32(size_t m, size_t n, size_t p, const float *A, const int8_t *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f32_i8_f64(size_t m, size_t n, size_t p, const float *A, const int8_t *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f32_i8_i8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_f32_i8_u8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_f32_u8_f32(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f32_u8_f64(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f32_u8_i8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_f32_u8_u8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_f64_f32_f32(size_t m, size_t n, size_t p, const double *A, const float *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f64_f32_f64(size_t m, size_t n, size_t p, const double *A, const float *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f64_f32_i8(size_t m, size_t n, size_t p, const double *A, const float *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_f64_f32_u8(size_t m, size_t n, size_t p, const double *A, const float *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_f64_f64_f32(size_t m, size_t n, size_t p, const double *A, const double *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += A[i*n+k] * B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f64_f64_i8(size_t m, size_t n, size_t p, const double *A, const double *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_f64_f64_u8(size_t m, size_t n, size_t p, const double *A, const double *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_f64_i8_f32(size_t m, size_t n, size_t p, const double *A, const int8_t *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f64_i8_f64(size_t m, size_t n, size_t p, const double *A, const int8_t *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f64_i8_i8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_f64_i8_u8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_f64_u8_f32(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f64_u8_f64(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_f64_u8_i8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_f64_u8_u8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_i8_f32_f32(size_t m, size_t n, size_t p, const int8_t *A, const float *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * B[k*p+j]; C[i*p+j] = s; }
}
static void ref_i8_f32_f64(size_t m, size_t n, size_t p, const int8_t *A, const float *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_i8_f32_i8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_i8_f32_u8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_i8_f64_f32(size_t m, size_t n, size_t p, const int8_t *A, const double *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_i8_f64_f64(size_t m, size_t n, size_t p, const int8_t *A, const double *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * B[k*p+j]; C[i*p+j] = s; }
}
static void ref_i8_f64_i8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_i8_f64_u8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_i8_i8_f32(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_i8_i8_f64(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_i8_i8_i8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_i8_i8_u8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_i8_u8_f32(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_i8_u8_f64(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_i8_u8_i8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_i8_u8_u8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_u8_f32_f32(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * B[k*p+j]; C[i*p+j] = s; }
}
static void ref_u8_f32_f64(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_u8_f32_i8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_u8_f32_u8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_u8_f64_f32(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_u8_f64_f64(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * B[k*p+j]; C[i*p+j] = s; }
}
static void ref_u8_f64_i8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_u8_f64_u8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_u8_i8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_u8_i8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_u8_i8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}
static void ref_u8_u8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, float *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { float s = 0; for (size_t k = 0; k < n; k++) s += (float)A[i*n+k] * (float)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_u8_u8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, double *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { double s = 0; for (size_t k = 0; k < n; k++) s += (double)A[i*n+k] * (double)B[k*p+j]; C[i*p+j] = s; }
}
static void ref_u8_u8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, int8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 127) ? 127 : (s < -128 ? -128 : s); }
}
static void ref_u8_u8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, uint8_t *C) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < p; j++) { int s = 0; for (size_t k = 0; k < n; k++) s += (int)A[i*n+k] * (int)B[k*p+j]; C[i*p+j] = (s > 255) ? 255 : (s < 0 ? 0 : s); }
}

#define TEST_FLT(ta, tb, tc, backend) \
static MunitResult test_##backend##_##ta##_##tb##_##tc(const MunitParameter *params, void *data) { \
    (void)params; (void)data; \
    ta A[] = {1, 2, 3, 4, 5, 6}; tb B[] = {1, 0, 0, 1, 0, 0}; tc C[4], E[4]; \
    ref_##ta##_##tb##_##tc(2, 3, 2, A, B, E); \
    matmul_##backend##_##ta##_##tb##_##tc(2, 3, 2, A, B, C, 0.0); \
    for (int i = 0; i < 4; i++) { float d = E[i] - C[i]; if (d < 0) d = -d; if (d > 1e-5f) return MUNIT_FAIL; } \
    return MUNIT_OK; \
}
#define TEST_DBL(ta, tb, tc, backend) \
static MunitResult test_##backend##_##ta##_##tb##_##tc(const MunitParameter *params, void *data) { \
    (void)params; (void)data; \
    ta A[] = {1, 2, 3, 4, 5, 6}; tb B[] = {1, 0, 0, 1, 0, 0}; tc C[4], E[4]; \
    ref_##ta##_##tb##_##tc(2, 3, 2, A, B, E); \
    matmul_##backend##_##ta##_##tb##_##tc(2, 3, 2, A, B, C, 0.0); \
    for (int i = 0; i < 4; i++) { double d = E[i] - C[i]; if (d < 0) d = -d; if (d > 1e-10) return MUNIT_FAIL; } \
    return MUNIT_OK; \
}
#define TEST_INT(ta, tb, tc, backend) \
static MunitResult test_##backend##_##ta##_##tb##_##tc(const MunitParameter *params, void *data) { \
    (void)params; (void)data; \
    ta A[] = {1, 2, 3, 4, 5, 6}; tb B[] = {1, 0, 0, 1, 0, 0}; tc C[4], E[4]; \
    ref_##ta##_##tb##_##tc(2, 3, 2, A, B, E); \
    matmul_##backend##_##ta##_##tb##_##tc(2, 3, 2, A, B, C, 0.0); \
    for (int i = 0; i < 4; i++) if (E[i] != C[i]) return MUNIT_FAIL; \
    return MUNIT_OK; \
}

TEST_FLT(f32, f32, f32, scalar) TEST_DBL(f32, f32, f64, scalar) TEST_INT(f32, f32, i8, scalar) TEST_INT(f32, f32, u8, scalar)
TEST_FLT(f32, f64, f32, scalar) TEST_DBL(f32, f64, f64, scalar) TEST_INT(f32, f64, i8, scalar) TEST_INT(f32, f64, u8, scalar)
TEST_FLT(f32, i8, f32, scalar) TEST_DBL(f32, i8, f64, scalar) TEST_INT(f32, i8, i8, scalar) TEST_INT(f32, i8, u8, scalar)
TEST_FLT(f32, u8, f32, scalar) TEST_DBL(f32, u8, f64, scalar) TEST_INT(f32, u8, i8, scalar) TEST_INT(f32, u8, u8, scalar)
TEST_FLT(f64, f32, f32, scalar) TEST_DBL(f64, f32, f64, scalar) TEST_INT(f64, f32, i8, scalar) TEST_INT(f64, f32, u8, scalar)
TEST_FLT(f64, f64, f32, scalar) TEST_DBL(f64, f64, f64, scalar) TEST_INT(f64, f64, i8, scalar) TEST_INT(f64, f64, u8, scalar)
TEST_FLT(f64, i8, f32, scalar) TEST_DBL(f64, i8, f64, scalar) TEST_INT(f64, i8, i8, scalar) TEST_INT(f64, i8, u8, scalar)
TEST_FLT(f64, u8, f32, scalar) TEST_DBL(f64, u8, f64, scalar) TEST_INT(f64, u8, i8, scalar) TEST_INT(f64, u8, u8, scalar)
TEST_FLT(i8, f32, f32, scalar) TEST_DBL(i8, f32, f64, scalar) TEST_INT(i8, f32, i8, scalar) TEST_INT(i8, f32, u8, scalar)
TEST_FLT(i8, f64, f32, scalar) TEST_DBL(i8, f64, f64, scalar) TEST_INT(i8, f64, i8, scalar) TEST_INT(i8, f64, u8, scalar)
TEST_FLT(i8, i8, f32, scalar) TEST_DBL(i8, i8, f64, scalar) TEST_INT(i8, i8, i8, scalar) TEST_INT(i8, i8, u8, scalar)
TEST_FLT(i8, u8, f32, scalar) TEST_DBL(i8, u8, f64, scalar) TEST_INT(i8, u8, i8, scalar) TEST_INT(i8, u8, u8, scalar)
TEST_FLT(u8, f32, f32, scalar) TEST_DBL(u8, f32, f64, scalar) TEST_INT(u8, f32, i8, scalar) TEST_INT(u8, f32, u8, scalar)
TEST_FLT(u8, f64, f32, scalar) TEST_DBL(u8, f64, f64, scalar) TEST_INT(u8, f64, i8, scalar) TEST_INT(u8, f64, u8, scalar)
TEST_FLT(u8, i8, f32, scalar) TEST_DBL(u8, i8, f64, scalar) TEST_INT(u8, i8, i8, scalar) TEST_INT(u8, i8, u8, scalar)
TEST_FLT(u8, u8, f32, scalar) TEST_DBL(u8, u8, f64, scalar) TEST_INT(u8, u8, i8, scalar) TEST_INT(u8, u8, u8, scalar)

static MunitTest tests[] = {
    {"/scalar-f32-f32-f32", test_scalar_f32_f32_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f32-f64", test_scalar_f32_f32_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f32-i8", test_scalar_f32_f32_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f32-u8", test_scalar_f32_f32_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f64-f32", test_scalar_f32_f64_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f64-f64", test_scalar_f32_f64_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f64-i8", test_scalar_f32_f64_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-f64-u8", test_scalar_f32_f64_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-i8-f32", test_scalar_f32_i8_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-i8-f64", test_scalar_f32_i8_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-i8-i8", test_scalar_f32_i8_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-i8-u8", test_scalar_f32_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-u8-f32", test_scalar_f32_u8_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-u8-f64", test_scalar_f32_u8_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-u8-i8", test_scalar_f32_u8_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f32-u8-u8", test_scalar_f32_u8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f32-f32", test_scalar_f64_f32_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f32-f64", test_scalar_f64_f32_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f32-i8", test_scalar_f64_f32_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f32-u8", test_scalar_f64_f32_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f64-f32", test_scalar_f64_f64_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f64-f64", test_scalar_f64_f64_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f64-i8", test_scalar_f64_f64_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-f64-u8", test_scalar_f64_f64_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-i8-f32", test_scalar_f64_i8_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-i8-f64", test_scalar_f64_i8_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-i8-i8", test_scalar_f64_i8_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-i8-u8", test_scalar_f64_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-u8-f32", test_scalar_f64_u8_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-u8-f64", test_scalar_f64_u8_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-u8-i8", test_scalar_f64_u8_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-f64-u8-u8", test_scalar_f64_u8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-f32-f32", test_scalar_i8_f32_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-f32-f64", test_scalar_i8_f32_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-f32-i8", test_scalar_i8_f32_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-f32-u8", test_scalar_i8_f32_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-f64-f32", test_scalar_i8_f64_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-f64-f64", test_scalar_i8_f64_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-f64-i8", test_scalar_i8_f64_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-f64-u8", test_scalar_i8_f64_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-i8-f32", test_scalar_i8_i8_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-i8-f64", test_scalar_i8_i8_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-i8-i8", test_scalar_i8_i8_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-i8-u8", test_scalar_i8_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-u8-f32", test_scalar_i8_u8_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-u8-f64", test_scalar_i8_u8_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-u8-i8", test_scalar_i8_u8_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-i8-u8-u8", test_scalar_i8_u8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-f32-f32", test_scalar_u8_f32_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-f32-f64", test_scalar_u8_f32_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-f32-i8", test_scalar_u8_f32_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-f32-u8", test_scalar_u8_f32_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-f64-f32", test_scalar_u8_f64_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-f64-f64", test_scalar_u8_f64_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-f64-i8", test_scalar_u8_f64_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-f64-u8", test_scalar_u8_f64_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-i8-f32", test_scalar_u8_i8_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-i8-f64", test_scalar_u8_i8_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-i8-i8", test_scalar_u8_i8_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-i8-u8", test_scalar_u8_i8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-u8-f32", test_scalar_u8_u8_f32, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-u8-f64", test_scalar_u8_u8_f64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-u8-i8", test_scalar_u8_u8_i8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/scalar-u8-u8-u8", test_scalar_u8_u8_u8, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL}
};

static const MunitSuite suite = { "/matmul", tests, NULL, 1, MUNIT_SUITE_OPTION_NONE };

int main(int argc, char *argv[MUNIT_ARRAY_PARAM(argc)]) {
    return munit_suite_main(&suite, NULL, argc, argv);
}
