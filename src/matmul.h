#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  MATMUL_SCALAR      = 0,
  MATMUL_AVX2        = 1,
  MATMUL_AVX512      = 2,
  MATMUL_AVX512_VNNI = 3,
  MATMUL_AVXVNNI     = 4
} matmul_feature_t;

matmul_feature_t matmul_get_feature(void);
const char      *matmul_get_feature_name(matmul_feature_t feat);

int matmul_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale);
int matmul_f32_f32_f64(size_t m, size_t n, size_t p, const float *A, const float *B, double *C, double scale);
int matmul_f32_f32_i8(size_t m, size_t n, size_t p, const float *A, const float *B, int8_t *C, double scale);
int matmul_f32_f32_u8(size_t m, size_t n, size_t p, const float *A, const float *B, uint8_t *C, double scale);
int matmul_f32_f64_f32(size_t m, size_t n, size_t p, const float *A, const double *B, float *C, double scale);
int matmul_f32_f64_f64(size_t m, size_t n, size_t p, const float *A, const double *B, double *C, double scale);
int matmul_f32_f64_i8(size_t m, size_t n, size_t p, const float *A, const double *B, int8_t *C, double scale);
int matmul_f32_f64_u8(size_t m, size_t n, size_t p, const float *A, const double *B, uint8_t *C, double scale);
int matmul_f32_i8_f32(size_t m, size_t n, size_t p, const float *A, const int8_t *B, float *C, double scale);
int matmul_f32_i8_f64(size_t m, size_t n, size_t p, const float *A, const int8_t *B, double *C, double scale);
int matmul_f32_i8_i8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, int8_t *C, double scale);
int matmul_f32_i8_u8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, uint8_t *C, double scale);
int matmul_f32_u8_f32(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, float *C, double scale);
int matmul_f32_u8_f64(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, double *C, double scale);
int matmul_f32_u8_i8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, int8_t *C, double scale);
int matmul_f32_u8_u8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, uint8_t *C, double scale);
int matmul_f64_f32_f32(size_t m, size_t n, size_t p, const double *A, const float *B, float *C, double scale);
int matmul_f64_f32_f64(size_t m, size_t n, size_t p, const double *A, const float *B, double *C, double scale);
int matmul_f64_f32_i8(size_t m, size_t n, size_t p, const double *A, const float *B, int8_t *C, double scale);
int matmul_f64_f32_u8(size_t m, size_t n, size_t p, const double *A, const float *B, uint8_t *C, double scale);
int matmul_f64_f64_f32(size_t m, size_t n, size_t p, const double *A, const double *B, float *C, double scale);
int matmul_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale);
int matmul_f64_f64_i8(size_t m, size_t n, size_t p, const double *A, const double *B, int8_t *C, double scale);
int matmul_f64_f64_u8(size_t m, size_t n, size_t p, const double *A, const double *B, uint8_t *C, double scale);
int matmul_f64_i8_f32(size_t m, size_t n, size_t p, const double *A, const int8_t *B, float *C, double scale);
int matmul_f64_i8_f64(size_t m, size_t n, size_t p, const double *A, const int8_t *B, double *C, double scale);
int matmul_f64_i8_i8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, int8_t *C, double scale);
int matmul_f64_i8_u8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, uint8_t *C, double scale);
int matmul_f64_u8_f32(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, float *C, double scale);
int matmul_f64_u8_f64(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, double *C, double scale);
int matmul_f64_u8_i8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, int8_t *C, double scale);
int matmul_f64_u8_u8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, uint8_t *C, double scale);
int matmul_i8_f32_f32(size_t m, size_t n, size_t p, const int8_t *A, const float *B, float *C, double scale);
int matmul_i8_f32_f64(size_t m, size_t n, size_t p, const int8_t *A, const float *B, double *C, double scale);
int matmul_i8_f32_i8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, int8_t *C, double scale);
int matmul_i8_f32_u8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, uint8_t *C, double scale);
int matmul_i8_f64_f32(size_t m, size_t n, size_t p, const int8_t *A, const double *B, float *C, double scale);
int matmul_i8_f64_f64(size_t m, size_t n, size_t p, const int8_t *A, const double *B, double *C, double scale);
int matmul_i8_f64_i8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, int8_t *C, double scale);
int matmul_i8_f64_u8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, uint8_t *C, double scale);
int matmul_i8_i8_f32(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, float *C, double scale);
int matmul_i8_i8_f64(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, double *C, double scale);
int matmul_i8_i8_i8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, int8_t *C, double scale);
int matmul_i8_i8_u8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, uint8_t *C, double scale);
int matmul_i8_u8_f32(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, float *C, double scale);
int matmul_i8_u8_f64(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, double *C, double scale);
int matmul_i8_u8_i8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, int8_t *C, double scale);
int matmul_i8_u8_u8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, uint8_t *C, double scale);
int matmul_u8_f32_f32(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, float *C, double scale);
int matmul_u8_f32_f64(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, double *C, double scale);
int matmul_u8_f32_i8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, int8_t *C, double scale);
int matmul_u8_f32_u8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, uint8_t *C, double scale);
int matmul_u8_f64_f32(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, float *C, double scale);
int matmul_u8_f64_f64(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, double *C, double scale);
int matmul_u8_f64_i8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, int8_t *C, double scale);
int matmul_u8_f64_u8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, uint8_t *C, double scale);
int matmul_u8_i8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, float *C, double scale);
int matmul_u8_i8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, double *C, double scale);
int matmul_u8_i8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, int8_t *C, double scale);
int matmul_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale);
int matmul_u8_u8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, float *C, double scale);
int matmul_u8_u8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, double *C, double scale);
int matmul_u8_u8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, int8_t *C, double scale);
int matmul_u8_u8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, uint8_t *C, double scale);

int matmul_scalar_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale);
int matmul_scalar_f32_f32_f64(size_t m, size_t n, size_t p, const float *A, const float *B, double *C, double scale);
int matmul_scalar_f32_f32_i8(size_t m, size_t n, size_t p, const float *A, const float *B, int8_t *C, double scale);
int matmul_scalar_f32_f32_u8(size_t m, size_t n, size_t p, const float *A, const float *B, uint8_t *C, double scale);
int matmul_scalar_f32_f64_f32(size_t m, size_t n, size_t p, const float *A, const double *B, float *C, double scale);
int matmul_scalar_f32_f64_f64(size_t m, size_t n, size_t p, const float *A, const double *B, double *C, double scale);
int matmul_scalar_f32_f64_i8(size_t m, size_t n, size_t p, const float *A, const double *B, int8_t *C, double scale);
int matmul_scalar_f32_f64_u8(size_t m, size_t n, size_t p, const float *A, const double *B, uint8_t *C, double scale);
int matmul_scalar_f32_i8_f32(size_t m, size_t n, size_t p, const float *A, const int8_t *B, float *C, double scale);
int matmul_scalar_f32_i8_f64(size_t m, size_t n, size_t p, const float *A, const int8_t *B, double *C, double scale);
int matmul_scalar_f32_i8_i8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, int8_t *C, double scale);
int matmul_scalar_f32_i8_u8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, uint8_t *C, double scale);
int matmul_scalar_f32_u8_f32(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, float *C, double scale);
int matmul_scalar_f32_u8_f64(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, double *C, double scale);
int matmul_scalar_f32_u8_i8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, int8_t *C, double scale);
int matmul_scalar_f32_u8_u8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, uint8_t *C, double scale);
int matmul_scalar_f64_f32_f32(size_t m, size_t n, size_t p, const double *A, const float *B, float *C, double scale);
int matmul_scalar_f64_f32_f64(size_t m, size_t n, size_t p, const double *A, const float *B, double *C, double scale);
int matmul_scalar_f64_f32_i8(size_t m, size_t n, size_t p, const double *A, const float *B, int8_t *C, double scale);
int matmul_scalar_f64_f32_u8(size_t m, size_t n, size_t p, const double *A, const float *B, uint8_t *C, double scale);
int matmul_scalar_f64_f64_f32(size_t m, size_t n, size_t p, const double *A, const double *B, float *C, double scale);
int matmul_scalar_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale);
int matmul_scalar_f64_f64_i8(size_t m, size_t n, size_t p, const double *A, const double *B, int8_t *C, double scale);
int matmul_scalar_f64_f64_u8(size_t m, size_t n, size_t p, const double *A, const double *B, uint8_t *C, double scale);
int matmul_scalar_f64_i8_f32(size_t m, size_t n, size_t p, const double *A, const int8_t *B, float *C, double scale);
int matmul_scalar_f64_i8_f64(size_t m, size_t n, size_t p, const double *A, const int8_t *B, double *C, double scale);
int matmul_scalar_f64_i8_i8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, int8_t *C, double scale);
int matmul_scalar_f64_i8_u8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, uint8_t *C, double scale);
int matmul_scalar_f64_u8_f32(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, float *C, double scale);
int matmul_scalar_f64_u8_f64(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, double *C, double scale);
int matmul_scalar_f64_u8_i8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, int8_t *C, double scale);
int matmul_scalar_f64_u8_u8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, uint8_t *C, double scale);
int matmul_scalar_i8_f32_f32(size_t m, size_t n, size_t p, const int8_t *A, const float *B, float *C, double scale);
int matmul_scalar_i8_f32_f64(size_t m, size_t n, size_t p, const int8_t *A, const float *B, double *C, double scale);
int matmul_scalar_i8_f32_i8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, int8_t *C, double scale);
int matmul_scalar_i8_f32_u8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, uint8_t *C, double scale);
int matmul_scalar_i8_f64_f32(size_t m, size_t n, size_t p, const int8_t *A, const double *B, float *C, double scale);
int matmul_scalar_i8_f64_f64(size_t m, size_t n, size_t p, const int8_t *A, const double *B, double *C, double scale);
int matmul_scalar_i8_f64_i8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, int8_t *C, double scale);
int matmul_scalar_i8_f64_u8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, uint8_t *C, double scale);
int matmul_scalar_i8_i8_f32(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, float *C, double scale);
int matmul_scalar_i8_i8_f64(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, double *C, double scale);
int matmul_scalar_i8_i8_i8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, int8_t *C, double scale);
int matmul_scalar_i8_i8_u8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, uint8_t *C, double scale);
int matmul_scalar_i8_u8_f32(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, float *C, double scale);
int matmul_scalar_i8_u8_f64(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, double *C, double scale);
int matmul_scalar_i8_u8_i8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, int8_t *C, double scale);
int matmul_scalar_i8_u8_u8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, uint8_t *C, double scale);
int matmul_scalar_u8_f32_f32(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, float *C, double scale);
int matmul_scalar_u8_f32_f64(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, double *C, double scale);
int matmul_scalar_u8_f32_i8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, int8_t *C, double scale);
int matmul_scalar_u8_f32_u8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, uint8_t *C, double scale);
int matmul_scalar_u8_f64_f32(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, float *C, double scale);
int matmul_scalar_u8_f64_f64(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, double *C, double scale);
int matmul_scalar_u8_f64_i8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, int8_t *C, double scale);
int matmul_scalar_u8_f64_u8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, uint8_t *C, double scale);
int matmul_scalar_u8_i8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, float *C, double scale);
int matmul_scalar_u8_i8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, double *C, double scale);
int matmul_scalar_u8_i8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, int8_t *C, double scale);
int matmul_scalar_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale);
int matmul_scalar_u8_u8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, float *C, double scale);
int matmul_scalar_u8_u8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, double *C, double scale);
int matmul_scalar_u8_u8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, int8_t *C, double scale);
int matmul_scalar_u8_u8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, uint8_t *C, double scale);

#define matmul(m, n, p, A, B, C, scale)    \
  _Generic((A),                            \
      float: _Generic((B),                 \
          float: _Generic((C),             \
              float: matmul_f32_f32_f32,   \
              double: matmul_f32_f32_f64,  \
              int8_t: matmul_f32_f32_i8,   \
              uint8_t: matmul_f32_f32_u8), \
          double: _Generic((C),            \
              float: matmul_f32_f64_f32,   \
              double: matmul_f32_f64_f64,  \
              int8_t: matmul_f32_f64_i8,   \
              uint8_t: matmul_f32_f64_u8), \
          int8_t: _Generic((C),            \
              float: matmul_f32_i8_f32,    \
              double: matmul_f32_i8_f64,   \
              int8_t: matmul_f32_i8_i8,    \
              uint8_t: matmul_f32_i8_u8),  \
          uint8_t: _Generic((C),           \
              float: matmul_f32_u8_f32,    \
              double: matmul_f32_u8_f64,   \
              int8_t: matmul_f32_u8_i8,    \
              uint8_t: matmul_f32_u8_u8)), \
      double: _Generic((B),                \
          float: _Generic((C),             \
              float: matmul_f64_f32_f32,   \
              double: matmul_f64_f32_f64,  \
              int8_t: matmul_f64_f32_i8,   \
              uint8_t: matmul_f64_f32_u8), \
          double: _Generic((C),            \
              float: matmul_f64_f64_f32,   \
              double: matmul_f64_f64_f64,  \
              int8_t: matmul_f64_f64_i8,   \
              uint8_t: matmul_f64_f64_u8), \
          int8_t: _Generic((C),            \
              float: matmul_f64_i8_f32,    \
              double: matmul_f64_i8_f64,   \
              int8_t: matmul_f64_i8_i8,    \
              uint8_t: matmul_f64_i8_u8),  \
          uint8_t: _Generic((C),           \
              float: matmul_f64_u8_f32,    \
              double: matmul_f64_u8_f64,   \
              int8_t: matmul_f64_u8_i8,    \
              uint8_t: matmul_f64_u8_u8)), \
      int8_t: _Generic((B),                \
          float: _Generic((C),             \
              float: matmul_i8_f32_f32,    \
              double: matmul_i8_f32_f64,   \
              int8_t: matmul_i8_f32_i8,    \
              uint8_t: matmul_i8_f32_u8),  \
          double: _Generic((C),            \
              float: matmul_i8_f64_f32,    \
              double: matmul_i8_f64_f64,   \
              int8_t: matmul_i8_f64_i8,    \
              uint8_t: matmul_i8_f64_u8),  \
          int8_t: _Generic((C),            \
              float: matmul_i8_i8_f32,     \
              double: matmul_i8_i8_f64,    \
              int8_t: matmul_i8_i8_i8,     \
              uint8_t: matmul_i8_i8_u8),   \
          uint8_t: _Generic((C),           \
              float: matmul_i8_u8_f32,     \
              double: matmul_i8_u8_f64,    \
              int8_t: matmul_i8_u8_i8,     \
              uint8_t: matmul_i8_u8_u8)),  \
      uint8_t: _Generic((B),               \
          float: _Generic((C),             \
              float: matmul_u8_f32_f32,    \
              double: matmul_u8_f32_f64,   \
              int8_t: matmul_u8_f32_i8,    \
              uint8_t: matmul_u8_f32_u8),  \
          double: _Generic((C),            \
              float: matmul_u8_f64_f32,    \
              double: matmul_u8_f64_f64,   \
              int8_t: matmul_u8_f64_i8,    \
              uint8_t: matmul_u8_f64_u8),  \
          int8_t: _Generic((C),            \
              float: matmul_u8_i8_f32,     \
              double: matmul_u8_i8_f64,    \
              int8_t: matmul_u8_i8_i8,     \
              uint8_t: matmul_u8_i8_u8),   \
          uint8_t: _Generic((C),           \
              float: matmul_u8_u8_f32,     \
              double: matmul_u8_u8_f64,    \
              int8_t: matmul_u8_u8_i8,     \
              uint8_t: matmul_u8_u8_u8)))((m), (n), (p), (A), (B), (C), (scale))

#ifdef __cplusplus
}
#endif

#endif
