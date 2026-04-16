/*
 * Copyright (c) 2026 finwo
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to use, copy,
 * modify, and distribute the Software, subject to the following conditions:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions, and the following disclaimer.
 *
 *  2. Redistributions in binary form, or any public offering of the Software
 *     (including hosted or managed services), must reproduce the above copyright
 *     notice, this list of conditions, and the following disclaimer in the
 *     documentation and/or other materials provided.
 *
 *  3. Any redistribution or public offering of the Software must clearly attribute
 *     the Software to the original copyright holder, reference this License, and
 *     include a link to the official project repository or website.
 *
 *  4. The Software may not be renamed, rebranded, or marketed in a manner that
 *     implies it is an independent or proprietary product. Derivative works must
 *     clearly state that they are based on the Software.
 *
 *  5. Modifications to copies of the Software must carry prominent notices stating
 *     that changes were made, the nature of the modifications, and the date of the
 *     modifications.
 *
 * Any violation of these conditions terminates the permissions granted herein.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#define _GNU_SOURCE
#include "matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static matmul_feature_t g_feature = 0;

static void init_feature(void) {
  if (g_feature != 0) return;
  g_feature = MATMUL_FLAG_SCALAR;
#ifdef __AVX2__
  if (__builtin_cpu_supports("avx2")) g_feature |= MATMUL_FLAG_AVX2;
#endif
#ifdef __AVX512F__
  if (__builtin_cpu_supports("avx512f")) g_feature |= MATMUL_FLAG_AVX512;
#endif
}

matmul_feature_t matmul_get_feature(void) {
  init_feature();
  return g_feature;
}

const char *matmul_get_feature_name(matmul_feature_t feat) {
  if (feat & MATMUL_FLAG_AVX512) return "avx512";
  if (feat & MATMUL_FLAG_AVX2) return "avx2";
  if (feat & MATMUL_FLAG_AVX512_VNNI) return "avx512_vnni";
  if (feat & MATMUL_FLAG_AVXVNNI) return "avx_vnni";
  if (feat & MATMUL_FLAG_SCALAR) return "scalar";
  return "unknown";
}

int matmul_scalar_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_f32_f64(size_t m, size_t n, size_t p, const float *A, const float *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_f32_i8(size_t m, size_t n, size_t p, const float *A, const float *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_f32_u8(size_t m, size_t n, size_t p, const float *A, const float *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_f64_f32(size_t m, size_t n, size_t p, const float *A, const double *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_f64_f64(size_t m, size_t n, size_t p, const float *A, const double *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_f64_i8(size_t m, size_t n, size_t p, const float *A, const double *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_f64_u8(size_t m, size_t n, size_t p, const float *A, const double *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_i8_f32(size_t m, size_t n, size_t p, const float *A, const int8_t *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_i8_f64(size_t m, size_t n, size_t p, const float *A, const int8_t *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_i8_i8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_i8_u8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_u8_f32(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_u8_f64(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_u8_i8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f32_u8_u8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_f32_f32(size_t m, size_t n, size_t p, const double *A, const float *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_f32_f64(size_t m, size_t n, size_t p, const double *A, const float *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_f32_i8(size_t m, size_t n, size_t p, const double *A, const float *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_f32_u8(size_t m, size_t n, size_t p, const double *A, const float *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_f64_f32(size_t m, size_t n, size_t p, const double *A, const double *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_f64_i8(size_t m, size_t n, size_t p, const double *A, const double *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_f64_u8(size_t m, size_t n, size_t p, const double *A, const double *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_i8_f32(size_t m, size_t n, size_t p, const double *A, const int8_t *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_i8_f64(size_t m, size_t n, size_t p, const double *A, const int8_t *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_i8_i8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_i8_u8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_u8_f32(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_u8_f64(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_u8_i8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_f64_u8_u8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_f32_f32(size_t m, size_t n, size_t p, const int8_t *A, const float *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_f32_f64(size_t m, size_t n, size_t p, const int8_t *A, const float *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_f32_i8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_f32_u8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_f64_f32(size_t m, size_t n, size_t p, const int8_t *A, const double *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_f64_f64(size_t m, size_t n, size_t p, const int8_t *A, const double *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_f64_i8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_f64_u8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_i8_f32(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_i8_f64(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_i8_i8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_i8_u8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_u8_f32(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_u8_f64(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_u8_i8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_i8_u8_u8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_f32_f32(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_f32_f64(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_f32_i8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_f32_u8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_f64_f32(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_f64_f64(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_f64_i8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_f64_u8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_i8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_i8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_i8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_u8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (float)A[i * n + k] * (float)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= (float)scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_u8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum /= scale;
      C[i * p + j] = sum;
    }
  }
  return 0;
}

int matmul_scalar_u8_u8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, int8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;
      C[i * p + j] = (int8_t)sum;
    }
  }
  return 0;
}

#ifdef __AVX2__
int matmul_avx2_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      __m256 sum_vec = _mm256_setzero_ps();
      size_t k       = 0;
      for (; k + 7 < n; k += 8) {
        __m256 a = _mm256_loadu_ps(&A[i * n + k]);
        __m256 b = _mm256_loadu_ps(&B[k * p + j]);
        sum_vec  = _mm256_fmadd_ps(a, b, sum_vec);
      }
      float sum[8];
      _mm256_storeu_ps(sum, sum_vec);
      float s = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
      for (; k < n; k++) s += A[i * n + k] * B[k * p + j];
      if (scale != 0 && scale != 1) s /= (float)scale;
      C[i * p + j] = s;
    }
  }
  return 0;
}

int matmul_avx2_f32_f32_f64(size_t m, size_t n, size_t p, const float *A, const float *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      __m256d sum_vec = _mm256_setzero_pd();
      size_t  k       = 0;
      for (; k + 3 < n; k += 4) {
        __m256 a   = _mm256_castpd_ps(_mm256_loadu_pd((const double *)&A[i * n + k]));
        __m256 b   = _mm256_castpd_ps(_mm256_loadu_pd((const double *)&B[k * p + j]));
        __m256 mul = _mm256_mul_ps(a, b);
        sum_vec    = _mm256_add_pd(_mm256_castps_pd(mul), sum_vec);
      }
      double sum[4];
      _mm256_storeu_pd(sum, sum_vec);
      double s = sum[0] + sum[1] + sum[2] + sum[3];
      for (; k < n; k++) s += (double)A[i * n + k] * (double)B[k * p + j];
      if (scale != 0 && scale != 1) s /= scale;
      C[i * p + j] = s;
    }
  }
  return 0;
}

int matmul_avx2_f32_f64_f32(size_t m, size_t n, size_t p, const float *A, const double *B, float *C, double scale) {
  return matmul_scalar_f32_f64_f32(m, n, p, A, B, C, scale);
}

int matmul_avx2_f32_f64_f64(size_t m, size_t n, size_t p, const float *A, const double *B, double *C, double scale) {
  return matmul_scalar_f32_f64_f64(m, n, p, A, B, C, scale);
}

int matmul_avx2_f64_f32_f32(size_t m, size_t n, size_t p, const double *A, const float *B, float *C, double scale) {
  return matmul_scalar_f64_f32_f32(m, n, p, A, B, C, scale);
}

int matmul_avx2_f64_f32_f64(size_t m, size_t n, size_t p, const double *A, const float *B, double *C, double scale) {
  return matmul_scalar_f64_f32_f64(m, n, p, A, B, C, scale);
}

int matmul_avx2_f64_f64_f32(size_t m, size_t n, size_t p, const double *A, const double *B, float *C, double scale) {
  return matmul_scalar_f64_f64_f32(m, n, p, A, B, C, scale);
}

int matmul_avx2_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      __m256d sum_vec = _mm256_setzero_pd();
      size_t  k       = 0;
      for (; k + 3 < n; k += 4) {
        __m256d a = _mm256_loadu_pd(&A[i * n + k]);
        __m256d b = _mm256_loadu_pd(&B[k * p + j]);
        sum_vec   = _mm256_fmadd_pd(a, b, sum_vec);
      }
      double sum[4];
      _mm256_storeu_pd(sum, sum_vec);
      double s = sum[0] + sum[1] + sum[2] + sum[3];
      for (; k < n; k++) s += A[i * n + k] * B[k * p + j];
      if (scale != 0 && scale != 1) s /= scale;
      C[i * p + j] = s;
    }
  }
  return 0;
}
#endif

#ifdef __AVX512F__
int matmul_avx512_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      __m512 sum_vec = _mm512_setzero_ps();
      size_t k       = 0;
      for (; k + 15 < n; k += 16) {
        __m512 a = _mm512_loadu_ps(&A[i * n + k]);
        __m512 b = _mm512_loadu_ps(&B[k * p + j]);
        sum_vec  = _mm512_fmadd_ps(a, b, sum_vec);
      }
      float s = _mm512_reduce_add_ps(sum_vec);
      for (; k < n; k++) s += A[i * n + k] * B[k * p + j];
      if (scale != 0 && scale != 1) s /= (float)scale;
      C[i * p + j] = s;
    }
  }
  return 0;
}

int matmul_avx512_f32_f32_f64(size_t m, size_t n, size_t p, const float *A, const float *B, double *C, double scale) {
  return matmul_scalar_f32_f32_f64(m, n, p, A, B, C, scale);
}

int matmul_avx512_f32_f64_f32(size_t m, size_t n, size_t p, const float *A, const double *B, float *C, double scale) {
  return matmul_scalar_f32_f64_f32(m, n, p, A, B, C, scale);
}

int matmul_avx512_f32_f64_f64(size_t m, size_t n, size_t p, const float *A, const double *B, double *C, double scale) {
  return matmul_scalar_f32_f64_f64(m, n, p, A, B, C, scale);
}

int matmul_avx512_f64_f32_f32(size_t m, size_t n, size_t p, const double *A, const float *B, float *C, double scale) {
  return matmul_scalar_f64_f32_f32(m, n, p, A, B, C, scale);
}

int matmul_avx512_f64_f32_f64(size_t m, size_t n, size_t p, const double *A, const float *B, double *C, double scale) {
  return matmul_scalar_f64_f32_f64(m, n, p, A, B, C, scale);
}

int matmul_avx512_f64_f64_f32(size_t m, size_t n, size_t p, const double *A, const double *B, float *C, double scale) {
  return matmul_scalar_f64_f64_f32(m, n, p, A, B, C, scale);
}

int matmul_avx512_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      __m512d sum_vec = _mm512_setzero_pd();
      size_t  k       = 0;
      for (; k + 7 < n; k += 8) {
        __m512d a = _mm512_loadu_pd(&A[i * n + k]);
        __m512d b = _mm512_loadu_pd(&B[k * p + j]);
        sum_vec   = _mm512_fmadd_pd(a, b, sum_vec);
      }
      double s = _mm512_reduce_add_pd(sum_vec);
      for (; k < n; k++) s += A[i * n + k] * B[k * p + j];
      if (scale != 0 && scale != 1) s /= scale;
      C[i * p + j] = s;
    }
  }
  return 0;
}
#endif

int matmul_scalar_u8_u8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, uint8_t *C, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      int sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale != 0 && scale != 1) sum = (int)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }
  return 0;
}

static int _matmul_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const float *, float *, double) = NULL;
  static int initialized                                                                    = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      impl = matmul_avx512_f32_f32_f32;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      impl = matmul_avx2_f32_f32_f32;
    else
#endif
      impl = matmul_scalar_f32_f32_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_f32_f64(size_t m, size_t n, size_t p, const float *A, const float *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const float *, double *, double) = NULL;
  static int initialized                                                                     = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      impl = matmul_avx512_f32_f32_f64;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      impl = matmul_avx2_f32_f32_f64;
    else
#endif
      impl = matmul_scalar_f32_f32_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_f32_i8(size_t m, size_t n, size_t p, const float *A, const float *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const float *, int8_t *, double) = NULL;
  static int initialized                                                                     = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_f32_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_f32_u8(size_t m, size_t n, size_t p, const float *A, const float *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const float *, uint8_t *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_f32_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_f64_f32(size_t m, size_t n, size_t p, const float *A, const double *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const double *, float *, double) = NULL;
  static int initialized                                                                     = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      impl = matmul_avx512_f32_f64_f32;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      impl = matmul_avx2_f32_f64_f32;
    else
#endif
      impl = matmul_scalar_f32_f64_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_f64_f64(size_t m, size_t n, size_t p, const float *A, const double *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const double *, double *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      impl = matmul_avx512_f32_f64_f64;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      impl = matmul_avx2_f32_f64_f64;
    else
#endif
      impl = matmul_scalar_f32_f64_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_f64_i8(size_t m, size_t n, size_t p, const float *A, const double *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const double *, int8_t *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_f64_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_f64_u8(size_t m, size_t n, size_t p, const float *A, const double *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const double *, uint8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_f64_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_i8_f32(size_t m, size_t n, size_t p, const float *A, const int8_t *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const int8_t *, float *, double) = NULL;
  static int initialized                                                                     = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_i8_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_i8_f64(size_t m, size_t n, size_t p, const float *A, const int8_t *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const int8_t *, double *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_i8_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_i8_i8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const int8_t *, int8_t *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_i8_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_i8_u8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const int8_t *, uint8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_i8_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_u8_f32(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const uint8_t *, float *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_u8_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_u8_f64(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const uint8_t *, double *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_u8_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_u8_i8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const uint8_t *, int8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_u8_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_u8_u8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const uint8_t *, uint8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_u8_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f32_f32(size_t m, size_t n, size_t p, const double *A, const float *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const float *, float *, double) = NULL;
  static int initialized                                                                     = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      impl = matmul_avx512_f64_f32_f32;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      impl = matmul_avx2_f64_f32_f32;
    else
#endif
      impl = matmul_scalar_f64_f32_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f32_f64(size_t m, size_t n, size_t p, const double *A, const float *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const float *, double *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      impl = matmul_avx512_f64_f32_f64;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      impl = matmul_avx2_f64_f32_f64;
    else
#endif
      impl = matmul_scalar_f64_f32_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f32_i8(size_t m, size_t n, size_t p, const double *A, const float *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const float *, int8_t *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_f32_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f32_u8(size_t m, size_t n, size_t p, const double *A, const float *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const float *, uint8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_f32_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f64_f32(size_t m, size_t n, size_t p, const double *A, const double *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const double *, float *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      impl = matmul_avx512_f64_f64_f32;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      impl = matmul_avx2_f64_f64_f32;
    else
#endif
      impl = matmul_scalar_f64_f64_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C,
                               double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const double *, double *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      impl = matmul_avx512_f64_f64_f64;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      impl = matmul_avx2_f64_f64_f64;
    else
#endif
      impl = matmul_scalar_f64_f64_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f64_i8(size_t m, size_t n, size_t p, const double *A, const double *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const double *, int8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_f64_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f64_u8(size_t m, size_t n, size_t p, const double *A, const double *B, uint8_t *C,
                              double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const double *, uint8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_f64_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_i8_f32(size_t m, size_t n, size_t p, const double *A, const int8_t *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const int8_t *, float *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_i8_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_i8_f64(size_t m, size_t n, size_t p, const double *A, const int8_t *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const int8_t *, double *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_i8_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_i8_i8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const int8_t *, int8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_i8_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_i8_u8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const int8_t *, uint8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_i8_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_u8_f32(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const uint8_t *, float *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_u8_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_u8_f64(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, double *C,
                              double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const uint8_t *, double *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_u8_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_u8_i8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const uint8_t *, int8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_u8_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_u8_u8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, uint8_t *C,
                             double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const uint8_t *, uint8_t *, double) = NULL;
  static int initialized                                                                         = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_u8_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_f32_f32(size_t m, size_t n, size_t p, const int8_t *A, const float *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const float *, float *, double) = NULL;
  static int initialized                                                                     = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_f32_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_f32_f64(size_t m, size_t n, size_t p, const int8_t *A, const float *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const float *, double *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_f32_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_f32_i8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const float *, int8_t *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_f32_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_f32_u8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const float *, uint8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_f32_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_f64_f32(size_t m, size_t n, size_t p, const int8_t *A, const double *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const double *, float *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_f64_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_f64_f64(size_t m, size_t n, size_t p, const int8_t *A, const double *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const double *, double *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_f64_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_f64_i8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const double *, int8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_f64_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_f64_u8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const double *, uint8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_f64_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_i8_f32(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const int8_t *, float *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_i8_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_i8_f64(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const int8_t *, double *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_i8_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_i8_i8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const int8_t *, int8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_i8_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_i8_u8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const int8_t *, uint8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_i8_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_u8_f32(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const uint8_t *, float *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_u8_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_u8_f64(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const uint8_t *, double *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_u8_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_u8_i8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const uint8_t *, int8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_u8_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_i8_u8_u8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const int8_t *, const uint8_t *, uint8_t *, double) = NULL;
  static int initialized                                                                         = 0;
  if (!initialized) {
    impl        = matmul_scalar_i8_u8_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_f32_f32(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const float *, float *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_f32_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_f32_f64(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const float *, double *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_f32_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_f32_i8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const float *, int8_t *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_f32_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_f32_u8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const float *, uint8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_f32_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_f64_f32(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const double *, float *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_f64_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_f64_f64(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, double *C,
                              double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const double *, double *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_f64_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_f64_i8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const double *, int8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_f64_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_f64_u8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, uint8_t *C,
                             double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const double *, uint8_t *, double) = NULL;
  static int initialized                                                                         = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_f64_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_i8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const int8_t *, float *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_i8_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_i8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const int8_t *, double *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_i8_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_i8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const int8_t *, int8_t *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_i8_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const int8_t *, uint8_t *, double) = NULL;
  static int initialized                                                                         = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_i8_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_u8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, float *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const uint8_t *, float *, double) = NULL;
  static int initialized                                                                        = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_u8_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_u8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, double *C,
                             double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const uint8_t *, double *, double) = NULL;
  static int initialized                                                                         = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_u8_f64;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_u8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, int8_t *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const uint8_t *, int8_t *, double) = NULL;
  static int initialized                                                                         = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_u8_i8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_u8_u8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, uint8_t *C,
                            double scale) {
  static int (*impl)(size_t, size_t, size_t, const uint8_t *, const uint8_t *, uint8_t *, double) = NULL;
  static int initialized                                                                          = 0;
  if (!initialized) {
    impl        = matmul_scalar_u8_u8_u8;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

int matmul_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
  return _matmul_f32_f32_f32(m, n, p, A, B, C, scale);
}
int matmul_f32_f32_f64(size_t m, size_t n, size_t p, const float *A, const float *B, double *C, double scale) {
  return _matmul_f32_f32_f64(m, n, p, A, B, C, scale);
}
int matmul_f32_f32_i8(size_t m, size_t n, size_t p, const float *A, const float *B, int8_t *C, double scale) {
  return _matmul_f32_f32_i8(m, n, p, A, B, C, scale);
}
int matmul_f32_f32_u8(size_t m, size_t n, size_t p, const float *A, const float *B, uint8_t *C, double scale) {
  return _matmul_f32_f32_u8(m, n, p, A, B, C, scale);
}
int matmul_f32_f64_f32(size_t m, size_t n, size_t p, const float *A, const double *B, float *C, double scale) {
  return _matmul_f32_f64_f32(m, n, p, A, B, C, scale);
}
int matmul_f32_f64_f64(size_t m, size_t n, size_t p, const float *A, const double *B, double *C, double scale) {
  return _matmul_f32_f64_f64(m, n, p, A, B, C, scale);
}
int matmul_f32_f64_i8(size_t m, size_t n, size_t p, const float *A, const double *B, int8_t *C, double scale) {
  return _matmul_f32_f64_i8(m, n, p, A, B, C, scale);
}
int matmul_f32_f64_u8(size_t m, size_t n, size_t p, const float *A, const double *B, uint8_t *C, double scale) {
  return _matmul_f32_f64_u8(m, n, p, A, B, C, scale);
}
int matmul_f32_i8_f32(size_t m, size_t n, size_t p, const float *A, const int8_t *B, float *C, double scale) {
  return _matmul_f32_i8_f32(m, n, p, A, B, C, scale);
}
int matmul_f32_i8_f64(size_t m, size_t n, size_t p, const float *A, const int8_t *B, double *C, double scale) {
  return _matmul_f32_i8_f64(m, n, p, A, B, C, scale);
}
int matmul_f32_i8_i8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, int8_t *C, double scale) {
  return _matmul_f32_i8_i8(m, n, p, A, B, C, scale);
}
int matmul_f32_i8_u8(size_t m, size_t n, size_t p, const float *A, const int8_t *B, uint8_t *C, double scale) {
  return _matmul_f32_i8_u8(m, n, p, A, B, C, scale);
}
int matmul_f32_u8_f32(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, float *C, double scale) {
  return _matmul_f32_u8_f32(m, n, p, A, B, C, scale);
}
int matmul_f32_u8_f64(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, double *C, double scale) {
  return _matmul_f32_u8_f64(m, n, p, A, B, C, scale);
}
int matmul_f32_u8_i8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, int8_t *C, double scale) {
  return _matmul_f32_u8_i8(m, n, p, A, B, C, scale);
}
int matmul_f32_u8_u8(size_t m, size_t n, size_t p, const float *A, const uint8_t *B, uint8_t *C, double scale) {
  return _matmul_f32_u8_u8(m, n, p, A, B, C, scale);
}
int matmul_f64_f32_f32(size_t m, size_t n, size_t p, const double *A, const float *B, float *C, double scale) {
  return _matmul_f64_f32_f32(m, n, p, A, B, C, scale);
}
int matmul_f64_f32_f64(size_t m, size_t n, size_t p, const double *A, const float *B, double *C, double scale) {
  return _matmul_f64_f32_f64(m, n, p, A, B, C, scale);
}
int matmul_f64_f32_i8(size_t m, size_t n, size_t p, const double *A, const float *B, int8_t *C, double scale) {
  return _matmul_f64_f32_i8(m, n, p, A, B, C, scale);
}
int matmul_f64_f32_u8(size_t m, size_t n, size_t p, const double *A, const float *B, uint8_t *C, double scale) {
  return _matmul_f64_f32_u8(m, n, p, A, B, C, scale);
}
int matmul_f64_f64_f32(size_t m, size_t n, size_t p, const double *A, const double *B, float *C, double scale) {
  return _matmul_f64_f64_f32(m, n, p, A, B, C, scale);
}
int matmul_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale) {
  return _matmul_f64_f64_f64(m, n, p, A, B, C, scale);
}
int matmul_f64_f64_i8(size_t m, size_t n, size_t p, const double *A, const double *B, int8_t *C, double scale) {
  return _matmul_f64_f64_i8(m, n, p, A, B, C, scale);
}
int matmul_f64_f64_u8(size_t m, size_t n, size_t p, const double *A, const double *B, uint8_t *C, double scale) {
  return _matmul_f64_f64_u8(m, n, p, A, B, C, scale);
}
int matmul_f64_i8_f32(size_t m, size_t n, size_t p, const double *A, const int8_t *B, float *C, double scale) {
  return _matmul_f64_i8_f32(m, n, p, A, B, C, scale);
}
int matmul_f64_i8_f64(size_t m, size_t n, size_t p, const double *A, const int8_t *B, double *C, double scale) {
  return _matmul_f64_i8_f64(m, n, p, A, B, C, scale);
}
int matmul_f64_i8_i8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, int8_t *C, double scale) {
  return _matmul_f64_i8_i8(m, n, p, A, B, C, scale);
}
int matmul_f64_i8_u8(size_t m, size_t n, size_t p, const double *A, const int8_t *B, uint8_t *C, double scale) {
  return _matmul_f64_i8_u8(m, n, p, A, B, C, scale);
}
int matmul_f64_u8_f32(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, float *C, double scale) {
  return _matmul_f64_u8_f32(m, n, p, A, B, C, scale);
}
int matmul_f64_u8_f64(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, double *C, double scale) {
  return _matmul_f64_u8_f64(m, n, p, A, B, C, scale);
}
int matmul_f64_u8_i8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, int8_t *C, double scale) {
  return _matmul_f64_u8_i8(m, n, p, A, B, C, scale);
}
int matmul_f64_u8_u8(size_t m, size_t n, size_t p, const double *A, const uint8_t *B, uint8_t *C, double scale) {
  return _matmul_f64_u8_u8(m, n, p, A, B, C, scale);
}
int matmul_i8_f32_f32(size_t m, size_t n, size_t p, const int8_t *A, const float *B, float *C, double scale) {
  return _matmul_i8_f32_f32(m, n, p, A, B, C, scale);
}
int matmul_i8_f32_f64(size_t m, size_t n, size_t p, const int8_t *A, const float *B, double *C, double scale) {
  return _matmul_i8_f32_f64(m, n, p, A, B, C, scale);
}
int matmul_i8_f32_i8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, int8_t *C, double scale) {
  return _matmul_i8_f32_i8(m, n, p, A, B, C, scale);
}
int matmul_i8_f32_u8(size_t m, size_t n, size_t p, const int8_t *A, const float *B, uint8_t *C, double scale) {
  return _matmul_i8_f32_u8(m, n, p, A, B, C, scale);
}
int matmul_i8_f64_f32(size_t m, size_t n, size_t p, const int8_t *A, const double *B, float *C, double scale) {
  return _matmul_i8_f64_f32(m, n, p, A, B, C, scale);
}
int matmul_i8_f64_f64(size_t m, size_t n, size_t p, const int8_t *A, const double *B, double *C, double scale) {
  return _matmul_i8_f64_f64(m, n, p, A, B, C, scale);
}
int matmul_i8_f64_i8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, int8_t *C, double scale) {
  return _matmul_i8_f64_i8(m, n, p, A, B, C, scale);
}
int matmul_i8_f64_u8(size_t m, size_t n, size_t p, const int8_t *A, const double *B, uint8_t *C, double scale) {
  return _matmul_i8_f64_u8(m, n, p, A, B, C, scale);
}
int matmul_i8_i8_f32(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, float *C, double scale) {
  return _matmul_i8_i8_f32(m, n, p, A, B, C, scale);
}
int matmul_i8_i8_f64(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, double *C, double scale) {
  return _matmul_i8_i8_f64(m, n, p, A, B, C, scale);
}
int matmul_i8_i8_i8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, int8_t *C, double scale) {
  return _matmul_i8_i8_i8(m, n, p, A, B, C, scale);
}
int matmul_i8_i8_u8(size_t m, size_t n, size_t p, const int8_t *A, const int8_t *B, uint8_t *C, double scale) {
  return _matmul_i8_i8_u8(m, n, p, A, B, C, scale);
}
int matmul_i8_u8_f32(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, float *C, double scale) {
  return _matmul_i8_u8_f32(m, n, p, A, B, C, scale);
}
int matmul_i8_u8_f64(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, double *C, double scale) {
  return _matmul_i8_u8_f64(m, n, p, A, B, C, scale);
}
int matmul_i8_u8_i8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, int8_t *C, double scale) {
  return _matmul_i8_u8_i8(m, n, p, A, B, C, scale);
}
int matmul_i8_u8_u8(size_t m, size_t n, size_t p, const int8_t *A, const uint8_t *B, uint8_t *C, double scale) {
  return _matmul_i8_u8_u8(m, n, p, A, B, C, scale);
}
int matmul_u8_f32_f32(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, float *C, double scale) {
  return _matmul_u8_f32_f32(m, n, p, A, B, C, scale);
}
int matmul_u8_f32_f64(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, double *C, double scale) {
  return _matmul_u8_f32_f64(m, n, p, A, B, C, scale);
}
int matmul_u8_f32_i8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, int8_t *C, double scale) {
  return _matmul_u8_f32_i8(m, n, p, A, B, C, scale);
}
int matmul_u8_f32_u8(size_t m, size_t n, size_t p, const uint8_t *A, const float *B, uint8_t *C, double scale) {
  return _matmul_u8_f32_u8(m, n, p, A, B, C, scale);
}
int matmul_u8_f64_f32(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, float *C, double scale) {
  return _matmul_u8_f64_f32(m, n, p, A, B, C, scale);
}
int matmul_u8_f64_f64(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, double *C, double scale) {
  return _matmul_u8_f64_f64(m, n, p, A, B, C, scale);
}
int matmul_u8_f64_i8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, int8_t *C, double scale) {
  return _matmul_u8_f64_i8(m, n, p, A, B, C, scale);
}
int matmul_u8_f64_u8(size_t m, size_t n, size_t p, const uint8_t *A, const double *B, uint8_t *C, double scale) {
  return _matmul_u8_f64_u8(m, n, p, A, B, C, scale);
}
int matmul_u8_i8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, float *C, double scale) {
  return _matmul_u8_i8_f32(m, n, p, A, B, C, scale);
}
int matmul_u8_i8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, double *C, double scale) {
  return _matmul_u8_i8_f64(m, n, p, A, B, C, scale);
}
int matmul_u8_i8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, int8_t *C, double scale) {
  return _matmul_u8_i8_i8(m, n, p, A, B, C, scale);
}
int matmul_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale) {
  return _matmul_u8_i8_u8(m, n, p, A, B, C, scale);
}
int matmul_u8_u8_f32(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, float *C, double scale) {
  return _matmul_u8_u8_f32(m, n, p, A, B, C, scale);
}
int matmul_u8_u8_f64(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, double *C, double scale) {
  return _matmul_u8_u8_f64(m, n, p, A, B, C, scale);
}
int matmul_u8_u8_i8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, int8_t *C, double scale) {
  return _matmul_u8_u8_i8(m, n, p, A, B, C, scale);
}
int matmul_u8_u8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, uint8_t *C, double scale) {
  return _matmul_u8_u8_u8(m, n, p, A, B, C, scale);
}
