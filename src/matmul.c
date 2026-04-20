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

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVXVNNI__
#include <immintrin.h>
#endif

#ifdef __AVX512VNNI__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define MATMUL_FLAG_SCALAR     (1 << 0)
#define MATMUL_FLAG_AVX2       (1 << 1)
#define MATMUL_FLAG_AVXVNNI    (1 << 2)
#define MATMUL_FLAG_AVX512     (1 << 3)
#define MATMUL_FLAG_AVX512VNNI (1 << 4)

typedef uint32_t matmul_feature_t;

static matmul_feature_t g_feature     = 0;
static int              g_initialized = 0;

static void init_feature(void) {
  g_feature = MATMUL_FLAG_SCALAR;
#ifdef __AVX512VNNI__
  if (__builtin_cpu_supports("avx512vnni")) g_feature |= MATMUL_FLAG_AVX512VNNI;
#endif
#ifdef __AVX512F__
  if (__builtin_cpu_supports("avx512f")) g_feature |= MATMUL_FLAG_AVX512;
#endif
#ifdef __AVXVNNI__
  if (__builtin_cpu_supports("avxvnni")) g_feature |= MATMUL_FLAG_AVXVNNI;
#endif
#ifdef __AVX2__
  if (__builtin_cpu_supports("avx2")) g_feature |= MATMUL_FLAG_AVX2;
#endif
}

matmul_feature_t matmul_get_feature(void) {
  if (!g_initialized) {
    init_feature();
    g_initialized = 1;
  }
  return g_feature;
}

const char *matmul_get_feature_name(matmul_feature_t feat) {
  if (feat & MATMUL_FLAG_AVX512VNNI) return "avx512vnni";
  if (feat & MATMUL_FLAG_AVX512) return "avx512";
  if (feat & MATMUL_FLAG_AVXVNNI) return "avxvnni";
  if (feat & MATMUL_FLAG_AVX2) return "avx2";
  if (feat & MATMUL_FLAG_SCALAR) return "scalar";
  return "unknown";
}

int matmul_scalar_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale) {
  (void)scale;
  const size_t ib = 64;
  const size_t jb = 64;
  const size_t kb = 32;

#pragma omp parallel for schedule(static)
  for (size_t ii = 0; ii < m; ii += ib) {
    size_t i_end = (ii + ib < m) ? ii + ib : m;
    for (size_t jj = 0; jj < p; jj += jb) {
      size_t  j_end = (jj + jb < p) ? jj + jb : p;
      size_t  ti    = i_end - ii;
      size_t  tj    = j_end - jj;
      int32_t acc[64 * 64];
      memset(acc, 0, ti * tj * sizeof(int32_t));

      for (size_t kk = 0; kk < n; kk += kb) {
        size_t k_end = (kk + kb < n) ? kk + kb : n;
        for (size_t i = ii; i < i_end; i++) {
          size_t li = i - ii;
          for (size_t j = jj; j < j_end; j++) {
            size_t lj  = j - jj;
            int    sum = 0;
            for (size_t k = kk; k < k_end; k++) {
              sum += (int)A[i * n + k] * (int)B[k * p + j];
            }
            acc[li * tj + lj] += sum;
          }
        }
      }

      for (size_t i = ii; i < i_end; i++) {
        size_t li = i - ii;
        for (size_t j = jj; j < j_end; j++) {
          size_t lj = j - jj;
          int    v  = acc[li * tj + lj];
          if (scale > 1.0) v = (int)(v / scale);
          if (v > 255)
            v = 255;
          else if (v < 0)
            v = 0;
          C[i * p + j] = (uint8_t)v;
        }
      }
    }
  }
  return 0;
}

#ifdef __AVX512VNNI__
static void pack_b_i8(size_t n, size_t p, const int8_t *B, int8_t *B_packed) {
  size_t n4  = n / 4;
  size_t p16 = p / 16;
  for (size_t j16 = 0; j16 < p16; j16++) {
    for (size_t k4 = 0; k4 < n4; k4++) {
      int8_t *dst = &B_packed[(j16 * n4 + k4) * 64];
      for (size_t dj = 0; dj < 16; dj++) {
        size_t j = j16 * 16 + dj;
        for (size_t dk = 0; dk < 4; dk++) {
          dst[dj * 4 + dk] = B[(k4 * 4 + dk) * p + j];
        }
      }
    }
  }
}

int matmul_avx512vnni_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C,
                               double scale) {
  (void)scale;

  size_t n4  = n / 4;
  size_t p16 = p / 16;

  int8_t *B_packed;
  if (posix_memalign((void **)&B_packed, 64, p16 * n4 * 64) != 0) return -1;
  pack_b_i8(n, p, B, B_packed);

  const uint32_t *A32 = (const uint32_t *)A;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; i++) {
    for (size_t j16 = 0; j16 < p16; j16++) {
      __m512i result = _mm512_setzero_si512();
      for (size_t k4 = 0; k4 < n4; k4++) {
        __m512i a_val = _mm512_set1_epi32(A32[i * n4 + k4]);
        __m512i b_val = _mm512_load_si512((__m512i const *)&B_packed[(j16 * n4 + k4) * 64]);
        result        = _mm512_dpbusd_epi32(result, a_val, b_val);
      }
      int32_t tmp[16] __attribute__((aligned(64)));
      _mm512_store_si512(tmp, result);
      for (size_t dj = 0; dj < 16; dj++) {
        int32_t v = tmp[dj];
        if (scale > 1.0) v = (int32_t)(v / scale);
        if (v > 255)
          v = 255;
        else if (v < 0)
          v = 0;
        C[i * p + j16 * 16 + dj] = (uint8_t)v;
      }
    }
    for (size_t j = p16 * 16; j < p; j++) {
      int32_t sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (int)A[i * n + k] * (int)B[k * p + j];
      }
      if (scale > 1.0) sum = (int32_t)(sum / scale);
      if (sum > 255)
        sum = 255;
      else if (sum < 0)
        sum = 0;
      C[i * p + j] = (uint8_t)sum;
    }
  }

  free(B_packed);
  return 0;
}
#endif

static int _matmul_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale) {
  static int initialized = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512VNNI__
    if (feat & MATMUL_FLAG_AVX512VNNI)
      matmul_u8_i8_u8 = matmul_avx512vnni_u8_i8_u8;
    else
#endif
      matmul_u8_i8_u8 = matmul_scalar_u8_i8_u8;
    initialized = 1;
  }
  return matmul_u8_i8_u8(m, n, p, A, B, C, scale);
}

int (*matmul_u8_i8_u8)(size_t, size_t, size_t, const uint8_t *, const int8_t *, uint8_t *, double) = _matmul_u8_i8_u8;

/* ========================================================================== */
/* f32_f32_f32 implementations                                                */
/* ========================================================================== */

int matmul_scalar_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
  const size_t ib = 64;
  const size_t jb = 64;
  const size_t kb = 16;

#pragma omp parallel for schedule(static)
  for (size_t ii = 0; ii < m; ii += ib) {
    size_t i_end = (ii + ib < m) ? ii + ib : m;
    for (size_t jj = 0; jj < p; jj += jb) {
      size_t j_end = (jj + jb < p) ? jj + jb : p;
      size_t ti    = i_end - ii;
      size_t tj    = j_end - jj;
      double acc[64 * 64];
      memset(acc, 0, ti * tj * sizeof(double));

      for (size_t kk = 0; kk < n; kk += kb) {
        size_t k_end = (kk + kb < n) ? kk + kb : n;
        for (size_t i = ii; i < i_end; i++) {
          size_t li = i - ii;
          for (size_t j = jj; j < j_end; j++) {
            size_t lj  = j - jj;
            double sum = 0.0;
            for (size_t k = kk; k < k_end; k++) {
              sum += (double)A[i * n + k] * (double)B[k * p + j];
            }
            acc[li * tj + lj] += sum;
          }
        }
      }

      for (size_t i = ii; i < i_end; i++) {
        size_t li = i - ii;
        for (size_t j = jj; j < j_end; j++) {
          size_t lj = j - jj;
          double v  = acc[li * tj + lj];
          if (scale > 1.0) v /= scale;
          C[i * p + j] = (float)v;
        }
      }
    }
  }
  return 0;
}

#ifdef __AVX2__
static void pack_b_f32(size_t n, size_t p, const float *B, float *B_packed) {
  size_t n8 = n / 8;
  size_t p8 = p / 8;
  for (size_t j8 = 0; j8 < p8; j8++) {
    for (size_t k8 = 0; k8 < n8; k8++) {
      float *dst = &B_packed[(j8 * n8 + k8) * 64];
      for (size_t dk = 0; dk < 8; dk++) {
        size_t k = k8 * 8 + dk;
        for (size_t dj = 0; dj < 8; dj++) {
          size_t j         = j8 * 8 + dj;
          dst[dk * 8 + dj] = B[k * p + j];
        }
      }
    }
  }
}

int matmul_avx2_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
  const size_t ib = 64;
  const size_t jb = 64;
  const size_t kb = 16;

  size_t n8 = n / 8;
  size_t p8 = p / 8;

  float *B_packed;
  if (posix_memalign((void **)&B_packed, 64, p8 * n8 * 64 * sizeof(float)) != 0) return -1;
  pack_b_f32(n, p, B, B_packed);

#pragma omp parallel for schedule(static)
  for (size_t ii = 0; ii < m; ii += ib) {
    size_t i_end = (ii + ib < m) ? ii + ib : m;
    for (size_t jj = 0; jj < p8 * 8; jj += jb) {
      size_t j_end = (jj + jb < p8 * 8) ? jj + jb : p8 * 8;
      size_t ti    = i_end - ii;
      size_t tj    = j_end - jj;
      float  acc[64 * 64];
      memset(acc, 0, ti * tj * sizeof(float));

      for (size_t kk = 0; kk < n; kk += kb) {
        size_t k_end  = (kk + kb < n) ? kk + kb : n;
        size_t k_end8 = kk + (k_end - kk) / 8 * 8;
        for (size_t i = ii; i < i_end; i++) {
          size_t li = i - ii;
          for (size_t j = jj; j + 8 <= j_end; j += 8) {
            size_t lj    = j - jj;
            size_t j8idx = j / 8;
            __m256 acc0  = _mm256_setzero_ps();
            __m256 acc1  = _mm256_setzero_ps();
            __m256 acc2  = _mm256_setzero_ps();
            __m256 acc3  = _mm256_setzero_ps();
            size_t k     = kk;
            for (; k + 4 <= k_end8; k += 4) {
              size_t k8_0 = k / 8, dk_0 = k % 8;
              size_t k8_1 = (k + 1) / 8, dk_1 = (k + 1) % 8;
              size_t k8_2 = (k + 2) / 8, dk_2 = (k + 2) % 8;
              size_t k8_3 = (k + 3) / 8, dk_3 = (k + 3) % 8;
              __m256 a0 = _mm256_set1_ps(A[i * n + k]);
              __m256 a1 = _mm256_set1_ps(A[i * n + k + 1]);
              __m256 a2 = _mm256_set1_ps(A[i * n + k + 2]);
              __m256 a3 = _mm256_set1_ps(A[i * n + k + 3]);
              __m256 b0 = _mm256_load_ps(&B_packed[(j8idx * n8 + k8_0) * 64 + dk_0 * 8]);
              __m256 b1 = _mm256_load_ps(&B_packed[(j8idx * n8 + k8_1) * 64 + dk_1 * 8]);
              __m256 b2 = _mm256_load_ps(&B_packed[(j8idx * n8 + k8_2) * 64 + dk_2 * 8]);
              __m256 b3 = _mm256_load_ps(&B_packed[(j8idx * n8 + k8_3) * 64 + dk_3 * 8]);
              acc0      = _mm256_fmadd_ps(a0, b0, acc0);
              acc1      = _mm256_fmadd_ps(a1, b1, acc1);
              acc2      = _mm256_fmadd_ps(a2, b2, acc2);
              acc3      = _mm256_fmadd_ps(a3, b3, acc3);
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);
            for (; k < k_end; k++) {
              size_t k8 = k / 8, dk = k % 8;
              __m256 a_bcast = _mm256_set1_ps(A[i * n + k]);
              __m256 b_val   = _mm256_load_ps(&B_packed[(j8idx * n8 + k8) * 64 + dk * 8]);
              acc0           = _mm256_fmadd_ps(a_bcast, b_val, acc0);
            }
            float tmp[8] __attribute__((aligned(32)));
            _mm256_store_ps(tmp, acc0);
            for (size_t dj = 0; dj < 8; dj++) acc[li * tj + lj + dj] += tmp[dj];
          }
          for (size_t j = jj + (tj / 8) * 8; j < j_end; j++) {
            size_t lj = j - jj;
            for (size_t k = kk; k < k_end; k++) {
              acc[li * tj + lj] += (double)A[i * n + k] * (double)B[k * p + j];
            }
          }
        }
      }

      for (size_t i = ii; i < i_end; i++) {
        size_t li = i - ii;
        for (size_t j = jj; j < j_end; j++) {
          size_t lj = j - jj;
          float  v  = acc[li * tj + lj];
          if (scale > 1.0) v /= (float)scale;
          C[i * p + j] = v;
        }
      }
    }
  }

  for (size_t i = 0; i < m; i++) {
    for (size_t j = p8 * 8; j < p; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale > 1.0) sum /= scale;
      C[i * p + j] = (float)sum;
    }
  }

  free(B_packed);
  return 0;
}
#endif

#ifdef __AVX512F__
static void pack_b_f32_512(size_t n, size_t p, const float *B, float *B_packed) {
  size_t n16 = n / 16;
  size_t p16 = p / 16;
  for (size_t j16 = 0; j16 < p16; j16++) {
    for (size_t k16 = 0; k16 < n16; k16++) {
      float *dst = &B_packed[(j16 * n16 + k16) * 256];
      for (size_t dk = 0; dk < 16; dk++) {
        size_t k = k16 * 16 + dk;
        for (size_t dj = 0; dj < 16; dj++) {
          size_t j          = j16 * 16 + dj;
          dst[dk * 16 + dj] = B[k * p + j];
        }
      }
    }
  }
}

int matmul_avx512_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
  const size_t ib = 64;
  const size_t jb = 64;
  const size_t kb = 16;

  size_t n16 = n / 16;
  size_t p16 = p / 16;

  float *B_packed;
  if (posix_memalign((void **)&B_packed, 64, p16 * n16 * 256 * sizeof(float)) != 0) return -1;
  pack_b_f32_512(n, p, B, B_packed);

#pragma omp parallel for schedule(static)
  for (size_t ii = 0; ii < m; ii += ib) {
    size_t i_end = (ii + ib < m) ? ii + ib : m;
    for (size_t jj = 0; jj < p16 * 16; jj += jb) {
      size_t j_end = (jj + jb < p16 * 16) ? jj + jb : p16 * 16;
      size_t ti    = i_end - ii;
      size_t tj    = j_end - jj;
      float  acc[64 * 64];
      memset(acc, 0, ti * tj * sizeof(float));

      for (size_t kk = 0; kk < n; kk += kb) {
        size_t k_end   = (kk + kb < n) ? kk + kb : n;
        size_t k_end16 = kk + (k_end - kk) / 16 * 16;
        for (size_t i = ii; i < i_end; i++) {
          size_t li = i - ii;
          for (size_t j = jj; j + 16 <= j_end; j += 16) {
            size_t lj     = j - jj;
            size_t j16idx = j / 16;
            __m512 acc0   = _mm512_setzero_ps();
            __m512 acc1   = _mm512_setzero_ps();
            __m512 acc2   = _mm512_setzero_ps();
            __m512 acc3   = _mm512_setzero_ps();
            size_t k      = kk;
            for (; k + 4 <= k_end16; k += 4) {
              size_t k16_0 = k / 16, dk_0 = k % 16;
              size_t k16_1 = (k + 1) / 16, dk_1 = (k + 1) % 16;
              size_t k16_2 = (k + 2) / 16, dk_2 = (k + 2) % 16;
              size_t k16_3 = (k + 3) / 16, dk_3 = (k + 3) % 16;
              __m512 a0 = _mm512_set1_ps(A[i * n + k]);
              __m512 a1 = _mm512_set1_ps(A[i * n + k + 1]);
              __m512 a2 = _mm512_set1_ps(A[i * n + k + 2]);
              __m512 a3 = _mm512_set1_ps(A[i * n + k + 3]);
              __m512 b0 = _mm512_load_ps(&B_packed[(j16idx * n16 + k16_0) * 256 + dk_0 * 16]);
              __m512 b1 = _mm512_load_ps(&B_packed[(j16idx * n16 + k16_1) * 256 + dk_1 * 16]);
              __m512 b2 = _mm512_load_ps(&B_packed[(j16idx * n16 + k16_2) * 256 + dk_2 * 16]);
              __m512 b3 = _mm512_load_ps(&B_packed[(j16idx * n16 + k16_3) * 256 + dk_3 * 16]);
              acc0      = _mm512_fmadd_ps(a0, b0, acc0);
              acc1      = _mm512_fmadd_ps(a1, b1, acc1);
              acc2      = _mm512_fmadd_ps(a2, b2, acc2);
              acc3      = _mm512_fmadd_ps(a3, b3, acc3);
            }
            acc0 = _mm512_add_ps(acc0, acc1);
            acc2 = _mm512_add_ps(acc2, acc3);
            acc0 = _mm512_add_ps(acc0, acc2);
            for (; k < k_end; k++) {
              size_t k16 = k / 16, dk = k % 16;
              __m512 a_bcast = _mm512_set1_ps(A[i * n + k]);
              __m512 b_val   = _mm512_load_ps(&B_packed[(j16idx * n16 + k16) * 256 + dk * 16]);
              acc0           = _mm512_fmadd_ps(a_bcast, b_val, acc0);
            }
            float tmp[16] __attribute__((aligned(64)));
            _mm512_store_ps(tmp, acc0);
            for (size_t dj = 0; dj < 16; dj++) acc[li * tj + lj + dj] += tmp[dj];
          }
          for (size_t j = jj + (tj / 16) * 16; j < j_end; j++) {
            size_t lj = j - jj;
            for (size_t k = kk; k < k_end; k++) {
              acc[li * tj + lj] += (double)A[i * n + k] * (double)B[k * p + j];
            }
          }
        }
      }

      for (size_t i = ii; i < i_end; i++) {
        size_t li = i - ii;
        for (size_t j = jj; j < j_end; j++) {
          size_t lj = j - jj;
          float  v  = acc[li * tj + lj];
          if (scale > 1.0) v /= (float)scale;
          C[i * p + j] = v;
        }
      }
    }
  }

  for (size_t i = 0; i < m; i++) {
    for (size_t j = p16 * 16; j < p; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) {
        sum += (double)A[i * n + k] * (double)B[k * p + j];
      }
      if (scale > 1.0) sum /= scale;
      C[i * p + j] = (float)sum;
    }
  }

  free(B_packed);
  return 0;
}
#endif

static int _matmul_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
  static int initialized = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      matmul_f32_f32_f32 = matmul_avx512_f32_f32_f32;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      matmul_f32_f32_f32 = matmul_avx2_f32_f32_f32;
    else
#endif
      matmul_f32_f32_f32 = matmul_scalar_f32_f32_f32;
    initialized = 1;
  }
  return matmul_f32_f32_f32(m, n, p, A, B, C, scale);
}

int (*matmul_f32_f32_f32)(size_t, size_t, size_t, const float *, const float *, float *, double) = _matmul_f32_f32_f32;

/* ========================================================================== */
/* f64_f64_f64 implementations                                                */
/* ========================================================================== */

int matmul_scalar_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale) {
  const size_t ib = 64;
  const size_t jb = 64;
  const size_t kb = 8;

#pragma omp parallel for schedule(static)
  for (size_t ii = 0; ii < m; ii += ib) {
    size_t i_end = (ii + ib < m) ? ii + ib : m;
    for (size_t jj = 0; jj < p; jj += jb) {
      size_t j_end = (jj + jb < p) ? jj + jb : p;
      size_t ti    = i_end - ii;
      size_t tj    = j_end - jj;
      double acc[64 * 64];
      memset(acc, 0, ti * tj * sizeof(double));

      for (size_t kk = 0; kk < n; kk += kb) {
        size_t k_end = (kk + kb < n) ? kk + kb : n;
        for (size_t i = ii; i < i_end; i++) {
          size_t li = i - ii;
          for (size_t j = jj; j < j_end; j++) {
            size_t lj  = j - jj;
            double sum = 0.0;
            for (size_t k = kk; k < k_end; k++) {
              sum += A[i * n + k] * B[k * p + j];
            }
            acc[li * tj + lj] += sum;
          }
        }
      }

      for (size_t i = ii; i < i_end; i++) {
        size_t li = i - ii;
        for (size_t j = jj; j < j_end; j++) {
          size_t lj = j - jj;
          double v  = acc[li * tj + lj];
          if (scale > 1.0) v /= scale;
          C[i * p + j] = v;
        }
      }
    }
  }
  return 0;
}

#ifdef __AVX2__
static void pack_b_f64(size_t n, size_t p, const double *B, double *B_packed) {
  size_t n4 = n / 4;
  size_t p4 = p / 4;
  for (size_t j4 = 0; j4 < p4; j4++) {
    for (size_t k4 = 0; k4 < n4; k4++) {
      double *dst = &B_packed[(j4 * n4 + k4) * 16];
      for (size_t dk = 0; dk < 4; dk++) {
        size_t k = k4 * 4 + dk;
        for (size_t dj = 0; dj < 4; dj++) {
          size_t j         = j4 * 4 + dj;
          dst[dk * 4 + dj] = B[k * p + j];
        }
      }
    }
  }
}

int matmul_avx2_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale) {
  const size_t ib = 64;
  const size_t jb = 64;
  const size_t kb = 8;

  size_t n4 = n / 4;
  size_t p4 = p / 4;

  double *B_packed;
  if (posix_memalign((void **)&B_packed, 64, p4 * n4 * 16 * sizeof(double)) != 0) return -1;
  pack_b_f64(n, p, B, B_packed);

#pragma omp parallel for schedule(static)
  for (size_t ii = 0; ii < m; ii += ib) {
    size_t i_end = (ii + ib < m) ? ii + ib : m;
    for (size_t jj = 0; jj < p4 * 4; jj += jb) {
      size_t j_end = (jj + jb < p4 * 4) ? jj + jb : p4 * 4;
      size_t ti    = i_end - ii;
      size_t tj    = j_end - jj;
      double acc[64 * 64];
      memset(acc, 0, ti * tj * sizeof(double));

      for (size_t kk = 0; kk < n; kk += kb) {
        size_t k_end  = (kk + kb < n) ? kk + kb : n;
        size_t k_end4 = kk + (k_end - kk) / 4 * 4;
        for (size_t i = ii; i < i_end; i++) {
          size_t li = i - ii;
          for (size_t j = jj; j + 4 <= j_end; j += 4) {
            size_t  lj    = j - jj;
            size_t  j4idx = j / 4;
            __m256d acc0  = _mm256_setzero_pd();
            __m256d acc1  = _mm256_setzero_pd();
            __m256d acc2  = _mm256_setzero_pd();
            __m256d acc3  = _mm256_setzero_pd();
            size_t  k     = kk;
            for (; k + 4 <= k_end4; k += 4) {
              size_t  k4_0 = k / 4, dk_0 = k % 4;
              size_t  k4_1 = (k + 1) / 4, dk_1 = (k + 1) % 4;
              size_t  k4_2 = (k + 2) / 4, dk_2 = (k + 2) % 4;
              size_t  k4_3 = (k + 3) / 4, dk_3 = (k + 3) % 4;
              __m256d a0 = _mm256_set1_pd(A[i * n + k]);
              __m256d a1 = _mm256_set1_pd(A[i * n + k + 1]);
              __m256d a2 = _mm256_set1_pd(A[i * n + k + 2]);
              __m256d a3 = _mm256_set1_pd(A[i * n + k + 3]);
              __m256d b0 = _mm256_load_pd(&B_packed[(j4idx * n4 + k4_0) * 16 + dk_0 * 4]);
              __m256d b1 = _mm256_load_pd(&B_packed[(j4idx * n4 + k4_1) * 16 + dk_1 * 4]);
              __m256d b2 = _mm256_load_pd(&B_packed[(j4idx * n4 + k4_2) * 16 + dk_2 * 4]);
              __m256d b3 = _mm256_load_pd(&B_packed[(j4idx * n4 + k4_3) * 16 + dk_3 * 4]);
              acc0       = _mm256_fmadd_pd(a0, b0, acc0);
              acc1       = _mm256_fmadd_pd(a1, b1, acc1);
              acc2       = _mm256_fmadd_pd(a2, b2, acc2);
              acc3       = _mm256_fmadd_pd(a3, b3, acc3);
            }
            acc0 = _mm256_add_pd(acc0, acc1);
            acc2 = _mm256_add_pd(acc2, acc3);
            acc0 = _mm256_add_pd(acc0, acc2);
            for (; k < k_end; k++) {
              size_t  k4 = k / 4, dk = k % 4;
              __m256d a_bcast = _mm256_set1_pd(A[i * n + k]);
              __m256d b_val   = _mm256_load_pd(&B_packed[(j4idx * n4 + k4) * 16 + dk * 4]);
              acc0            = _mm256_fmadd_pd(a_bcast, b_val, acc0);
            }
            double tmp[4] __attribute__((aligned(32)));
            _mm256_store_pd(tmp, acc0);
            for (size_t dj = 0; dj < 4; dj++) acc[li * tj + lj + dj] += tmp[dj];
          }
          for (size_t j = jj + (tj / 4) * 4; j < j_end; j++) {
            size_t lj = j - jj;
            for (size_t k = kk; k < k_end; k++) {
              acc[li * tj + lj] += A[i * n + k] * B[k * p + j];
            }
          }
        }
      }

      for (size_t i = ii; i < i_end; i++) {
        size_t li = i - ii;
        for (size_t j = jj; j < j_end; j++) {
          size_t lj = j - jj;
          double v  = acc[li * tj + lj];
          if (scale > 1.0) v /= scale;
          C[i * p + j] = v;
        }
      }
    }
  }

  for (size_t i = 0; i < m; i++) {
    for (size_t j = p4 * 4; j < p; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * p + j];
      }
      if (scale > 1.0) sum /= scale;
      C[i * p + j] = sum;
    }
  }

  free(B_packed);
  return 0;
}
#endif

#ifdef __AVX512F__
static void pack_b_f64_512(size_t n, size_t p, const double *B, double *B_packed) {
  size_t n8 = n / 8;
  size_t p8 = p / 8;
  for (size_t j8 = 0; j8 < p8; j8++) {
    for (size_t k8 = 0; k8 < n8; k8++) {
      double *dst = &B_packed[(j8 * n8 + k8) * 64];
      for (size_t dk = 0; dk < 8; dk++) {
        size_t k = k8 * 8 + dk;
        for (size_t dj = 0; dj < 8; dj++) {
          size_t j         = j8 * 8 + dj;
          dst[dk * 8 + dj] = B[k * p + j];
        }
      }
    }
  }
}

int matmul_avx512_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C, double scale) {
  const size_t ib = 64;
  const size_t jb = 64;
  const size_t kb = 8;

  size_t n8 = n / 8;
  size_t p8 = p / 8;

  double *B_packed;
  if (posix_memalign((void **)&B_packed, 64, p8 * n8 * 64 * sizeof(double)) != 0) return -1;
  pack_b_f64_512(n, p, B, B_packed);

#pragma omp parallel for schedule(static)
  for (size_t ii = 0; ii < m; ii += ib) {
    size_t i_end = (ii + ib < m) ? ii + ib : m;
    for (size_t jj = 0; jj < p8 * 8; jj += jb) {
      size_t j_end = (jj + jb < p8 * 8) ? jj + jb : p8 * 8;
      size_t ti    = i_end - ii;
      size_t tj    = j_end - jj;
      double acc[64 * 64];
      memset(acc, 0, ti * tj * sizeof(double));

      for (size_t kk = 0; kk < n; kk += kb) {
        size_t k_end  = (kk + kb < n) ? kk + kb : n;
        size_t k_end8 = kk + (k_end - kk) / 8 * 8;
        for (size_t i = ii; i < i_end; i++) {
          size_t li = i - ii;
          for (size_t j = jj; j + 8 <= j_end; j += 8) {
            size_t  lj    = j - jj;
            size_t  j8idx = j / 8;
            __m512d acc0  = _mm512_setzero_pd();
            __m512d acc1  = _mm512_setzero_pd();
            __m512d acc2  = _mm512_setzero_pd();
            __m512d acc3  = _mm512_setzero_pd();
            size_t  k     = kk;
            for (; k + 4 <= k_end8; k += 4) {
              size_t  k8_0 = k / 8, dk_0 = k % 8;
              size_t  k8_1 = (k + 1) / 8, dk_1 = (k + 1) % 8;
              size_t  k8_2 = (k + 2) / 8, dk_2 = (k + 2) % 8;
              size_t  k8_3 = (k + 3) / 8, dk_3 = (k + 3) % 8;
              __m512d a0 = _mm512_set1_pd(A[i * n + k]);
              __m512d a1 = _mm512_set1_pd(A[i * n + k + 1]);
              __m512d a2 = _mm512_set1_pd(A[i * n + k + 2]);
              __m512d a3 = _mm512_set1_pd(A[i * n + k + 3]);
              __m512d b0 = _mm512_load_pd(&B_packed[(j8idx * n8 + k8_0) * 64 + dk_0 * 8]);
              __m512d b1 = _mm512_load_pd(&B_packed[(j8idx * n8 + k8_1) * 64 + dk_1 * 8]);
              __m512d b2 = _mm512_load_pd(&B_packed[(j8idx * n8 + k8_2) * 64 + dk_2 * 8]);
              __m512d b3 = _mm512_load_pd(&B_packed[(j8idx * n8 + k8_3) * 64 + dk_3 * 8]);
              acc0       = _mm512_fmadd_pd(a0, b0, acc0);
              acc1       = _mm512_fmadd_pd(a1, b1, acc1);
              acc2       = _mm512_fmadd_pd(a2, b2, acc2);
              acc3       = _mm512_fmadd_pd(a3, b3, acc3);
            }
            acc0 = _mm512_add_pd(acc0, acc1);
            acc2 = _mm512_add_pd(acc2, acc3);
            acc0 = _mm512_add_pd(acc0, acc2);
            for (; k < k_end; k++) {
              size_t  k8 = k / 8, dk = k % 8;
              __m512d a_bcast = _mm512_set1_pd(A[i * n + k]);
              __m512d b_val   = _mm512_load_pd(&B_packed[(j8idx * n8 + k8) * 64 + dk * 8]);
              acc0            = _mm512_fmadd_pd(a_bcast, b_val, acc0);
            }
            double tmp[8] __attribute__((aligned(64)));
            _mm512_store_pd(tmp, acc0);
            for (size_t dj = 0; dj < 8; dj++) acc[li * tj + lj + dj] += tmp[dj];
          }
          for (size_t j = jj + (tj / 8) * 8; j < j_end; j++) {
            size_t lj = j - jj;
            for (size_t k = kk; k < k_end; k++) {
              acc[li * tj + lj] += A[i * n + k] * B[k * p + j];
            }
          }
        }
      }

      for (size_t i = ii; i < i_end; i++) {
        size_t li = i - ii;
        for (size_t j = jj; j < j_end; j++) {
          size_t lj = j - jj;
          double v  = acc[li * tj + lj];
          if (scale > 1.0) v /= scale;
          C[i * p + j] = v;
        }
      }
    }
  }

  for (size_t i = 0; i < m; i++) {
    for (size_t j = p8 * 8; j < p; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * p + j];
      }
      if (scale > 1.0) sum /= scale;
      C[i * p + j] = sum;
    }
  }

  free(B_packed);
  return 0;
}
#endif

static int _matmul_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C,
                               double scale) {
  static int initialized = 0;
  if (!initialized) {
    matmul_feature_t feat = matmul_get_feature();
#ifdef __AVX512F__
    if (feat & MATMUL_FLAG_AVX512)
      matmul_f64_f64_f64 = matmul_avx512_f64_f64_f64;
    else
#endif
#ifdef __AVX2__
        if (feat & MATMUL_FLAG_AVX2)
      matmul_f64_f64_f64 = matmul_avx2_f64_f64_f64;
    else
#endif
      matmul_f64_f64_f64 = matmul_scalar_f64_f64_f64;
    initialized = 1;
  }
  return matmul_f64_f64_f64(m, n, p, A, B, C, scale);
}

int (*matmul_f64_f64_f64)(size_t, size_t, size_t, const double *, const double *, double *,
                          double) = _matmul_f64_f64_f64;
