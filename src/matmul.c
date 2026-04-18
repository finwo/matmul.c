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
int matmul_avx512vnni_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C,
                               double scale) {
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
        size_t k_limit = (kk + kb < n) ? kk + kb : n;

        for (size_t i = ii; i < i_end; i++) {
          size_t   li      = i - ii;
          int32_t *acc_row = &acc[li * tj];

          for (size_t j = jj; j < j_end; j += 16) {
            size_t  j_chunk = (j + 16 <= j_end) ? 16 : (j_end - j);
            __m512i result  = _mm512_setzero_si512();

            for (size_t k = kk; k < k_limit; k += 4) {
              size_t k_chunk = (k + 4 <= k_limit) ? 4 : (k_limit - k);

              uint32_t a4 = 0;
              for (size_t dk = 0; dk < k_chunk; dk++) {
                a4 |= (uint32_t)A[i * n + k + dk] << (dk * 8);
              }
              __m512i a_val = _mm512_set1_epi32(a4);

              int8_t b_buf[64] = {0};
              for (size_t dj = 0; dj < j_chunk; dj++) {
                for (size_t dk = 0; dk < k_chunk; dk++) {
                  b_buf[dj * 4 + dk] = B[(k + dk) * p + j + dj];
                }
              }
              __m512i b_val = _mm512_load_si512((__m512i const *)b_buf);

              result = _mm512_dpbusd_epi32(result, a_val, b_val);
            }

            int32_t tmp[16] __attribute__((aligned(64)));
            _mm512_store_si512(tmp, result);

            size_t j_offset = j - jj;
            for (size_t c = 0; c < j_chunk; c++) {
              acc_row[j_offset + c] += tmp[c];
            }
          }
        }
      }

      for (size_t i = ii; i < i_end; i++) {
        size_t li = i - ii;
        for (size_t j = jj; j < j_end; j++) {
          size_t  lj = j - jj;
          int32_t v  = acc[li * tj + lj];
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
