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

#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int (*matmul_u8_i8_u8)(size_t, size_t, size_t, const uint8_t *, const int8_t *, uint8_t *, double);
extern int (*matmul_f32_f32_f32)(size_t, size_t, size_t, const float *, const float *, float *, double);
extern int (*matmul_f64_f64_f64)(size_t, size_t, size_t, const double *, const double *, double *, double);
extern int (*matmul_not_implemented)(size_t, size_t, size_t, const void *, const void *, void *, double);

#define matmul(m, n, p, A, B, C, scale)                                     \
  _Generic((A),                                                             \
      const uint8_t *: _Generic((B),                                        \
          const int8_t *: _Generic((C), uint8_t *: matmul_u8_i8_u8),        \
          int8_t *: _Generic((C), uint8_t *: matmul_not_implemented),       \
          const float *: _Generic((C), uint8_t *: matmul_not_implemented),  \
          float *: _Generic((C), uint8_t *: matmul_not_implemented),        \
          const double *: _Generic((C), uint8_t *: matmul_not_implemented), \
          double *: _Generic((C), uint8_t *: matmul_not_implemented)),      \
      uint8_t *: _Generic((B),                                              \
          const int8_t *: _Generic((C), uint8_t *: matmul_u8_i8_u8),        \
          int8_t *: _Generic((C), uint8_t *: matmul_not_implemented),       \
          const float *: _Generic((C), uint8_t *: matmul_not_implemented),  \
          float *: _Generic((C), uint8_t *: matmul_not_implemented),        \
          const double *: _Generic((C), uint8_t *: matmul_not_implemented), \
          double *: _Generic((C), uint8_t *: matmul_not_implemented)),      \
      const float *: _Generic((B),                                          \
          const int8_t *: _Generic((C), uint8_t *: matmul_not_implemented), \
          int8_t *: _Generic((C), uint8_t *: matmul_not_implemented),       \
          const float *: _Generic((C), float *: matmul_f32_f32_f32),        \
          float *: _Generic((C), float *: matmul_not_implemented),          \
          const double *: _Generic((C), float *: matmul_not_implemented),   \
          double *: _Generic((C), float *: matmul_not_implemented)),        \
      float *: _Generic((B),                                                \
          const int8_t *: _Generic((C), uint8_t *: matmul_not_implemented), \
          int8_t *: _Generic((C), uint8_t *: matmul_not_implemented),       \
          const float *: _Generic((C), float *: matmul_f32_f32_f32),        \
          float *: _Generic((C), float *: matmul_not_implemented),          \
          const double *: _Generic((C), float *: matmul_not_implemented),   \
          double *: _Generic((C), float *: matmul_not_implemented)),        \
      const double *: _Generic((B),                                         \
          const int8_t *: _Generic((C), uint8_t *: matmul_not_implemented), \
          int8_t *: _Generic((C), uint8_t *: matmul_not_implemented),       \
          const float *: _Generic((C), float *: matmul_not_implemented),    \
          float *: _Generic((C), float *: matmul_not_implemented),          \
          const double *: _Generic((C), double *: matmul_f64_f64_f64),      \
          double *: _Generic((C), double *: matmul_not_implemented)),       \
      double *: _Generic((B),                                               \
          const int8_t *: _Generic((C), uint8_t *: matmul_not_implemented), \
          int8_t *: _Generic((C), uint8_t *: matmul_not_implemented),       \
          const float *: _Generic((C), float *: matmul_not_implemented),    \
          float *: _Generic((C), float *: matmul_not_implemented),          \
          const double *: _Generic((C), double *: matmul_f64_f64_f64),      \
          double *: _Generic((C), double *: matmul_not_implemented)),       \
      void *: matmul_not_implemented)((m), (n), (p), (A), (B), (C), (scale))

#ifdef __cplusplus
}
#endif

#endif
