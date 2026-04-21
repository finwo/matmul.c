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

#define __matmul_C(type_a,type_b)               \
  _Generic((C),                                 \
    uint8_t *: matmul_##type_a##_##type_b##_u8, \
    float *: matmul_##type_a##_##type_b##_f32,  \
    double *: matmul_##type_a##_##type_b##_f64  \
  )

#define __matmul_B(type_a)                 \
  _Generic((B),                            \
    int8_t *: __matmul_C(type_a,i8)        \
    const int8_t *: __matmul_C(type_a,i8)  \
    float *: __matmul_C(type_a,f32)        \
    const float *: __matmul_C(type_a,f32)  \
    double *: __matmul_C(type_a,f64)       \
    const double *: __matmul_C(type_a,f64) \
  )

#define matmul(m,n,p,A,B,C,scale)   \
  _Generic((A),                     \
    uint8_t *: __matmul_B(u8)       \
    const uint8_t *: __matmul_B(u8) \
    float *: __matmul_B(f32)        \
    const float *: __matmul_B(f32)  \
    double *: __matmul_B(f64)       \
    const double *: __matmul_B(f64) \
  )

#ifdef __cplusplus
}
#endif

#endif
