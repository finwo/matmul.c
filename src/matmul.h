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

#define __matmul_TYPE_u8  uint8_t
#define __matmul_TYPE_i8  int8_t
#define __matmul_TYPE_f32 float
#define __matmul_TYPE_f64 double

#define __matmul_EXTERN(__MATMUL_ARG_A, __MATMUL_ARG_B, __MATMUL_ARG_C) extern int (*matmul_##__MATMUL_ARG_A##_##__MATMUL_ARG_B##_##__MATMUL_ARG_C)(size_t, size_t, size_t, const __matmul_TYPE_##__MATMUL_ARG_A *, const __matmul_TYPE_##__MATMUL_ARG_B *, __matmul_TYPE_##__MATMUL_ARG_C *, double);

#define matmul_externs                                                                        \
  __matmul_EXTERN(u8 , i8 , u8) __matmul_EXTERN(u8 , i8 , f32) __matmul_EXTERN(u8 , i8 , f64) \
  __matmul_EXTERN(u8 , f32, u8) __matmul_EXTERN(u8 , f32, f32) __matmul_EXTERN(u8 , f32, f64) \
  __matmul_EXTERN(u8 , f64, u8) __matmul_EXTERN(u8 , f64, f32) __matmul_EXTERN(u8 , f64, f64) \
  __matmul_EXTERN(f32, i8 , u8) __matmul_EXTERN(f32, i8 , f32) __matmul_EXTERN(f32, i8 , f64) \
  __matmul_EXTERN(f32, f32, u8) __matmul_EXTERN(f32, f32, f32) __matmul_EXTERN(f32, f32, f64) \
  __matmul_EXTERN(f32, f64, u8) __matmul_EXTERN(f32, f64, f32) __matmul_EXTERN(f32, f64, f64) \
  __matmul_EXTERN(f64, i8 , u8) __matmul_EXTERN(f64, i8 , f32) __matmul_EXTERN(f64, i8 , f64) \
  __matmul_EXTERN(f64, f32, u8) __matmul_EXTERN(f64, f32, f32) __matmul_EXTERN(f64, f32, f64) \
  __matmul_EXTERN(f64, f64, u8) __matmul_EXTERN(f64, f64, f32) __matmul_EXTERN(f64, f64, f64)

matmul_externs

#define __matmul_C(__matmul_arg_type_a,__matmul_arg_type_b)               \
  _Generic((__MATMUL_ARG_C),                    \
    uint8_t *: matmul_##__matmul_arg_type_a##_##__matmul_arg_type_b##_u8, \
    float *: matmul_##__matmul_arg_type_a##_##__matmul_arg_type_b##_f32,  \
    double *: matmul_##__matmul_arg_type_a##_##__matmul_arg_type_b##_f64  \
  )

#define __matmul_B(__matmul_arg_type_a)                 \
  _Generic((__MATMUL_ARG_B),               \
    int8_t *: __matmul_C(__matmul_arg_type_a,i8)        \
    const int8_t *: __matmul_C(__matmul_arg_type_a,i8)  \
    float *: __matmul_C(__matmul_arg_type_a,f32)        \
    const float *: __matmul_C(__matmul_arg_type_a,f32)  \
    double *: __matmul_C(__matmul_arg_type_a,f64)       \
    const double *: __matmul_C(__matmul_arg_type_a,f64) \
  )

#define matmul(m,n,p,__MATMUL_ARG_A,__MATMUL_ARG_B,__MATMUL_ARG_C,scale) \
  _Generic((__MATMUL_ARG_A),        \
    uint8_t *: __matmul_B(u8)       \
    const uint8_t *: __matmul_B(u8) \
    float *: __matmul_B(f32)        \
    const float *: __matmul_B(f32)  \
    double *: __matmul_B(f64)       \
    const double *: __matmul_B(f64) \
  )(m,n,p,__MATMUL_ARG_A,__MATMUL_ARG_B,__MATMUL_ARG_C,scale)

#ifdef __cplusplus
}
#endif

#endif
