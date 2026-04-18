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

#ifndef TEST_MATMUL_SIMD_H
#define TEST_MATMUL_SIMD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MATMUL_SIMD_VERSION "1.0.0"

#define MATMUL_FLAG_SCALAR      (1 << 0)
#define MATMUL_FLAG_AVX2        (1 << 1)
#define MATMUL_FLAG_AVX512      (1 << 2)
#define MATMUL_FLAG_AVX512_VNNI (1 << 3)
#define MATMUL_FLAG_AVXVNNI     (1 << 4)

typedef uint32_t matmul_feature_t;

matmul_feature_t matmul_get_feature(void);
const char      *matmul_get_feature_name(matmul_feature_t feat);

int matmul_scalar_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale);

#ifdef __AVX2__
#include <immintrin.h>
int matmul_avx2_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C, double scale);
#endif

#ifdef __AVX512VNNI__
int matmul_avx512vnni_u8_i8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const int8_t *B, uint8_t *C,
                               double scale);
#endif

#ifdef __cplusplus
}
#endif

#endif
