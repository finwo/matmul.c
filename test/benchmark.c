#include "nemequ/munit.h"
#include "../src/matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static float *alloc_f32(size_t size) {
    return (float*)calloc(size, sizeof(float));
}

static double *alloc_f64(size_t size) {
    return (double*)calloc(size, sizeof(double));
}

static int8_t *alloc_i8(size_t size) {
    return (int8_t*)calloc(size, sizeof(int8_t));
}

static uint8_t *alloc_u8(size_t size) {
    return (uint8_t*)calloc(size, sizeof(uint8_t));
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static MunitResult test_bench_f32_128x128(const MunitParameter *params, void *data) {
    (void)params; (void)data;
    
    size_t m = 128, n = 128, p = 128;
    size_t size = m * n;
    float *A = alloc_f32(size);
    float *B = alloc_f32(size);
    float *C = alloc_f32(m * p);
    
    for (size_t i = 0; i < size; i++) {
        A[i] = (float)(i % 100) * 0.1f;
        B[i] = (float)((i + 7) % 100) * 0.1f;
    }
    
    double start = get_time_ms();
    for (int iter = 0; iter < 10; iter++) {
        matmul(m, n, p, A, B, C);
    }
    double elapsed = get_time_ms() - start;
    
    matmul_feature_t feat = matmul_get_feature();
    printf("f32 128x128: %.2f ms (feature: %s)\n", elapsed / 10.0, matmul_get_feature_name(feat));
    
    free(A); free(B); free(C);
    return MUNIT_OK;
}

static MunitResult test_bench_f64_128x128(const MunitParameter *params, void *data) {
    (void)params; (void)data;
    
    size_t m = 128, n = 128, p = 128;
    size_t size = m * n;
    double *A = alloc_f64(size);
    double *B = alloc_f64(size);
    double *C = alloc_f64(m * p);
    
    for (size_t i = 0; i < size; i++) {
        A[i] = (double)(i % 100) * 0.1;
        B[i] = (double)((i + 7) % 100) * 0.1;
    }
    
    double start = get_time_ms();
    for (int iter = 0; iter < 10; iter++) {
        matmul_f64(m, n, p, A, B, C);
    }
    double elapsed = get_time_ms() - start;
    
    printf("f64 128x128: %.2f ms\n", elapsed / 10.0);
    
    free(A); free(B); free(C);
    return MUNIT_OK;
}

static MunitResult test_bench_i8_128x128(const MunitParameter *params, void *data) {
    (void)params; (void)data;
    
    size_t m = 128, n = 128, p = 128;
    size_t size = m * n;
    int8_t *A = alloc_i8(size);
    int8_t *B = alloc_i8(size);
    int8_t *C = alloc_i8(m * p);
    
    for (size_t i = 0; i < size; i++) {
        A[i] = (int8_t)(i % 256 - 128);
        B[i] = (int8_t)((i + 7) % 256 - 128);
    }
    
    double start = get_time_ms();
    for (int iter = 0; iter < 10; iter++) {
        matmul_i8(m, n, p, A, B, C);
    }
    double elapsed = get_time_ms() - start;
    
    printf("i8 128x128: %.2f ms\n", elapsed / 10.0);
    
    free(A); free(B); free(C);
    return MUNIT_OK;
}

static MunitResult test_bench_u8_128x128(const MunitParameter *params, void *data) {
    (void)params; (void)data;
    
    size_t m = 128, n = 128, p = 128;
    size_t size = m * n;
    uint8_t *A = alloc_u8(size);
    uint8_t *B = alloc_u8(size);
    uint8_t *C = alloc_u8(m * p);
    
    for (size_t i = 0; i < size; i++) {
        A[i] = (uint8_t)(i % 256);
        B[i] = (uint8_t)((i + 7) % 256);
    }
    
    double start = get_time_ms();
    for (int iter = 0; iter < 10; iter++) {
        matmul_u8(m, n, p, A, B, C);
    }
    double elapsed = get_time_ms() - start;
    
    printf("u8 128x128: %.2f ms\n", elapsed / 10.0);
    
    free(A); free(B); free(C);
    return MUNIT_OK;
}

static MunitTest tests[] = {
    {"/f32-128x128", test_bench_f32_128x128, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/f64-128x128", test_bench_f64_128x128, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/i8-128x128", test_bench_i8_128x128, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {"/u8-128x128", test_bench_u8_128x128, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL},
    {NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL}
};

static const MunitSuite suite = {
    {"/benchmark", tests, NULL, 1, MUNIT_SUITE_OPTION_NONE},
};

int main(int argc, char *argv[MUNIT_ARRAY_PARAM(argc)]) {
    return munit_suite_main(&suite, NULL, argc, argv);
}