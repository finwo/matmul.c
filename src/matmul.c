#define _GNU_SOURCE
#include "matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static matmul_feature_t g_feature = MATMUL_SCALAR;

matmul_feature_t matmul_get_feature(void) {
  return g_feature;
}

const char *matmul_get_feature_name(matmul_feature_t feat) {
  switch (feat) {
    case MATMUL_AVX512:
      return "avx512";
    case MATMUL_AVX2:
      return "avx2";
    case MATMUL_AVX512_VNNI:
      return "avx512_vnni";
    case MATMUL_AVXVNNI:
      return "avx_vnni";
    case MATMUL_SCALAR:
      return "scalar";
  }
  return "unknown";
}

int matmul_scalar_f32_f32_f32(size_t m, size_t n, size_t p, const float *A, const float *B, float *C, double scale) {
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

int matmul_scalar_u8_u8_u8(size_t m, size_t n, size_t p, const uint8_t *A, const uint8_t *B, uint8_t *C, double scale) {
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
    impl        = matmul_scalar_f32_f32_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_f32_f64(size_t m, size_t n, size_t p, const float *A, const float *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const float *, double *, double) = NULL;
  static int initialized                                                                     = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_f32_f64;
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
    impl        = matmul_scalar_f32_f64_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f32_f64_f64(size_t m, size_t n, size_t p, const float *A, const double *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const float *, const double *, double *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_f32_f64_f64;
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
    impl        = matmul_scalar_f64_f32_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f32_f64(size_t m, size_t n, size_t p, const double *A, const float *B, double *C, double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const float *, double *, double) = NULL;
  static int initialized                                                                      = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_f32_f64;
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
    impl        = matmul_scalar_f64_f64_f32;
    initialized = 1;
  }
  return impl(m, n, p, A, B, C, scale);
}

static int _matmul_f64_f64_f64(size_t m, size_t n, size_t p, const double *A, const double *B, double *C,
                               double scale) {
  static int (*impl)(size_t, size_t, size_t, const double *, const double *, double *, double) = NULL;
  static int initialized                                                                       = 0;
  if (!initialized) {
    impl        = matmul_scalar_f64_f64_f64;
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
