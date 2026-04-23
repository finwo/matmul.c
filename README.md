# Matmul - Accelerated Matrix Multiplication Library

A lightweight, type-safe matrix multiplication library with runtime dispatch and SIMD acceleration.

**Current Implementations:**
- `uint8_t` × `int8_t` → `uint8_t`
- `float` × `float` → `float`
- `double` × `double` → `double`

## Installation

This library is installable through the [dep](https://github.com/finwo/dep) package manager. If using [dep-repository](https://github.com/finwo/dep-repository), installable through `dep add finwo/matmul`, or simply by adding the following line to your .dep file:

```
finwo/matmul https://git.finwo.net/lib/matmul.c/archives/heads/main.tar.gz
```

Alternatively, you can include the [matmul.c](src/matmul.c) and [matmul.h](src/matmul.h) files in your project directly.

## Features

- **SIMD Acceleration**: AVX2, AVX512, AVX-VNNI, and AVX512-VNNI with automatic CPU feature detection
- **Multi-Core Parallelization**: OpenMP-based parallel processing across matrix rows
- **Tiled Algorithm**: Cache-blocking for improved locality at large matrix sizes
- **No Dependencies**: Pure C11 implementation with no external libraries
- **Well Tested**: Unit tests verify correctness across all implementations

## Quick Start

### Quantized Multiplication (u8 × i8 → u8)
```c
#include "finwo/matmul.h"
#include <stdio.h>

int main() {
    // 2x3 matrix A (uint8_t)
    uint8_t A[6] = {1, 2, 3, 4, 5, 6};
    // 3x2 matrix B (int8_t)
    int8_t  B[6] = {1, 0, 0, 1, 0, 0};
    // 2x2 result matrix C (uint8_t)
    uint8_t C[4];

    // Multiply A(2x3) * B(3x2) = C(2x2)
    matmul(2, 3, 2, A, B, C, 0.0);  // scale=0: no scaling

    printf("C[0] = %u, C[1] = %u, C[2] = %u, C[3] = %u\n",
           C[0], C[1], C[2], C[3]);

    return 0;
}
```

### Floating Point Multiplication (f32 × f32 → f32)
```c
#include "finwo/matmul.h"
#include <stdio.h>

int main() {
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2
    float B[4] = {5.0f, 6.0f, 7.0f, 8.0f}; // 2x2
    float C[4];

    matmul(2, 2, 2, A, B, C, 1.0);

    printf("C[0] = %f\n", C[0]);
    return 0;
}
```

Compile with: `cc -o example example.c -lm -fopenmp`

## API Reference

### Generic Macro (Recommended)
```c
matmul(m, n, p, A, B, C, scale);
```
- Automatically selects the correct function based on types of A, B, and C
- `scale`: Divide each output element by this value before writing (0 or 1 = no scaling)

### Direct Function Calls
```c
// Implemented type combinations
int matmul_u8_i8_u8(size_t m, size_t n, size_t p,
                    const uint8_t *A, const int8_t *B,
                    uint8_t *C, double scale);

int matmul_f32_f32_f32(size_t m, size_t n, size_t p,
                       const float *A, const float *B,
                       float *C, double scale);

int matmul_f64_f64_f64(size_t m, size_t n, size_t p,
                       const double *A, const double *B,
                       double *C, double scale);

// Scalar and SIMD variants (internal/specialized)
int matmul_scalar_u8_i8_u8(...);
int matmul_avx512vnni_u8_i8_u8(...);
int matmul_scalar_f32_f32_f32(...);
int matmul_avx2_f32_f32_f32(...);
int matmul_avx512_f32_f32_f32(...);
```

### Type Naming Conventions
| Shorthand | C Type     | Description           |
|-----------|------------|-----------------------|
| `f32`     | `float`    | 32-bit floating point |
| `f64`     | `double`   | 64-bit floating point |
| `i8`      | `int8_t`   | Signed 8-bit integer  |
| `u8`      | `uint8_t`  | Unsigned 8-bit integer|

Function names follow pattern: `matmul_{A_type}_{B_type}_{C_type}`

### Scale Parameter
The `scale` parameter enables quantization-aware multiplication:
- `scale = 0` or `scale = 1`: No scaling (write raw result)
- `scale = 64`: Divide output by 64 before writing
- Useful for emulating: full-scale A input with quantized B input representing -2..1.984375 instead of -128..127

Implementation:
```c
if (scale != 0 && scale != 1) {
    result = result / scale;
}
```

## Building

```bash
# Compile library and tests
make

# Run tests
./test_matmul

# Run benchmarks (optional, needs ~800MB RAM for 16K×16K)
./benchmark

# Clean build artifacts
make clean
```

Requires a C11 compiler (gcc, clang, MSVC) with OpenMP support.

## Testing

The library includes unit tests verifying correctness across all implementations:
- Run: `./test_matmul`
- Tests verify correctness against reference scalar implementation
- Output shows PASS/FAIL status for each implementation (scalar, AVX2, AVX512, dispatched)

## Implementation Notes

- **Automatic dispatch**: The first call runtime-detects CPU features and selects the optimal implementation for the given types
- **Dispatch priority**:
    - `u8_i8_u8`: AVX512-VNNI → Scalar
    - `f32_f32_f32`: AVX512 → AVX2 → Scalar
    - `f64_f64_f64`: AVX512 → AVX2 → Scalar
- **Parallelization**: OpenMP `parallel for` with `static` scheduling across row blocks
- **Tiling**: Blocking factors tuned for L1/L2 cache (ib=32/64, jb=64, kb=32/64 depending on SIMD width)

## License

Licensed under custom terms (Copyright 2026 finwo); see LICENSE.md for full details.

---

*Built with C11. Zero runtime overhead for type dispatch.*
