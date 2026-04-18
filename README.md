# Matmul - Accelerated Matrix Multiplication Library

A lightweight, type-safe matrix multiplication library with runtime dispatch and SIMD acceleration.

**Current Implementation:** `uint8_t` × `int8_t` → `uint8_t`  
*(additional type combinations planned)*

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

    // C should be [1, 2, 4, 5]
    printf("C[0] = %u, C[1] = %u, C[2] = %u, C[3] = %u\n",
           C[0], C[1], C[2], C[3]);

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
// Currently implemented type combination (u8 × i8 → u8)
int matmul_u8_i8_u8(size_t m, size_t n, size_t p,
                    const uint8_t *A, const int8_t *B,
                    uint8_t *C, double scale);

// Scalar and SIMD variants
int matmul_scalar_u8_i8_u8(...);
int matmul_avx2_u8_i8_u8(...);
int matmul_avx512_u8_i8_u8(...);
int matmul_avxvnni_u8_i8_u8(...);
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

## Performance

On an AMD Ryzen 7 9800X3D (Zen 5, 16 cores), with AVX2/AVX-VNNI/AVX512/AVX512-VNNI enabled:

| Matrix Size | Time (ms)  | GFLOPS  |
|-------------|-----------|---------|
| 1024×1024   | ~2.4      | ~880    |
| 4096×4096   | ~173      | ~795    |
| 16384×16384 | ~12970    | ~678    |

The 16384×16384 case exceeds L3 cache (~768MB total), so performance is memory-bound. For optimal performance use matrices that fit in L3 (≤4096 on most CPUs).

## Testing

The library includes unit tests verifying correctness across all implementations:
- Run: `./test_matmul`
- Tests verify correctness against reference scalar implementation
- Output shows PASS/FAIL status for each implementation (scalar, AVX2, AVX512, dispatched)

## Implementation Notes

- **Automatic dispatch**: The first call runtime-detects CPU features and selects the optimal implementation
- **Dispatch priority**: AVX512-VNNI → AVX512 → AVX-VNNI → AVX2 → Scalar
- **Parallelization**: OpenMP `parallel for` with `static` scheduling across row blocks
- **Tiling**: Blocking factors tuned for L1/L2 cache (ib=32/64, jb=64, kb=32/64 depending on SIMD width)

## License

Licensed under custom terms (Copyright 2026 finwo); see LICENSE.md for full details.

---

*Built with C11. Zero runtime overhead for type dispatch.*
