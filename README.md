# Matmul - High-Performance Matrix Multiplication Library

A lightweight, type-safe matrix multiplication library with compile-time dispatch, scaling support, and infrastructure for SIMD acceleration.

## Installation

This library is installable through the [dep](https://github.com/finwo/dep) package manager. If using [dep-repository](https://github.com/finwo/dep-repository), installable through `dep add finwo/matmul`, or simply by adding the following line to your .dep file:

```
finwo/matmul https://git.finwo.net/lib/matmul.c/archives/heads/main.tar.gz
```

Alternatively, you can include the [matmul.c](src/matmul.c) and [matmul.h](src/matmul.h) files in your project directly.

## Features

- **64 Type Combinations**: Supports all combinations of f32/f64/i8/u8 for input matrices A, B and output C
- **Compile-Time Dispatch**: Uses C11 `_Generic` for zero-overhead type detection
- **Scale Parameter**: Divide output before writing (e.g., scale=64 for quantization emulation)
- **Future SIMD Ready**: Infrastructure in place for AVX2/AVX512/VNNI acceleration
- **No Dependencies**: Pure C11 implementation with no external libraries
- **Well Tested**: 64 unit tests covering all type combinations

## Quick Start

```c
#include "finwo/matmul.h"
#include <stdio.h>

int main() {
    // 2x3 matrix A
    float A[6] = {1, 2, 3, 4, 5, 6};
    // 3x2 matrix B
    float B[6] = {1, 0, 0, 1, 0, 0};
    // 2x2 result matrix C
    float C[4];

    // Multiply A(2x3) * B(3x2) = C(2x2)
    matmul(2, 3, 2, A, B, C, 0.0);  // scale=0: no scaling

    // C should be [1, 2, 4, 5]
    printf("C[0] = %f, C[1] = %f, C[2] = %f, C[3] = %f\n",
           C[0], C[1], C[2], C[3]);

    return 0;
}
```

Compile with: `cc -o example example.c -lm`

## API Reference

### Generic Macro (Recommended)
```c
matmul(m, n, p, A, B, C, scale);
```
- Automatically selects the correct function based on types of A, B, and C
- `scale`: Divide each output element by this value before writing (0 or 1 = no scaling)

### Direct Function Calls
Each of the 64 type combinations is available directly:
```c
// Floating point
matmul_f32_f32_f32(m, n, p, A, B, C, scale);
matmul_f32_f32_f64(m, n, p, A, B, C, scale);
matmul_f32_f64_f32(m, n, p, A, B, C, scale);
// ... etc for all 64 combinations

// Integer types
matmul_i8_i8_i8(m, n, p, A, B, C, scale);
matmul_u8_u8_u8(m, n, p, A, B, C, scale);
matmul_i8_u8_i8(m, n, p, A, B, C, scale);
// ... etc
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

# Clean build artifacts
make clean
```

Requires a C11 compiler (gcc, clang, MSVC).

## Testing

The library includes 64 unit tests covering all type combinations:
- Run: `./test_matmul`
- Tests verify correctness against reference implementations
- Output shows PASS/FAIL status for each type combination

## Future Work

SIMD acceleration infrastructure is already in place:
- Auto-dispatch functions (`_matmul_*`) replace function pointers on first call
- Ready for AVX2, AVX512, AVX512-VNNI, and AVX-VNNI implementations
- When implemented, dispatch will select best available CPU features at runtime
- Fallback chain: AVX512-VNNI → AVX512 → AVX2 → Scalar

## License

Licensed under custom terms (Copyright 2026 finwo); see LICENSE.md for full details.

---
*Built with C11. Zero runtime overhead for type dispatch.*
