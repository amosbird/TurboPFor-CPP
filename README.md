# TurboPFor++

TurboPFor++ is a modern, high-performance integer compression library for C++, developed with the help of AI. It is a modernized evolution of the TurboPFor algorithm, implementing the PFor (Patched Frame of Reference) algorithm with state-of-the-art SIMD optimizations and C++20 templates.

## Overview

TurboPFor++ provides fast, scalar, and SIMD-accelerated implementations of integer compression. Currently, it supports the **P4 (PFor)** algorithm, specifically optimized for use in **ClickHouse inverted indexes**.

The library features a novel **"Fused Unpack + Patch + Delta"** decoding pipeline. By performing bit-unpacking, exception patching, and delta decoding in a single pass within CPU registers, it significantly reduces memory bandwidth pressure and branch mispredictions.

## Key Features

- **Extreme Performance**:
  - **AVX2 Optimized**: Custom AVX2 kernels for 256-integer blocks (`256v`), achieving up to **50% faster decoding** than reference implementations in high-exception scenarios.
  - **SSE4.2 Optimized**: Efficient SSE4.2 kernels for 128-integer blocks (`128v`).
  - **Branchless Design**: Critical paths are branch-free to maintain pipeline saturation even with unpredictable data distributions.
- **Modern C++**: Written in clean C++20, utilizing template metaprogramming for compile-time generation of optimal unrolled loops and immediate values.
- **Robustness**: Validated against reference scalar implementations to ensure binary compatibility and correctness.
- **Simple API**: Easy-to-use C-style API wrapped in a C++ namespace.

## Performance

TurboPFor++ has been benchmarked against standard reference implementations. The optimized fused pipeline delivers consistent gains:

| Variant | Avg Decode Speedup | Max Speedup (High Exceptions) |
|---------|-------------------|-------------------------------|
| **128v (SSE4.2)** | **+12%** | **+25%** |
| **256v (AVX2)**   | **+26%** | **+50%** |

*Benchmarks run on Intel/AMD modern architectures testing various bit-widths and exception probabilities.*

## Requirements

- **Compiler**: C++20 compatible compiler (Clang 10+, GCC 10+, MSVC 2019+).
- **Build System**: CMake 3.16 or higher.
- **Hardware**:
  - CPU with **AVX2** for `p4Enc256v32` / `p4D1Dec256v32`.
  - CPU with **SSE4.2** for `p4Enc128v32` / `p4D1Dec128v32`.
  - Scalar fallbacks are available for generic hardware.

## Quick Start

### Building

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

### Running Benchmarks

```bash
# Run the comprehensive A/B test suite
./ab_test --simd256
./ab_test --simd128
```

### Usage Example

```cpp
#include "turbopfor.h"
#include <vector>
#include <iostream>

int main() {
    // 1. Prepare data (must be multiple of 128 or 256 for SIMD variants)
    size_t n = 256; 
    std::vector<uint32_t> original(n);
    for(size_t i = 0; i < n; ++i) original[i] = i * 2; // Sample data

    // 2. Allocate buffer (worst-case size estimation)
    std::vector<unsigned char> compressed(n * 4 + 1024);
    std::vector<uint32_t> decompressed(n);

    // 3. Compress using AVX2 (256v)
    unsigned char* end_ptr = turbopfor::p4Enc256v32(
        original.data(), 
        n, 
        compressed.data()
    );
    size_t compressed_size = end_ptr - compressed.data();
    std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;

    // 4. Decompress with Delta decoding
    // 'start' is the initial value for delta decoding (usually 0 or the last value of prev block)
    turbopfor::p4D1Dec256v32(
        compressed.data(), 
        n, 
        decompressed.data(), 
        0
    );

    return 0;
}
```

## API Reference

The library exposes a straightforward API in the `turbopfor` namespace:

```cpp
namespace turbopfor {
    // 256-element blocks (AVX2)
    unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out);
    unsigned char * p4D1Dec256v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

    // 128-element blocks (SSE4.2)
    unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out);
    unsigned char * p4D1Dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

    // n-element blocks (Scalar)
    unsigned char * p4Enc32(uint32_t * in, unsigned n, unsigned char * out);
    unsigned char * p4D1Dec32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
}
```

## Project Structure

- `src/simd/`: Optimized SIMD implementations (AVX2/SSE4.2 templates).
- `src/scalar/`: Reference scalar implementations.
- `benchmarks/`: Performance testing tools (`ab_test`).
- `tests/`: Correctness and binary compatibility tests.
- `include/`: Public headers.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
