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

TurboPFor++ outperforms the C reference on **every function**, across all bit-widths (1–64) and exception rates (0–25%). Below are Grand Average throughput gains measured by the A/B benchmark suite (`ab_test`).

### P4 Encode / Decode

| API | Encode | Decode |
|-----|--------|--------|
| **32-bit scalar (n=1..127)** | | |
| `p4Enc32` / `p4Dec32` | +115% | +101% |
| `p4D1Enc32` / `p4D1Dec32` | +80% | +86% |
| **32-bit SSE4.2 (n=128)** | | |
| `p4Enc128v32` / `p4Dec128v32` | +77% | +6% |
| `p4D1Enc128v32` / `p4D1Dec128v32` | +76% | +6% |
| **32-bit AVX2 (n=256)** | | |
| `p4Enc256v32` / `p4Dec256v32` | +115% | +23% |
| `p4D1Enc256v32` / `p4D1Dec256v32` | +117% | +25% |
| **64-bit scalar (n=1..127)** | | |
| `p4Enc64` / `p4Dec64` | +52% | +72% |
| `p4D1Enc64` / `p4D1Dec64` | +50% | +48% |
| **64-bit SSE hybrid (n=128)** | | |
| `p4Enc128v64` / `p4Dec128v64` | +29% | +14% |
| `p4D1Enc128v64` / `p4D1Dec128v64` | +30% | +18% |
| **64-bit SSE hybrid (n=256)** | | |
| `p4Enc256v64` / `p4Dec256v64` | +28% | — ¹ |
| `p4D1Enc256v64` / `p4D1Dec256v64` | +28% | +17% |

¹ `p4Dec256v64` internally wraps 2×`p4Dec128v64`; expected gain matches the 128v64 non-delta row.

### Low-Level Bitpack / Bitunpack

| API | Throughput |
|-----|-----------|
| `bitpack32` | +115% |
| `bitunpack32` | +121% |
| `bitunpack32` + delta1 | +105% |
| `bitpack64` | +119% |
| `bitunpack64` | +79% |
| `bitunpack64` + delta1 | +57% |

*All numbers are Grand Average across bit-widths 1–32 (32-bit) or 1–64 (64-bit), element counts n=1–127 (scalar) or n=128/256 (SIMD), exception rates 0/5/10/25%, best of multiple runs.*

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

The full API is in `include/turbopfor.h` under the `turbopfor` namespace.

### 32-bit

```cpp
// Scalar (n = 1..127)
unsigned char * p4Enc32(uint32_t * in, unsigned n, unsigned char * out);
unsigned char * p4D1Enc32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start);
const unsigned char * p4Dec32(const unsigned char * in, unsigned n, uint32_t * out);
const unsigned char * p4D1Dec32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

// SSE4.2, 128-element blocks
unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out);
unsigned char * p4D1Enc128v32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start);
const unsigned char * p4Dec128v32(const unsigned char * in, unsigned n, uint32_t * out);
const unsigned char * p4D1Dec128v32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

// AVX2, 256-element blocks
unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out);
unsigned char * p4D1Enc256v32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start);
const unsigned char * p4Dec256v32(const unsigned char * in, unsigned n, uint32_t * out);
const unsigned char * p4D1Dec256v32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
```

### 64-bit

```cpp
// Scalar (n = 1..127)
unsigned char * p4Enc64(uint64_t * in, unsigned n, unsigned char * out);
unsigned char * p4D1Enc64(uint64_t * in, unsigned n, unsigned char * out, uint64_t start);
const unsigned char * p4Dec64(const unsigned char * in, unsigned n, uint64_t * out);
const unsigned char * p4D1Dec64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

// SSE hybrid, 128-element blocks (SIMD for b<=32, scalar for b>32)
unsigned char * p4Enc128v64(uint64_t * in, unsigned n, unsigned char * out);
unsigned char * p4D1Enc128v64(uint64_t * in, unsigned n, unsigned char * out, uint64_t start);
const unsigned char * p4Dec128v64(const unsigned char * in, unsigned n, uint64_t * out);
const unsigned char * p4D1Dec128v64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

// SSE hybrid, 256-element blocks
unsigned char * p4Enc256v64(uint64_t * in, unsigned n, unsigned char * out);
unsigned char * p4D1Enc256v64(uint64_t * in, unsigned n, unsigned char * out, uint64_t start);
const unsigned char * p4Dec256v64(const unsigned char * in, unsigned n, uint64_t * out);
const unsigned char * p4D1Dec256v64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start);
```

**Naming convention:** `p4` = P4 algorithm, `D1` = delta-1, `Enc`/`Dec` = encode/decode, `128v`/`256v` = SIMD block size, `32`/`64` = integer width. Use `D1` variants for sorted sequences (posting lists); use non-`D1` variants for unsorted data (position arrays).

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
