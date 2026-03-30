# AGENTS.md — TurboPFor++

## Overview

TurboPFor++ is a C++20 integer compression library implementing the PFor (Patched Frame of Reference) algorithm with SSE4.2 and AVX2 SIMD optimizations. It is designed for use in **ClickHouse inverted indexes** and must maintain **binary format compatibility** with the original [TurboPFor C library](https://github.com/powturbo/TurboPFor-Integer-Compression).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

- **Build system**: CMake 3.16+, C++20
- **Preferred compiler**: Clang (auto-detected; GCC also works)
- **Key options**: `-DENABLE_SSE42=ON` (default), `-DENABLE_AVX2=ON` (default)
- **Build output**: `build/` directory (gitignored)

The build fetches the original TurboPFor C library via `FetchContent` as `turbopfor_ref_base` / `turbopfor_ref_avx2` for binary compatibility testing.

## Test

```bash
./build/binary_compat_test
./build/vbyte64_test
```

Run both after any change. `binary_compat_test` is the primary test — it validates:
- Scalar ↔ C reference encode/decode compatibility (32-bit and 64-bit, n=1..127)
- SIMD ↔ scalar ↔ C reference cross-validation (128v, 256v)
- Bitpack/bitunpack compatibility at all bit widths
- 128v64 and 256v64 hybrid format roundtrips
- Delta1 decode correctness

Tests print `passed/failed` counts per test suite. Any `FAIL` line on stderr indicates a regression.

## Benchmark

```bash
./build/ab_test --simd128
./build/ab_test --simd256
./build/ab_test --simd128v64d1 --iters 300000 --runs 7
```

`ab_test` does A/B comparisons of our implementation vs. the original TurboPFor C reference. Output shows encode/decode throughput in MB/s.

## Project Structure

```
include/
  turbopfor.h          # Public API — all user-facing function declarations
src/
  dispatch.cpp         # Routes public API to scalar or SIMD based on compile flags
  scalar/              # Pure scalar implementations (reference/fallback)
    p4_scalar.h        #   Scalar dispatch header (turbopfor::scalar namespace)
    p4_scalar_internal.h/cpp
    p4enc32.cpp, p4d1dec32.cpp
    p4enc64.cpp, p4d1dec64.cpp
    bitpack*_scalar.cpp
    p4enc128v32_scalar.cpp, p4d1dec128v32_scalar.cpp
    p4enc256v32_scalar.cpp, p4d1dec256v32_scalar.cpp
    p4enc128v64_scalar.cpp, p4d1dec128v64_scalar.cpp
    p4enc256v64_scalar.cpp, p4d1dec256v64_scalar.cpp
  simd/                # SIMD-optimized implementations
    p4_simd.h          #   SIMD dispatch header (turbopfor::simd namespace)
    p4_simd_internal.h/cpp
    p4enc128v32.cpp, p4d1dec128v32.cpp     # SSE4.2 128-element 32-bit
    p4enc256v32.cpp, p4d1dec256v32.cpp     # AVX2 256-element 32-bit
    p4enc128v64.cpp, p4d1dec128v64.cpp     # SSE4.1 128-element 64-bit hybrid
    p4enc256v64.cpp, p4d1dec256v64.cpp     # 256-element 64-bit hybrid
    bitpack*_simd.cpp, bitunpack*_simd.cpp
    bitunpack_sse_templates.h, bitunpack_avx2_templates.h
tests/
  binary_compat_test.cpp    # Main correctness test suite
  vbyte64_test.cpp          # VByte64 compatibility test
benchmarks/
  ab_test.cpp               # A/B benchmark vs TurboPFor C reference
```

## Architecture

### Dispatch Pattern

Public API (`include/turbopfor.h`) → `src/dispatch.cpp` → `turbopfor::simd::*` or `turbopfor::scalar::*`

Dispatch is compile-time via `#ifdef ENABLE_SSE42` / `#ifdef ENABLE_AVX2`. There is no runtime dispatch.

### Namespace Hierarchy

| Namespace | Purpose |
|---|---|
| `turbopfor` | Public API (declared in `include/turbopfor.h`) |
| `turbopfor::scalar` | Scalar implementations |
| `turbopfor::scalar::detail` | Internal scalar helpers (bitpack, bitunpack, vbyte) |
| `turbopfor::simd` | SIMD implementations |
| `turbopfor::simd::detail` | Internal SIMD helpers |

SIMD and scalar provide **identical function signatures** in their respective namespaces. SIMD code calls into `scalar::detail` for operations that don't benefit from SIMD (e.g., vbyte encoding, small-n fallbacks).

### Function Naming

All codec functions follow this naming scheme:

```
p4{Op}{BlockSize}v{Width}
```

- **Op**: `Enc` (encode), `Dec` (decode non-delta), `D1Dec` (decode with delta1)
- **BlockSize**: `128` or `256` (SIMD block elements), omitted for scalar-only
- **Width**: `32` or `64` (integer bit width)

Examples: `p4Enc128v32`, `p4D1Dec256v64`, `p4Enc32`

### Function Signatures

```cpp
// Encode: returns pointer past end of compressed output
unsigned char * p4EncXXX(T * in, unsigned n, unsigned char * out);

// Decode with delta1: returns pointer past end of compressed input
unsigned char * p4D1DecXXX(unsigned char * in, unsigned n, T * out, T start);
```

### 64-bit Hybrid Format

The `128v64` and `256v64` variants use a **hybrid format**:
- When `b <= 32`: uses the 128v32/256v32 SIMD vertical bitpacking format (data is pair-swapped for SIMD alignment)
- When `b > 32`: falls back to scalar 64-bit horizontal bitpacking

This is a critical design choice — the pair-swap behavior for `b <= 32` must be reversed during decode.

## Code Conventions

### Style

- **Formatter**: `.clang-format` — WebKit-based, 4-space indent, 140 column limit, Allman braces
- **Header guards**: `#pragma once`
- **Variables**: `snake_case`
- **Functions**: `camelCase` (matching TurboPFor C naming convention)
- **Types/structs**: `PascalCase`
- **Constants**: `ALL_CAPS`
- **Count parameters**: Always `unsigned` (not `size_t`), matching the original C API

### SIMD Code

- Direct SSE/AVX2 intrinsics (`_mm_*`, `_mm256_*`); no inline assembly
- `alignas(16)` / `alignas(32)` for stack-allocated SIMD arrays
- `__builtin_ctz` / `__builtin_popcountll` for bit operations
- No `__uint128_t` (ClickHouse portability constraint)

### Comments

- `///` Doxygen-style on public declarations
- Implementation files use `//` block comments for format documentation
- Comments should explain **performance rationale**, not describe what code does

## Critical Constraints

1. **Binary compatibility**: Encoded output must be byte-identical to TurboPFor C (after normalizing padding bits). This is verified by `binary_compat_test`.

2. **No `__uint128_t`**: Forbidden for ClickHouse cross-platform portability.

3. **SIMD compile flags**: SIMD source files get their own compile flags via `set_source_files_properties` in CMakeLists.txt. SSE files get `-mssse3 -msse4.1 -msse4.2 -mpopcnt`; AVX2 files additionally get `-mavx2`. Scalar files must NOT use SIMD instructions.

4. **TurboPFor 63→64 quirk**: The C reference never encodes `b=63` for 64-bit values — it upgrades to `b=64`. Our implementation must match this behavior.

5. **Padding bits**: Trailing bits in bitpacked data are undefined. The `normalizeP4Enc*` functions in the test suite mask these before byte comparison.

6. **Exception encoding**: Two strategies exist depending on `bx` (exception bit width):
   - `bx <= MAX_BITS`: bitmap + packed exceptions + packed base values
   - `bx > MAX_BITS`: count + packed base values + vbyte exceptions + position list

## Agent Workflow

Previous development used a structured workflow documented in `.agents/changes/`:

```
00.request.md → 01-specification.md → 02-plan.md → 03-tasks-* → 04-commit-msg.md → PROGRESS.md
```

Commit messages use **conventional commits**: `feat:`, `fix:`, `docs:`, etc. — lowercase imperative, no scope parenthetical.
