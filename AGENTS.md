# AGENTS.md — TurboPFor++

## What This Project Is

A C++20 reimplementation of the [TurboPFor](https://github.com/powturbo/TurboPFor-Integer-Compression) integer compression library, specifically the P4 (PFor / Patched Frame-of-Reference) algorithm. Targets ClickHouse inverted indexes. Provides scalar fallbacks and SIMD-accelerated (SSE4.2, AVX2) encode/decode for 32-bit and 64-bit integers.

The original C TurboPFor library is fetched at build time via CMake `FetchContent` and used as a reference implementation for correctness tests and A/B benchmarks.

## Project Goals & Constraints

1. **Performance: must beat TurboPFor C on every function.** Every encode/decode function must be faster than the C reference across all bit-widths and exception rates. Use `ab_test` benchmarks to verify — a regression on any metric is a bug.
2. **Correctness: byte-level binary compatibility with TurboPFor C.** Encoded output must be bit-identical to the C reference (after padding-bit normalization). C-encoded data must decode correctly through C++ decoders and vice versa. This is verified by `turbopfor_test`.
3. **Completeness: all C reference functions must have C++ equivalents.** See "Missing API Functions" below for current gaps.

When making changes, always run both `turbopfor_test` (correctness) AND `ab_test` (performance) to verify no regressions.

## API Completeness

All C reference functions have C++ equivalents. Both delta-1 decode (`p4D1Dec*`) and non-delta decode (`p4Dec*`) are provided for 32-bit and 64-bit.

**Encode vs Decode convention:** Both delta-1 encode (`p4D1Enc*`) and non-delta encode (`p4Enc*`) are provided. Delta-1 encoders apply delta-1 pre-encoding (computing successive differences minus 1) before P4 compression. Use `p4D1Enc*`/`p4D1Dec*` for sorted sequences. Use `p4Enc*`/`p4Dec*` for non-delta data (e.g., position arrays).

## Build & Test Commands

```bash
# Configure (prefers clang if available)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build everything (library + tests + benchmarks)
cmake --build build -j

# Run correctness tests (MUST pass — 0 failures)
./build/turbopfor_test

# Run performance benchmarks (our impl must beat ref on every metric)
./build/ab_test --simd128          # 128v32 SIMD vs C reference
./build/ab_test --simd256          # 256v32 SIMD (AVX2) vs C reference
./build/ab_test --all              # scalar p4enc32/p4d1dec32, n=1..127
./build/ab_test --p64              # 64-bit P4 scalar
./build/ab_test --simd128v64       # 128v64 SIMD (non-delta)
./build/ab_test --simd128v64d1     # 128v64 SIMD (delta1)
./build/ab_test --simd256v64d1     # 256v64 SIMD (delta1)
./build/ab_test --bitpack          # low-level 32-bit bitpack
./build/ab_test --bitunpack        # low-level 32-bit bitunpack
./build/ab_test --bitunpackd1      # low-level 32-bit bitunpack + delta1
./build/ab_test --bitpack64        # low-level 64-bit bitpack
./build/ab_test --bitunpack64      # low-level 64-bit bitunpack
./build/ab_test --bitunpackd1_64   # low-level 64-bit bitunpack + delta1
./build/ab_test --d1enc            # delta1 encode (p4d1enc32, n=1..127)

# Benchmark tuning flags
./build/ab_test --simd128 --bw 8              # specific bit-width
./build/ab_test --simd128 --bw-range 1-16     # bit-width range
./build/ab_test --simd128 --exc-pct 25        # exception percentage
./build/ab_test --simd128 --n 128             # specific block size
./build/ab_test --simd128 --iters 100000      # iterations per test
./build/ab_test --simd128 --runs 5            # number of runs

# Full benchmark suite (runs all modes, collects summary)
bash benchmarks/run_all.sh ./build/ab_test
```

### CMake Options

| Option | Default | Effect |
|--------|---------|--------|
| `ENABLE_SSE42` | `ON` | Compile SSE4.2 SIMD paths for 128v functions |
| `ENABLE_AVX2` | `ON` | Compile AVX2 SIMD paths for 256v functions |

Disable with `-DENABLE_SSE42=OFF` or `-DENABLE_AVX2=OFF`. Scalar fallbacks are always compiled.

### Build Artifacts

- `libturbopfor.a` — The static library (public API)
- `turbopfor_test` — Test executable (correctness + binary compatibility)
- `ab_test` — Benchmark executable (throughput comparison vs C reference)

## Architecture

### Namespace Hierarchy

```
turbopfor::              — Public API (include/turbopfor.h)
turbopfor::scalar::      — Scalar implementations (src/scalar/p4_scalar.h)
turbopfor::simd::        — SIMD implementations (src/simd/p4_simd.h)
turbopfor::scalar::detail:: — Internal scalar utilities (bitpack, vbyte, etc.)
turbopfor::simd::detail::   — Internal SIMD utilities (bitpack, prefix sum, etc.)
```

### Source Layout

```
include/turbopfor.h          — Public API header (the only file consumers include)
src/dispatch.cpp             — Routes public API to scalar or SIMD via #ifdef ENABLE_SSE42/ENABLE_AVX2
src/scalar/                  — Pure scalar implementations
  p4_scalar.h                — Scalar namespace declarations
  p4_scalar_internal.h       — Core utilities: bsr, bitWidth, mask, load/store, vbyte, constants
  p4_scalar_internal.cpp     — Non-inline scalar utilities (vbyte, p4Bits, writeHeader)
  p4_scalar_bitpack_impl.h   — Template-based scalar bitpacking engine (compile-time specialization)
  p4_scalar_bitunpack_impl.h — Template-based scalar bitunpacking engine
  p4bits128_scalar.h         — Optimal bit-width selection for 128-element blocks (cost model)
  p4enc32.cpp / p4d1dec32.cpp          — Scalar P4 encode/decode (arbitrary n, 32-bit)
  p4enc64.cpp / p4d1dec64.cpp          — Scalar P4 encode/decode (arbitrary n, 64-bit)
  p4enc128v32_scalar.cpp / ...         — Scalar 128v/256v encode/decode (32-bit)
  p4enc128v64_scalar.cpp / ...         — Scalar 128v/256v encode/decode (64-bit)
  bitpack128v32_scalar.cpp / ...       — Scalar vertical bitpack/unpack
src/simd/                    — SIMD-accelerated implementations
  p4_simd.h                  — SIMD namespace declarations
  p4_simd_internal.h         — 128v SIMD internals (SSE4.1/4.2)
  p4_simd_internal_256v.h    — 256v SIMD internals (AVX2), includes mm256_scan_epi32 prefix sum
  bitunpack_sse_templates.h  — SSE unpack template engine (~1760 lines, 4 code-gen strategies)
  bitunpack_avx2_templates.h — AVX2 unpack template engine
  p4enc128v32.cpp / p4d1dec128v32.cpp  — SIMD 128v P4 encode/decode
  p4enc256v32.cpp / p4d1dec256v32.cpp  — SIMD 256v P4 encode/decode (AVX2)
  p4enc128v64.cpp / ...                — SIMD 128v/256v P4 encode/decode (64-bit hybrid)
```

### Data Flow

1. **Public API** (`turbopfor::p4Enc128v32`) → `src/dispatch.cpp` → `#ifdef` selects `simd::` or `scalar::`
2. **Encode path**: `p4Enc*()` → `p4Bits*()` (bit-width analysis) → `writeHeader()` → payload (bitpack + exceptions)
3. **Decode path**: `p4D1Dec*()` → parse header → dispatch on encoding type → fused unpack+delta1+patch

### Key Design Patterns

**Two-level compile-time dispatch for bitpacking:**
Runtime bit-width `b` → function pointer table → runtime element count `n` → switch → compile-time template `<B, N>`. All shift amounts and masks become immediate values in the generated code.

**Fused decode pipeline:**
The SIMD decoders perform bit-unpacking, exception patching, and delta1 prefix-sum in a single pass within CPU registers, avoiding intermediate memory writes. This is the primary performance advantage over the C reference.

**Cold-path isolation:**
Exception-handling functions are marked `noinline` to keep the hot decode path's icache footprint minimal. The fast path (no exceptions) is a direct tail call with no stack frame overhead.

## P4 Block Format

The wire format is binary-compatible with the original C TurboPFor. The 1-byte header encodes:

| Header bits | Meaning |
|-------------|---------|
| `b & 0xC0 == 0xC0` | Constant block — all values identical |
| `b & 0x40 == 0, !(b & 0x80)` | Bitpack only — no exceptions |
| `b & 0x40 == 0, b & 0x80` | PFOR with bitmap exceptions (2nd byte = `bx`) |
| `b & 0x40 != 0` | PFOR with vbyte exceptions |

**Exception bit-width sentinel values** (internal, in `bx` after `p4Bits`):
- `0` — no exceptions
- `1..32` (or `1..64` for 64-bit) — bitmap patching with `bx` exception bits
- `MAX_BITS + 1` (33/65) — vbyte exceptions
- `MAX_BITS + 2` (34/66) — constant block

## Non-Obvious Gotchas

### Binary Format Compatibility

- **The 63→64 bit quirk (64-bit only):** TurboPFor C never encodes `b=63`; it upgrades to `b=64`. The C++ implementation must replicate this. See `writeHeader64()` and `normalizeP4Enc64()`.
- **Padding bits are undefined:** Trailing bits in the last byte of bitpacked data are garbage. Tests use `normalizeP4Enc32/64()` to mask them before byte-comparison. If you add a new encoder, your tests must do the same.
- **Vbyte encoding uses `VB_MAX=0xFD`**, reserving `0xFE` (all-zeros marker) and `0xFF` (uncompressed escape). The thresholds differ between 32-bit and 64-bit — see constants in `p4_scalar_internal.h`.

### SIMD

- **128v layout is 4-lane interleaved:** Values `v[0], v[4], v[8], ...` go to lane 0. This is NOT sequential packing.
- **256v layout is 8-lane interleaved** (same idea, 8 lanes for AVX2).
- **IP32 pair-swap for 64-bit SIMD:** When encoding 64-bit values with `b≤32`, low 32-bit halves are extracted via an interleaved pattern (IP32). The decoder must reverse this with `_mm_shuffle_epi32(..., _MM_SHUFFLE(1,0,3,2))`.
- **64-bit start overflow:** When `start > UINT32_MAX`, the fused delta1 path (which uses 32-bit SIMD prefix sum) falls back to SIMD unpack + scalar delta1 to avoid truncation.
- **MSan false positives:** SIMD loads may read beyond actually-written elements. `TURBOPFOR_MSAN_UNPOISON` is used to suppress these. If adding new SIMD code that reads partially-initialized buffers, apply it.
- **`alignas(16)` / `alignas(32)`:** All SIMD temp arrays must be aligned. Missing alignment causes segfaults.

### Build System

- **SIMD flags are set per-file**, not globally. SSE files get `-msse4.2`, AVX2 files get `-mavx2`. If you add a new SIMD source file, you must add it to both the `TURBOPFOR_SOURCES` list AND the `set_source_files_properties()` call for the appropriate instruction set.
- **Test executable compiles SIMD sources directly** (not just linking the library) so it can test both scalar and SIMD implementations regardless of library build options.
- **The C reference library is built twice:** `turbopfor_ref_base` (without `-mavx2`) and `turbopfor_ref_avx2` (with `-mavx2`), because the C code uses `#ifdef __AVX2__` guards.

### Template Code Generation

The SSE unpack templates (`bitunpack_sse_templates.h`) provide **4 code-generation strategies** for the same operation:
1. **Fully unrolled** — fastest, but ~86KB code (exceeds L1i on some CPUs)
2. **Periodic** — exploits `gcd(B, 32)` periodicity, ~5-15KB code, same instruction quality
3. **Loop-based** — smallest code, runtime loop counter
4. **64-bit accumulation** — avoids 32-bit overflow in prefix sum when `start` is large

Choose periodic for production code paths. Fully unrolled is for benchmarking only.

## Code Style

- **clang-format** is configured (`.clang-format`): WebKit base style, 4-space indent, 140-column limit, Allman braces
- **Naming:** `camelCase` for functions (`p4Enc32`, `bitWidth32`), `snake_case` for local variables, `SCREAMING_CASE` for macros and constants
- **`TURBOPFOR_ALWAYS_INLINE`** — used extensively on performance-critical inline functions
- **`TURBOPFOR_LIKELY` / `TURBOPFOR_UNLIKELY`** — branch prediction hints on hot paths
- **`__restrict`** — used on pointer parameters throughout bitpack/unpack functions
- **C++20 required** — uses `constexpr`, `if constexpr`, fold expressions, `std::index_sequence`

## Testing Approach

Tests are **binary compatibility tests**, not unit tests in the traditional sense. They verify:

1. **Encode compatibility:** C++ encoder produces byte-identical output to C encoder (after padding normalization)
2. **Decode compatibility:** C++ decoder produces identical output to C decoder
3. **Cross-decode:** C-encoded data decoded by C++ and vice versa
4. **Roundtrip correctness:** encode → decode matches expected delta1-accumulated values

Test patterns include: sequential, all-zeros, constant, random at every bit-width (1-32 or 1-64), and exception percentages (5%, 10%, 25%).

The test binary returns 0 on success, 1 on any failure. No test framework is used — just `printf` + manual pass/fail counting.

### Adding a New Function: Checklist

When adding a new codec function (e.g., `p4Dec256v32`):

1. **Scalar implementation** — add to `src/scalar/` with a function in `turbopfor::scalar::` namespace. Declare in `src/scalar/p4_scalar.h`.
2. **SIMD implementation** (if applicable) — add to `src/simd/`. Declare in `src/simd/p4_simd.h`.
3. **Dispatch** — add the public API function in `include/turbopfor.h` and the `#ifdef` dispatch in `src/dispatch.cpp`.
4. **CMakeLists.txt** — add new `.cpp` files to `TURBOPFOR_SOURCES` and (for SIMD) to the `set_source_files_properties()` call with correct ISA flags. Also add to `turbopfor_test` sources if tests need direct access.
5. **Tests** — add a binary compatibility test in `tests/` that verifies against the C reference (`extern "C"` declaration). Follow the existing pattern: encode with both C and C++, normalize padding bits, compare bytes, cross-decode.
6. **Benchmarks** — add an A/B benchmark in `benchmarks/ab_test.cpp` comparing throughput against the C reference. **Verify our implementation is faster.**
7. **Const-correctness** — decode/unpack functions take `const unsigned char *` input. Encode functions may mutate the input array (TurboPFor convention).

## Performance Engineering Notes

### Why This Implementation Is Faster

- **Fused unpack+patch+delta1:** The C reference does 3 separate passes (unpack → patch → prefix-sum). This library fuses them into one SIMD pass, cutting memory bandwidth by ~3x.
- **Compile-time bitpack specialization:** All shift/mask values are immediate constants via templates, eliminating variable-shift instructions.
- **Cold-path isolation:** Exception handling is `noinline`, keeping the fast path (no exceptions) in L1i.
- **Branchless SIMD exception patching:** Uses `pshufb` + LUT instead of branches.

### Performance Regression Checklist

If a benchmark shows regression after a change:

1. Check icache pressure — did you add too much inlined code to the hot path? The periodic-unroll template strategy keeps code within ~5-15KB total.
2. Check alignment — SIMD temp arrays must be `alignas(16)` for SSE, `alignas(32)` for AVX2.
3. Check that exception paths remain `noinline`.
4. Check that `TURBOPFOR_ALWAYS_INLINE` is on critical inner-loop functions.
5. Compare generated assembly — small changes in source can cause the compiler to make different register allocation or vectorization decisions.
