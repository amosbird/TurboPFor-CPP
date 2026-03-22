# Critical Context for All Tasks

## Change Request Summary

We're optimizing our C++ reimplementation of TurboPFor's 128v64 SIMD codec to outperform the TurboPFor C reference by ≥+1% on **every single bit width** (b=1..64) across three codepaths: D1 decode, D1+EX decode (with bitmap exceptions), and encode.

## Specification Reference

See `01-specification.md` for full requirements. See `02-plan.md` for the technical approach.

## Project Architecture

### 128v64 Hybrid Format
The 128v64 format encodes 128 × 64-bit integers. It's **hybrid**:
- **b ≤ 32 (SIMD path)**: Uses IP32 SIMD shuffle to extract low 32 bits from uint64_t values, then delegates to 128v32 bitpacking (SSE 32-bit lane operations). On decode, the reverse: unpack 32-bit values via 128v32, then STO64 zero-extend to 64-bit.
- **b > 32 (scalar path)**: Falls back to scalar `bitunpack64`/`bitpack64`. Already +32-42% faster than TurboPFor C.

### Decode Architecture (D1 = delta-1)
The decode pipeline for b ≤ 32 is fused into a single pass per group of 4 elements:
1. Load __m128i stripe from bitstream
2. Shift/mask to extract 4 × 32-bit values
3. (D1+EX only) SSSE3 shuffle to merge exceptions from packed array
4. Delta-1 prefix scan: `ov += shift(ov,4); ov += shift(ov,8); ov += sv + cv`
5. STO64: `unpacklo(ov, zero)` → store 2 × uint64_t, extract carry `sv`, `unpackhi(ov, zero)` → store 2 × uint64_t

### Periodic-Unroll (Key Innovation)
For bit width B, the bit-offset pattern `(G*B) % 32` repeats with period P = 32/gcd(B,32). We unroll exactly one period (P groups), then loop over 32/P periods. This gives:
- **Compile-time shifts** (all `_mm_srli_epi32(iv, Offset)` have immediate constants)
- **Small code** (~100-400 bytes per case vs ~2.7KB fully-unrolled)
- **Predictable loop** (same trip count every call)

Period examples: B=1→P=32(1 iter), B=4→P=8(4 iter), B=8→P=4(8 iter), B=16→P=2(16 iter), B=28→P=8(4 iter), B=31→P=32(1 iter), B=32→P=1(32 iter).

### Current Hybrid Dispatch
`bitunpack_sse_sto64_d1_hybrid_entry<B, Count>` selects:
- **Fully-unrolled** when B=0 or P ≤ 2 (B=16, B=32)
- **Periodic-unroll** for all other B

### Encode Architecture
Current (two-pass):
1. IP32 shuffle loop: load 4 × uint64_t → SIMD shuffle → store to `uint32_t tmp[128]`
2. `bitpack128v32(tmp, out, b)`: load from tmp → bitpack → store to output

TurboPFor C (single-pass): Fuses IP32 into the bitpack macro via `#define IP32(...)`.

## Key Files

| File | Role |
|------|------|
| `src/simd/bitpack128v64_simd.cpp` | Dispatch: encode, plain decode, D1 decode, D1+EX decode. Contains `STO64_SWITCH` macro and hybrid entry selectors. **241 lines.** |
| `src/simd/bitunpack_sse_templates.h` | ALL template variants: `UnpackStepSSE_STO64_D1`, `UnpackPeriodStepSSE_STO64_D1`, `_EX` variants, entry points. Also `GCD`, `PeriodLen`, `MaskGenSSE`. **1305 lines.** |
| `src/simd/bitpack128v32_simd.cpp` | 32-bit bitpack: specialized for b=1,2,4,8,16,32 + generic loop for others. **342 lines.** |
| `src/simd/p4d1dec128v64.cpp` | Top-level P4 D1 decoder. Routes to SIMD fast path or scalar slow path. **199 lines.** |
| `src/simd/p4enc128v64.cpp` | Top-level P4 encoder. **115 lines.** |
| `src/simd/p4_simd_internal.h` | Internal declarations (`bitpack128v64`, `bitunpack128v64`, `bitunpackD1_128v64`, etc.) |
| `src/simd/p4_simd_internal.cpp` | Shared utilities (`applyDelta1_64`, `loadU64Fast`, etc.) |
| `benchmarks/ab_test.cpp` | Benchmark harness. |

## Key Design Decisions

- **No `__uint128_t`**: ClickHouse portability constraint.
- **Only clang 18.1.3**: No GCC-specific tricks.
- **`__m128i sv` by value, not reference**: Passing the carry register by reference forces stack spills. Always pass by value.
- **Switch dispatch, not function pointers**: `switch(b)` produces a single indirect jump; function pointer tables produce indirect calls with stack frame overhead.
- **Binary compatibility**: Our encode must produce bit-identical output to TurboPFor C.

## Build & Test Commands

```bash
# Build
cmake --build build --target ab_test -j$(nproc)
cmake --build build --target binary_compat_test vbyte64_test -j$(nproc)

# Correctness tests (MUST pass after every change)
./build/binary_compat_test && ./build/vbyte64_test

# Benchmarks
./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-64   # D1 decode
./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 1-64  # D1+EX decode
./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 1-64                  # Encode + non-D1 decode

# Single bit width (faster iteration)
./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --bw 28
```

## Coding Standards

- Use `ALWAYS_INLINE` (defined in `bitunpack_sse_templates.h`) for all template functions that should be inlined.
- Use `constexpr` for compile-time constants in templates.
- All SSE intrinsics use `_mm_` prefix functions, not inline assembly.
- Comments must explain **performance rationale**, not just what the code does.
- Keep the existing namespace structure: `turbopfor::simd::detail` for internals.

## Common Pitfalls

- **Variable shifts kill performance**: `_mm_slli_epi32(v, runtime_var)` compiles to `psrld %xmm, %xmm` (variable shift) which is much slower than `psrld $imm, %xmm`. All shifts in the inner loop MUST use compile-time constants.
- **L1i cache pressure**: The fully-unrolled switch for 33 cases × ~2.7KB = ~86KB exceeds 32KB L1i. This caused the original b=28 outlier (-11%). Monitor function sizes with `objdump -d`.
- **Alignment lottery**: Code placement within the L1i cache is non-deterministic. A change that helps one bit width may hurt another. Always benchmark ALL bit widths, not just the targeted ones.
- **`binary_compat_test` is the ground truth**: If encode changes produce different bytes, the test will catch it. Run it after every encode change.
- **Benchmark noise**: Use `--runs 7` (best of 7) and `--iters 300000`. Don't trust single-run measurements. For marginal cases (±1%), run multiple times.
