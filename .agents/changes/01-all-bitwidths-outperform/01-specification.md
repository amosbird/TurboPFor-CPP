# Specification: Make All Bit Widths Outperform TurboPFor C by ≥+1%

## Overview

Our C++ reimplementation of TurboPFor's 128v64 SIMD codec (delta-1 P4 decode/encode for 128-element blocks of 64-bit integers) currently averages +3.2% faster than TurboPFor C on D1 decode (b=1..32, no exceptions). However, 5 bit widths are below the +1% target: b=2 (-0.2%), b=4 (-0.9%), b=30 (-1.2%), b=31 (-0.4%), b=32 (+0.1%).

Additionally, encode performance is consistently -8% to -11% slower than TurboPFor C across nearly all bit widths due to a two-pass architecture (IP32 shuffle to temp buffer, then bitpack) vs TurboPFor C's fused single-pass approach.

This specification covers the systematic optimization of **three codepaths** — D1 decode, D1+EX decode (with bitmap exceptions), and encode — to achieve ≥+1% improvement over TurboPFor C on **every single bit width** b=1..64 (where b=63 maps to 64 in TurboPFor's format). The b≤32 SIMD path has the most work; the b>32 scalar path is already +32-42% faster and mainly needs verification.

## Functional Requirements

### Core Functionality

- **D1 Decode** (`bitunpackD1_128v64`): For every b in 1..64, our decode must be ≥+1% faster than TurboPFor C's `p4d1dec128v64` in the `ab_test` benchmark (300K iters, 7 runs, best-of-runs, no exceptions). For b≤32 this uses the SIMD path; for b>32 the scalar path.
- **D1+EX Decode** (`bitd1unpack128v64_ex`): For every b in 1..64, with bitmap exceptions (10% exception rate), our decode must be ≥+1% faster than TurboPFor C.
- **Encode** (`bitpack128v64`): For every b in 1..64, our encode must be ≥+1% faster than TurboPFor C's `bitpack128v64`.

### Edge Cases

- **b=0**: All deltas are zero; handled by dedicated loop (no bitstream to read). Must remain correct and fast.
- **b=32**: Trivial case (full 32-bit words, no masking). Currently +0.1%, needs dedicated fast path.
- **b>32**: Falls back to scalar `bitunpack64`/`bitpack64`. Already +32-42% faster (template specialization). Needs verification across all b=33..64 but unlikely to need optimization work.
- **Constant blocks** (header flag 0xC0): Handled separately in `p4D1Dec128v64`. Not benchmarked by `ab_test --simd128v64d1`. Out of scope.
- **Variable-byte exceptions** (header flag 0x40): Uses different exception path. Not targeted.

## Non-Functional Requirements

- **Performance**: Every bit width b=1..64 must show ≥+1% improvement in benchmarks. The b≤32 SIMD path is the primary optimization target; the b>32 scalar path is already fast but must be verified. Acceptable to regress currently-strong bit widths (e.g., b=7 at +5.5%) as long as they stay ≥+1%.
- **Correctness**: `binary_compat_test` and `vbyte64_test` must pass. Binary format compatibility with TurboPFor C is mandatory.
- **Portability**: No `__uint128_t`. Must compile with clang 18.1.3 on x86_64 with SSE4.2+SSSE3.
- **Code quality**: Systematic, principled approach. No per-bitwidth ad-hoc hacks. All optimizations must be explained by a general principle (code size, loop structure, data flow, etc.).
- **Maintainability**: Template-based architecture should remain readable. New abstractions must be documented with comments explaining the performance rationale.

## Integration Points

- **`src/simd/bitpack128v64_simd.cpp`**: Main dispatch file for encode, plain decode, D1 decode, and D1+EX decode. Contains `STO64_SWITCH` macro and hybrid entry point selectors.
- **`src/simd/bitunpack_sse_templates.h`**: All template implementations (fully-unrolled, periodic-unroll, loop-based). This is the core file for decode optimizations.
- **`src/simd/bitpack128v32_simd.cpp`**: 32-bit bitpack implementations used by the encode path. Currently has specialized versions for b=1,2,4,8,16 and a generic loop for others.
- **`src/simd/p4d1dec128v64.cpp`**: Top-level P4 decoder. Routes to fast path (fused SIMD) or slow path (scalar multi-phase) based on b+bx.
- **`src/simd/p4enc128v64.cpp`**: Top-level P4 encoder. Contains the IP32 shuffle + `bitpack128v64` call.
- **`benchmarks/ab_test.cpp`**: Benchmark harness. Key flags: `--simd128v64d1`, `--simd128v64`, `--bw`, `--bw-range`, `--exc-pct`, `--iters`, `--runs`.

## Constraints and Assumptions

### Constraints

- **Compiler**: Only clang 18.1.3 is supported. No GCC-specific tricks.
- **No `__uint128_t`**: ClickHouse portability requirement.
- **Build flags**: `-O3 -DNDEBUG -std=gnu++20 -g -ffp-contract=off -ffunction-sections -fdata-sections -fstrict-aliasing -falign-functions=64 -fomit-frame-pointer -ftree-vectorize -funroll-loops -msse2 -mssse3 -msse4.1 -msse4.2 -mpopcnt -fvectorize -mbranches-within-32B-boundaries`
- **Binary compatibility**: Encoded format must remain bit-identical to TurboPFor C.
- **Existing tests must pass**: No regressions in `binary_compat_test` or `vbyte64_test`.

### Assumptions

- Benchmark environment is stable (no other CPU-intensive processes). The `ab_test` best-of-7-runs methodology accounts for noise.
- TurboPFor C reference implementation (in `build/_deps/turbopfor_upstream-src/`) is already built and correct.
- The +1% threshold is measured on the specific benchmark machine, not as a theoretical guarantee across all hardware. The implementation should be robust enough to achieve +1% reliably.

## Scope

### In Scope

1. **D1 decode optimization** (b=1..32 SIMD path): Fix b=2, b=4, b=30, b=31, b=32 to ≥+1%.
2. **D1+EX decode optimization** (b=1..32 SIMD path): Benchmark and fix any bit widths below +1%.
3. **Encode optimization** (b=1..32 SIMD path): Eliminate the temp buffer overhead by fusing IP32 shuffle into the bitpack loop.
4. **Scalar path verification** (b=33..64): Run benchmarks to confirm ≥+1% on all scalar bit widths for decode, D1+EX, and encode. Fix if any fall below.

### Out of Scope

- b>32 scalar path (already +32-42% faster, but must be verified — see In Scope item 4)
- v32 (32-bit) codec optimization
- Variable-byte exception path
- Constant block encoding/decoding
- Non-delta (plain) decode optimization (already good)
- Changes to the benchmark harness itself
- Changes to the TurboPFor C reference implementation

## Optimization Strategy Overview

### Decode: Hybrid P Threshold Tuning + Dedicated Fast Paths

The current hybrid dispatch uses P≤2 as the threshold for fully-unrolled code. The weak cases suggest this threshold or the approach needs refinement:

| Weak Case | Period P | Current Dispatch | Hypothesis |
|-----------|----------|-----------------|------------|
| b=2 | 16 | periodic (2 iter) | Small P body with many iterations — loop overhead or code alignment |
| b=4 | 8 | periodic (4 iter) | Same as b=2 — loop overhead for short periods |
| b=30 | 16 | periodic (2 iter) | Large P body (15 span crossings) — possible code alignment issue |
| b=31 | 32 | periodic (1 iter) | P=32 means 1 iteration = full unroll, but periodic wrapper adds overhead |
| b=32 | 1 | fully-unrolled | Trivial case — dedicated memcpy-like fast path may help |

Potential approaches (to be explored during implementation):
- **Adjust P threshold**: Try P≤32 (which would make b=31 fully-unrolled), or P≤16 (adds b=2,30), or P≤8 (adds b=4).
- **Dedicated b=32 fast path**: b=32 requires no masking, no shifts — just load + D1 prefix scan + STO64.
- **`[[clang::minsize]]`** on periodic bodies: May reduce alignment sensitivity for b=30.
- **Per-bitwidth compilation units**: If all else fails, put each case in its own .cpp file to control code placement independently. This is a last resort.

### Encode: Fused IP32 Bitpack

TurboPFor C's encode fuses IP32 shuffle into the bitpack loop via `#define IP32(...)` before `#include "bitpack_.h"`. Our code does two passes:
1. IP32 shuffle: load 4×uint64_t → SIMD shuffle → store to `uint32_t tmp[128]`
2. `bitpack128v32(tmp, out, b)`: load from tmp → bitpack → store to output

The fix is to create a fused `bitpack128v64_fused()` that loads uint64_t pairs, shuffles inline, and feeds directly into the bitpack logic — eliminating 512 bytes of writes + 512 bytes of reads for the temp buffer. This matches TurboPFor C's architecture exactly.

### D1+EX Decode: Parallel Optimization

The D1+EX path uses the same hybrid dispatch framework as D1 (periodic vs fully-unrolled, same P threshold). Any decode improvements that fix the D1 path will likely carry over to D1+EX. However, D1+EX adds SSSE3 shuffle-based exception patching per group, which increases code size and may shift the optimal P threshold. Must benchmark independently.

## Success Criteria

- [ ] **D1 decode**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw B` shows ≥+1% for every B in 1..64
- [ ] **D1+EX decode**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw B` shows ≥+1% for every B in 1..64
- [ ] **Encode**: `./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw B` encode column shows ≥+1% for every B in 1..64
- [ ] `./build/binary_compat_test` passes
- [ ] `./build/vbyte64_test` passes
- [ ] No per-bitwidth ad-hoc hacks — all changes are principled and systematic

## Open Questions

- **D1+EX baseline**: We don't have fresh benchmark data for D1+EX. Need to run `--exc-pct 10` across all bit widths before making changes. Some bit widths may already be ≥+1%.
- **Encode `bitpack128v32` interaction**: Our `bitpack128v32` uses a generic loop for non-power-of-2 bit widths while TurboPFor C uses fully-unrolled per-bitwidth macros (via `BITPACK128V32`). Is the encode gap purely from the temp buffer, or also from `bitpack128v32` being slower? Need to benchmark `bitpack128v32` independently.
- **Optimal P threshold**: The best threshold may differ between D1 and D1+EX due to different code sizes per case. May need separate thresholds.
- **Code size budget**: The current hybrid function is ~60KB. After adding fused encode, total SIMD code in the hot path grows. Need to monitor L1i pressure across all three codepaths.
