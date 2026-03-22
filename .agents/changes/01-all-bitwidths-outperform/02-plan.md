# Implementation Plan: Make All Bit Widths Outperform TurboPFor C by ≥+1%

## Overview

This plan addresses three codepaths (D1 decode, D1+EX decode, encode) across the full bit width range b=1..64. The work is structured as a **measure → optimize → verify** pipeline, with each codepath treated independently but sharing underlying techniques.

The key insight driving the plan: our weak decode cases (b=2,4,30,31,32) are all caused by **code-size/loop-overhead tradeoffs** in the hybrid periodic/fully-unrolled dispatch, while our encode weakness is an **architectural mismatch** (two-pass with temp buffer vs TurboPFor C's fused single-pass). These are fundamentally different problems requiring different solutions.

## Architecture Changes

### Decode: Refined Hybrid Dispatch

The current hybrid selects periodic-unroll when P > 2 and fully-unrolled when P ≤ 2 (i.e., B=0, B=16, B=32). The weakness pattern suggests:

- **b=31 (P=32, 1 iteration)**: The periodic wrapper's loop preamble/postamble adds overhead for a single iteration that's identical to full unroll. Fix: add P=32 to the fully-unrolled exception list.
- **b=2 (P=16, 2 iter), b=4 (P=8, 4 iter)**: Small periods with many iterations. The period body is very small (few instructions), so loop overhead is proportionally large. Fix: raise P threshold so these get fully-unrolled.
- **b=30 (P=16, 2 iter)**: Large period body (15 span crossings in 16 groups). Likely code alignment issue. Fix: try raising P threshold; if still weak, try `[[clang::minsize]]` or `__attribute__((aligned(64)))` on the period body.
- **b=32 (P=1, fully-unrolled)**: Trivial case — no masking, no shifts, no spans. The generic template still generates mask setup and dead code paths. Fix: dedicated hand-written fast path that's just load → D1 prefix scan → STO64.

The new threshold strategy: **use fully-unrolled for B where P ≤ 2 OR P = 32** (covers b=31), and **benchmark iteratively** with P thresholds of 8 and 16 to find the optimal cutoff for the remaining weak cases.

### Encode: Fused IP32 Bitpack

Replace the current two-pass encode:
```
IP32 shuffle → uint32_t tmp[128] → bitpack128v32(tmp, out, b)
```
with a fused single-pass that loads uint64_t pairs, shuffles inline, and feeds directly into the bitpack logic:
```
load 2×__m128i (4 uint64_t) → shuffle to __m128i (4 uint32_t) → pack into output
```

This requires creating a new template-based `bitpack128v64_fused<B>()` that combines the IP32 shuffle with the bitpack loop body. The `switch(b)` dispatch in `bitpack128v64()` selects the correct template instantiation.

For non-power-of-2 bit widths, we currently call `bitpack128v32_general()` which uses runtime `b` with variable shifts. TurboPFor C uses fully-unrolled per-bitwidth macros. If fusing IP32 alone doesn't reach +1%, we may also need to template-specialize the bitpack32 inner loop (either per-bitwidth or periodic-unroll, matching the decode approach).

## Implementation Steps

### Step 0: Baseline Measurement

**Files to modify/create**: None (benchmark only)

**Technical approach**: Run comprehensive benchmarks across all three codepaths and all bit widths to establish a fresh baseline. This must happen first because performance may have shifted since the last measurement.

Benchmark commands:
```bash
# D1 decode, no exceptions, b=1..64
./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-64

# D1 decode, with exceptions (10%), b=1..64
./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 1-64

# Encode + non-delta decode, b=1..64
./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 1-64
```

Record results in a markdown table as the reference point for all subsequent changes.

**Dependencies**: Build must succeed first.

### Step 1: D1 Decode — Threshold Tuning and b=32 Fast Path

**Files to modify**:
- `src/simd/bitpack128v64_simd.cpp` — Update `bitunpack_sse_sto64_d1_hybrid_entry` threshold logic, add b=32 dedicated fast path to `bitunpackD1_128v64()`
- `src/simd/bitunpack_sse_templates.h` — (possibly) Add dedicated B=32 D1+STO64 template

**Technical approach**:

1. **b=32 dedicated fast path**: In `bitunpackD1_128v64()`, before the `STO64_SWITCH`, add a special case for `b == 32` that avoids the template machinery entirely. For B=32, unpack is trivial: each group of 4 values is already a full __m128i, no masking or shifting needed. The fast path is just:
   ```
   for each group: load __m128i → D1 prefix scan → STO64 store
   ```
   This should be ~20 instructions total for 32 groups, vs the template generating unnecessary mask/shift logic.

2. **P=32 → fully-unrolled**: Change the threshold in `bitunpack_sse_sto64_d1_hybrid_entry` from `P <= 2` to `P <= 2 || P == 32`. This makes b=31 (and any odd B with P=32) use the fully-unrolled path, eliminating the periodic wrapper overhead for single-iteration loops.

3. **Benchmark and iterate**: Run D1 decode benchmark. If b=2, b=4, b=30 are still below +1%, try raising the threshold further:
   - Try `P <= 8` (adds b=4 to fully-unrolled, P=8)
   - Try `P <= 16` (adds b=2, b=6, b=10, b=14, b=30 to fully-unrolled, P=16)
   - Monitor total function size with `objdump -d` or `size` to ensure we don't recreate the original L1i pressure problem

**Dependencies**: Step 0 (baseline data)

### Step 2: D1+EX Decode — Apply Same Dispatch Changes

**Files to modify**:
- `src/simd/bitpack128v64_simd.cpp` — Update `bitunpack_sse_sto64_d1_ex_hybrid_entry` with same threshold as Step 1
- `src/simd/bitunpack_sse_templates.h` — Add B=32 D1+EX dedicated template if needed

**Technical approach**:

Apply the same P threshold change from Step 1 to the D1+EX hybrid entry. The D1+EX path has larger per-case code (adds SSSE3 exception shuffle per group), so the optimal threshold may differ. If the threshold from Step 1 causes L1i issues for D1+EX, use a different (lower) threshold.

Add a dedicated b=32 fast path for D1+EX if the generic template is slow: B=32 with exceptions means load __m128i → exception patch → D1 prefix scan → STO64.

Benchmark with `--exc-pct 10 --bw-range 1-32`.

**Dependencies**: Step 1 (decode threshold determined)

### Step 3: Encode — Fused IP32 Bitpack

**Files to modify**:
- `src/simd/bitpack128v64_simd.cpp` — Replace `bitpack128v64()` with fused version
- `src/simd/bitpack128v32_simd.cpp` — May need template-specialized pack functions callable from the fused encoder

**Technical approach**:

1. **Create fused bitpack128v64**: Replace the current implementation:
   ```cpp
   // Current: two passes
   alignas(16) uint32_t tmp[128];
   for (...) { IP32 shuffle → store to tmp }
   bitpack128v32(tmp, out, b);
   ```
   with a fused approach where the IP32 shuffle feeds directly into the pack loop:
   ```cpp
   // New: single pass, per-group inline shuffle
   template <unsigned B>
   unsigned char * bitpack128v64_fused(const uint64_t * in, unsigned char * out) {
       const __m128i * ip = reinterpret_cast<const __m128i *>(in);
       __m128i * op = reinterpret_cast<__m128i *>(out);
       __m128i ov = _mm_setzero_si128();
       unsigned shift = 0;
       
       for (unsigned g = 0; g < 32; ++g) {
           // IP32 inline: load 2 × __m128i, shuffle to 1 × __m128i
           __m128i lo = _mm_loadu_si128(ip++);
           __m128i hi = _mm_loadu_si128(ip++);
           __m128i iv = _mm_or_si128(
               _mm_shuffle_epi32(lo, _MM_SHUFFLE(2,0,3,1)),
               _mm_shuffle_epi32(hi, _MM_SHUFFLE(3,1,2,0)));
           
           // Bitpack: same logic as bitpack128v32_general but with compile-time B
           if constexpr (B < 32) iv = _mm_and_si128(iv, mask);
           // ... shift, OR, store logic with compile-time shifts
       }
   }
   ```

2. **Switch dispatch**: Add `STO64_SWITCH(b, CALL_FUSED_PACK)` in `bitpack128v64()` for b≤32.

3. **Decide template vs runtime B for bitpack inner loop**: TurboPFor C's `BITPACK128V32` is fully unrolled per bitwidth (each of the 33 cases has hand-written shifts). Our fused version uses runtime B with variable shifts (`_mm_slli_epi32(iv, shift)` where shift changes per group). This may be fast enough since the IP32 savings dominate, but if not, we may need periodic-unroll for the pack inner loop too (same mathematical insight as decode: the shift pattern repeats with period P).

4. **Benchmark**: Run `--simd128v64 --bw-range 1-32` to measure encode improvement.

**Dependencies**: Step 0 (baseline data). Independent of Steps 1-2.

### Step 4: Scalar Path Verification (b=33..64)

**Files to modify**: None expected (verification only)

**Technical approach**: Run benchmarks for b=33..64 across all three codepaths. The scalar path uses template-specialized `bitunpack64Scalar<B>()` and `bitpack64Scalar()` which are already significantly faster than TurboPFor C. If any bit width is below +1%, investigate and fix.

Benchmark commands:
```bash
./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 33-64
./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 33-64
./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 33-64
```

**Dependencies**: None (can run in parallel with Steps 1-3)

### Step 5: Final Comprehensive Verification

**Files to modify**: None (verification only)

**Technical approach**: Run all three benchmark suites across b=1..64 one final time to confirm every bit width meets ≥+1%. Run `binary_compat_test` and `vbyte64_test`. Document final results in a summary table.

```bash
cmake --build build --target binary_compat_test vbyte64_test -j$(nproc) && ./build/binary_compat_test && ./build/vbyte64_test
./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-64
./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 1-64
./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 1-64
```

**Dependencies**: All previous steps complete

## Testing Strategy

- **Correctness tests**: `binary_compat_test` (verifies binary format compatibility with TurboPFor C for encode/decode round-trip) and `vbyte64_test` (verifies vbyte64 encoding) must pass after every step.
- **Performance tests**: `ab_test` benchmarks with `--iters 300000 --runs 7` (best-of-7 runs of 300K iterations each). The benchmark uses interleaved 10K-iteration chunks to account for CPU frequency scaling.
- **Regression monitoring**: After each step, verify that no previously-passing bit widths have regressed below +1%. If a step improves some bit widths but regresses others, the step needs refinement before proceeding.
- **Code size monitoring**: Use `objdump -d build/path/to/object | wc -c` or `size` to track function sizes after decode threshold changes. The ~60KB hybrid function should not grow beyond ~70KB (2× L1i).

## Risks and Mitigations

- **Risk: P threshold that fixes b=2,4 causes new outliers in other bit widths** → Mitigation: Benchmark ALL bit widths after each threshold change, not just the targeted ones. Use `--bw-range 1-32` always.
- **Risk: Fused IP32 bitpack doesn't reach +1% because bitpack32 inner loop is also slow** → Mitigation: If fusing alone is insufficient, add periodic-unroll or per-bitwidth template specialization for the pack inner loop. The periodic-unroll technique from decode applies identically to encode.
- **Risk: D1+EX optimal threshold differs from D1, requiring two different dispatch strategies** → Mitigation: The hybrid entry templates already separate D1 and D1+EX selectors. They can use independent thresholds without code duplication.
- **Risk: b=30 (P=16, large period body) is alignment-sensitive and resists threshold tuning** → Mitigation: Try `__attribute__((aligned(64)))` on the period body function, or `[[clang::minsize]]` to reduce code size. Last resort: separate compilation unit for b=30's case.

## Rollout Considerations

- All changes are internal to the SIMD implementation files. No public API changes.
- Binary format compatibility is maintained (encode produces identical bytes).
- No feature flags needed — the optimizations are unconditional improvements.
- The encode refactor (fused IP32) is the most invasive change and should be reviewed carefully for correctness via `binary_compat_test`.
