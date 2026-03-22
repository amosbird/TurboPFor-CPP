# Task 2: D1 Decode — b=32 Dedicated Fast Path and P=32 Fully-Unrolled

**Depends on**: Task 1 (baseline data)
**Estimated complexity**: Medium
**Type**: Feature

## Objective

Fix the two easiest decode weak cases: b=32 (currently +0.1%) with a dedicated fast path, and b=31 (currently -0.4%) by routing P=32 bit widths to the fully-unrolled path instead of periodic-unroll.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `src/simd/bitpack128v64_simd.cpp` — Update hybrid dispatch threshold, add b=32 fast path
- `src/simd/bitunpack_sse_templates.h` — (possibly) Add dedicated B=32 D1+STO64 fast path template

## Detailed Steps

1. Update `PROGRESS.md` to mark this task as In Progress.

2. **Add b=32 dedicated fast path** in `bitunpackD1_128v64()` (`src/simd/bitpack128v64_simd.cpp`):

   Before the `STO64_SWITCH(b, CALL_STO64_D1)` dispatch, add a special case:
   ```cpp
   if (b == 32u)
   {
       // B=32 fast path: no masking, no shifting, no span crossings.
       // Each group of 4 values is a complete __m128i.
       const __m128i * ip = reinterpret_cast<const __m128i *>(in);
       __m128i * op = reinterpret_cast<__m128i *>(out);
       const __m128i cv = _mm_setr_epi32(1, 2, 3, 4);
       const __m128i zv = _mm_setzero_si128();
       __m128i sv = _mm_set1_epi32(static_cast<uint32_t>(start));
       
       for (unsigned g = 0; g < 32; ++g)
       {
           __m128i ov = _mm_loadu_si128(ip++);
           // Delta1 prefix scan
           ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
           ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
           ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));
           // STO64
           _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
           sv = _mm_shuffle_epi32(ov, 0xFF);
           _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));
       }
       return const_cast<unsigned char *>(ip_end);  // ip_end = in + (128*32+7)/8
   }
   ```
   This eliminates mask setup, the template instantiation overhead, and any dead code paths for B=32.

3. **Route P=32 to fully-unrolled**: In `bitunpack_sse_sto64_d1_hybrid_entry` (same file), change the threshold:
   ```cpp
   // Before:
   if constexpr (B == 0 || P <= 2)
   
   // After:
   if constexpr (B == 0 || P <= 2 || P == 32)
   ```
   This makes b=31 (and all odd B values where P=32, i.e. B=1,3,5,...,31) use the fully-unrolled path. For odd B, P=32 means the periodic version does 1 iteration of the full 32-group body — identical to fully-unrolled but with loop preamble/postamble overhead. The fully-unrolled version eliminates this overhead.

   **Important**: B=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29 already have P=32 and currently use periodic (1 iteration). This change makes ALL of them fully-unrolled. Since these are odd bit widths with no span crossings at power-of-2 boundaries, their fully-unrolled code size is moderate (~1-2KB each). Total added code: ~15 × 2KB = ~30KB. Combined with existing fully-unrolled cases (B=0,16,32), total function size estimate: ~60KB → ~90KB. This may trigger L1i pressure. **Monitor carefully with benchmarks.**

4. Build and run correctness tests:
   ```bash
   cmake --build build --target ab_test binary_compat_test vbyte64_test -j$(nproc)
   ./build/binary_compat_test && ./build/vbyte64_test
   ```

5. Benchmark D1 decode across all b=1..32:
   ```bash
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-32
   ```

6. **Evaluate results**:
   - If b=31 and b=32 are now ≥+1%: success for those two.
   - If ANY previously-good bit width (especially odd ones like b=1,3,5,...,29) regressed below +1%: the P=32→fully-unrolled change caused L1i pressure. In that case, revert the P=32 change and instead only add b=31 to the fully-unrolled list explicitly (e.g., `P <= 2 || B == 31`).
   - Record results in `PROGRESS.md`.

7. Update `PROGRESS.md` to mark this task as Completed.

8. Commit with message: `perf(simd128v64): add b=32 D1 decode fast path and route P=32 to fully-unrolled`

## Acceptance Criteria
- [ ] b=32 D1 decode shows ≥+1% improvement
- [ ] b=31 D1 decode shows ≥+1% improvement
- [ ] No previously-passing bit width (b=1..30) regressed below +1%
- [ ] `binary_compat_test` and `vbyte64_test` pass
- [ ] Results recorded in `PROGRESS.md`

## Testing
- **Correctness**: `./build/binary_compat_test && ./build/vbyte64_test`
- **Performance**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-32`

## Notes
- The b=32 fast path is essentially a hand-written loop — it bypasses the entire template machinery. This is justified because B=32 is a degenerate case (no masking, no shifting) where the template generates unnecessary code.
- The P=32 change affects all odd bit widths. Watch for regressions in the odd B range. If the total function grows too large, the alternative is to only add `B == 31` to the exception list, not all P=32 cases.
- If P=32→fully-unrolled causes problems, consider the alternative: keep periodic for P=32 but optimize the periodic entry point to skip loop setup when NumPeriods==1 (a `if constexpr (NumPeriods == 1)` branch that calls the period body directly without the `for` loop).
