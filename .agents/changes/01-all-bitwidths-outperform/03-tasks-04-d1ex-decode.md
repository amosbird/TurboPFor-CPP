# Task 4: D1+EX Decode — Apply Dispatch Optimizations and Benchmark

**Depends on**: Task 3 (D1 decode threshold finalized)
**Estimated complexity**: Medium
**Type**: Feature

## Objective

Apply the same hybrid dispatch optimizations from Tasks 2-3 to the D1+EX (delta-1 with bitmap exception) decode path, then benchmark to confirm ≥+1% across all b=1..32.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `src/simd/bitpack128v64_simd.cpp` — Update `bitunpack_sse_sto64_d1_ex_hybrid_entry` threshold, add b=32 fast path for D1+EX
- `src/simd/bitunpack_sse_templates.h` — (possibly) Adjust D1+EX periodic entry if different threshold needed

## Detailed Steps

1. Update `PROGRESS.md` to mark this task as In Progress.

2. **Apply the same P threshold** from Task 2-3 to D1+EX hybrid entry.

   In `bitunpack_sse_sto64_d1_ex_hybrid_entry` (`src/simd/bitpack128v64_simd.cpp`), update the threshold to match what was determined for D1:
   ```cpp
   // Match the D1 threshold (whatever was determined in Tasks 2-3)
   if constexpr (B == 0 || P <= THRESHOLD || P == 32)
       return bitunpack_sse_sto64_d1_ex_entry<B, Count>(in, out, sv, bitmap, pex);
   else
       return bitunpack_sse_sto64_d1_ex_periodic_entry<B, Count>(in, out, sv, bitmap, pex);
   ```

3. **Add b=32 fast path for D1+EX** in `bitd1unpack128v64_ex()`:

   Similar to the D1 fast path but with exception patching:
   ```cpp
   if (b == 32u)
   {
       __m128i sv_local = _mm_set1_epi32(static_cast<uint32_t>(start));
       __m128i * op = reinterpret_cast<__m128i *>(out);
       const __m128i cv = _mm_setr_epi32(1, 2, 3, 4);
       const __m128i zv = _mm_setzero_si128();
       const __m128i * ip = reinterpret_cast<const __m128i *>(in);
       
       for (unsigned g = 0; g < 32; ++g)
       {
           __m128i ov = _mm_loadu_si128(ip++);
           
           // Exception patching (same SSSE3 shuffle as template)
           uint64_t w = (g < 16) ? bitmap[0] : bitmap[1];
           unsigned shift = (g % 16) * 4;
           unsigned m = (w >> shift) & 0xF;
           __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
           __m128i exc_s = _mm_slli_epi32(exc, 32);  // B=32: shift left by 32 → zeros!
           // Wait — for B=32, exception shift is exc << 32 which is 0 for 32-bit values.
           // Actually for B=32, b+bx could exceed 32 and this path wouldn't be taken.
           // Check: this function is only called when b+bx <= 32.
           // If b=32, then bx must be 0, so there are NO exceptions.
           // So the b=32+EX fast path may never be reached. Verify in benchmark.
           
           // ... (same D1 + STO64 as D1 fast path)
       }
   }
   ```

   **IMPORTANT**: Check whether b=32 with exceptions is even possible. If b=32 and b+bx≤32, then bx=0, meaning no exceptions. The D1+EX path (`bitd1unpack128v64_ex`) is only called when bx>0. So the b=32 fast path in D1+EX **may not be needed**. Verify by checking `p4D1Dec128v64PayloadBitmap` — it's called when bx>0, and the fused path requires b+bx≤32. If b=32, this forces bx=0 which contradicts bx>0. So **skip the b=32 D1+EX fast path**.

4. **If D1 approach used Approach C (manual unroll for NumPeriods≤2)**, apply the same optimization to `bitunpack_sse_sto64_d1_ex_periodic_entry` in `bitunpack_sse_templates.h`.

5. Build and test:
   ```bash
   cmake --build build --target ab_test binary_compat_test vbyte64_test -j$(nproc)
   ./build/binary_compat_test && ./build/vbyte64_test
   ```

6. Benchmark D1+EX decode:
   ```bash
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 1-32
   ```

7. **Evaluate results**:
   - If all b=1..32 show ≥+1%: success.
   - If some bit widths are below +1% and they're different from the D1 weak cases: the D1+EX path has different code size characteristics (each case is larger due to exception handling). May need a different (lower) P threshold for D1+EX. Try adjusting independently.
   - If the same bit widths are weak: the shared approach works, but the per-case code is just larger. Try the approaches from Task 3 (Approach C or D) specifically for D1+EX.

8. Also benchmark D1 decode again to make sure D1+EX changes didn't affect D1:
   ```bash
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-32
   ```
   (D1 and D1+EX are in the same compilation unit but different functions, so they shouldn't interfere — but verify.)

9. Update `PROGRESS.md` with results.

10. Commit: `perf(simd128v64): optimize D1+EX decode dispatch to match D1 threshold`

## Acceptance Criteria
- [ ] D1+EX decode shows ≥+1% for every b=1..32 with 10% exception rate
- [ ] D1 decode (no exceptions) still shows ≥+1% for every b=1..32 (no regression)
- [ ] `binary_compat_test` and `vbyte64_test` pass
- [ ] Results recorded in `PROGRESS.md`

## Testing
- **Correctness**: `./build/binary_compat_test && ./build/vbyte64_test`
- **D1+EX performance**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 1-32`
- **D1 regression check**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-32`

## Notes
- The D1+EX code is structurally identical to D1 but with an extra SSSE3 shuffle + `popcount` per group for exception patching. This adds ~30-40 bytes per group to the code. For the fully-unrolled path, each case is ~3.5KB instead of ~2.7KB.
- The D1+EX and D1 hybrid entries are separate template functions, so they CAN have different P thresholds. If needed, use a different threshold for D1+EX.
- For b=32 with exceptions: as analyzed above, the fused SIMD path (b+bx ≤ 32) requires bx=0 when b=32, which means no exceptions. The b=32+EX case falls through to the scalar multi-phase path in `p4D1Dec128v64PayloadBitmap`. No optimization needed for b=32+EX in the SIMD path.
