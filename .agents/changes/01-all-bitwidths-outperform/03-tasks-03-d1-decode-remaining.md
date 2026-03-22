# Task 3: D1 Decode — Fix Remaining Weak Bit Widths (b=2, b=4, b=30)

**Depends on**: Task 2 (b=31, b=32 fixed; baseline P threshold established)
**Estimated complexity**: High
**Type**: Feature

## Objective

Fix the remaining D1 decode weak cases: b=2 (-0.2%), b=4 (-0.9%), b=30 (-1.2%). These all use the periodic-unroll path but with different period characteristics. The approach is iterative: try P threshold adjustments, measure, and refine.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `src/simd/bitpack128v64_simd.cpp` — Update hybrid dispatch threshold
- `src/simd/bitunpack_sse_templates.h` — Potentially optimize periodic entry point

## Detailed Steps

1. Update `PROGRESS.md` to mark this task as In Progress.

2. **Understand the weak cases**:
   - **b=2 (P=16, 2 iterations)**: Each period body processes 16 groups = 64 elements. Period body is relatively small (no span crossings for B=2). 2 iterations of a small body — loop overhead may be significant relative to body work.
   - **b=4 (P=8, 4 iterations)**: Each period body processes 8 groups = 32 elements. 4 iterations. Same issue — small body, many iterations.
   - **b=30 (P=16, 2 iterations)**: Each period body processes 16 groups but with 15 span crossings (offset+30 > 32 for most groups). Large body (~500+ bytes). 2 iterations. Possibly code alignment issue.

3. **Approach A — Raise P threshold to fully-unroll these cases**:

   In `bitunpack_sse_sto64_d1_hybrid_entry`, try:
   ```cpp
   if constexpr (B == 0 || P <= 16 || P == 32)
   ```
   This makes B=2(P=16), B=6(P=16), B=10(P=16), B=14(P=16), B=30(P=16) fully-unrolled, and also B=4(P=8), B=12(P=8), B=20(P=8), B=28(P=8).

   **Code size concern**: Each fully-unrolled case is ~2.7KB. Adding ~9 more cases = ~24KB additional. Check if total function size stays reasonable.

   Build and benchmark:
   ```bash
   cmake --build build --target ab_test -j$(nproc)
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-32
   ```

   **If this works** (all b=1..32 ≥+1% and no regressions): great, commit and move on.

   **If this causes regressions** (L1i pressure from large function): try Approach B.

4. **Approach B — Try P ≤ 8 instead of P ≤ 16**:
   ```cpp
   if constexpr (B == 0 || P <= 8 || P == 32)
   ```
   This adds B=4(P=8), B=12(P=8), B=20(P=8), B=28(P=8) to fully-unrolled. Fewer cases than P≤16, smaller total function. Benchmark and see if b=2 and b=30 (still periodic) improved from the overall smaller function size.

5. **Approach C — Optimize the periodic loop entry for NumPeriods==2**:

   If threshold-based approaches cause regressions, optimize the periodic entry point itself. In `bitunpack_sse_sto64_d1_periodic_entry` (in `bitunpack_sse_templates.h`), add:
   ```cpp
   constexpr unsigned NumPeriods = MaxG / P;
   if constexpr (NumPeriods <= 2)
   {
       // Manually unroll: call period body twice (or once), no loop
       bitunpack_sse_sto64_d1_period_body<B>(ip, iv, op, mask, cv, sv, zv);
       if constexpr (NumPeriods == 2)
           bitunpack_sse_sto64_d1_period_body<B>(ip, iv, op, mask, cv, sv, zv);
   }
   else
   {
       for (unsigned period = 0; period < NumPeriods; ++period)
           bitunpack_sse_sto64_d1_period_body<B>(ip, iv, op, mask, cv, sv, zv);
   }
   ```
   This eliminates loop overhead for b=2(2 iter) and b=30(2 iter) without increasing the switch case count.

6. **Approach D — `[[clang::minsize]]` on period body** (last resort for b=30):

   If b=30 specifically resists optimization, try annotating the period body with size-optimization hints:
   ```cpp
   template <unsigned B>
   [[clang::minsize]] ALWAYS_INLINE void bitunpack_sse_sto64_d1_period_body(...)
   ```
   This may cause clang to choose smaller instruction encodings that improve code alignment.

7. **Iterate until all b=1..32 show ≥+1%**. After each approach, benchmark ALL bit widths (not just the targeted ones) to check for regressions.

8. Run correctness tests:
   ```bash
   ./build/binary_compat_test && ./build/vbyte64_test
   ```

9. Update `PROGRESS.md` with final results.

10. Commit with message describing which approach(es) were used: `perf(simd128v64): optimize D1 decode for b=2,4,30 via [approach]`

## Acceptance Criteria
- [ ] b=2 D1 decode shows ≥+1% improvement
- [ ] b=4 D1 decode shows ≥+1% improvement
- [ ] b=30 D1 decode shows ≥+1% improvement
- [ ] No previously-passing bit width regressed below +1%
- [ ] `binary_compat_test` and `vbyte64_test` pass
- [ ] Results recorded in `PROGRESS.md`

## Testing
- **Correctness**: `./build/binary_compat_test && ./build/vbyte64_test`
- **Performance**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-32`
- **Quick single-bw check**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --bw 2` (and --bw 4, --bw 30)

## Notes
- The approaches are ordered by likelihood of success and invasiveness. Try A first, then B, then C, then D.
- If Approach A (P≤16) works, it's the simplest and most maintainable solution — just one threshold change.
- b=28 was the original L1i outlier (-11%) that motivated the periodic-unroll approach. If raising the P threshold causes b=28 to regress, that's a clear sign of L1i pressure. b=28 has P=8, so P≤8 would make it fully-unrolled — watch it carefully.
- Approach C (manual unroll for NumPeriods≤2) is the most surgical fix. It doesn't change which cases are periodic vs fully-unrolled; it just eliminates the loop overhead for 2-iteration cases.
- The key metric is the `objdump -d` size of `bitunpackD1_128v64`. Current is ~60KB. If it grows beyond ~80KB, L1i pressure is likely.
