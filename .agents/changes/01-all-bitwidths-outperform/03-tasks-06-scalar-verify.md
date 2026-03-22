# Task 6: Scalar Path Verification (b=33..64)

**Depends on**: Task 1 (baseline data)
**Estimated complexity**: Low
**Type**: Testing

## Objective

Verify that the scalar path (b=33..64) achieves ≥+1% across all three codepaths (D1 decode, D1+EX decode, encode). The scalar path uses template-specialized `bitunpack64Scalar<B>()` which is already measured at +32-42% faster than TurboPFor C. This task is primarily verification, not optimization — but includes a fix plan if any outliers are found.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Background

For b > 32, the 128v64 codec falls back to scalar paths:
- **Decode**: `bitunpack64Scalar()` — template-specialized per B, each B gets its own compiled function with compile-time shift/mask constants. Called from `bitunpackD1_128v64()` (line 195 in `bitpack128v64_simd.cpp`) and `bitunpack128v64()` (line 144).
- **D1 decode**: After scalar unpack, delta-1 is applied as a separate scalar loop: `out[i] = (start += out[i]) + (i + 1)` (line 198-199).
- **D1+EX decode**: For b > 32 with exceptions, the multi-phase scalar path in `p4D1Dec128v64PayloadBitmap` handles both unpack + exception merge + delta.
- **Encode**: `bitpack64Scalar()` — also template-specialized.

These should all be +30-42% faster because TurboPFor C's `bitunpack64`/`bitpack64` are single generic functions with runtime B, while ours use per-B compiled specializations with compile-time constants.

## Files to Modify/Create

- `.agents/changes/01-all-bitwidths-outperform/PROGRESS.md` — Record results
- (Possibly) `src/scalar/p4_scalar_bitunpack64_impl.h` or related — only if a fix is needed

## Detailed Steps

1. Update `PROGRESS.md` to mark this task as In Progress.

2. **Benchmark D1 decode, b=33..64, no exceptions**:
   ```bash
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 33-64
   ```
   Record all bit widths. Expected: +30-42% across the board.

3. **Benchmark D1+EX decode, b=33..64, 10% exceptions**:
   ```bash
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 33-64
   ```
   Note: For b > 32, the exception handling follows a different code path (scalar multi-phase) than the SIMD fused path. The exception bits bx are determined at encode time. When b is large (e.g., b=60), bx tends to be small (e.g., 1-4 bits), so the exception path is lightweight.

4. **Benchmark encode, b=33..64**:
   ```bash
   ./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 33-64
   ```

5. **Evaluate results**:
   - If ALL bit widths across all three codepaths show ≥+1%: record in PROGRESS.md and commit. Task is done.
   - If any bit width is below +1%: investigate. Possible causes:
     - **b=33**: Close to the SIMD/scalar boundary. The SIMD path handles b=32, scalar handles b=33. If b=33 is slow, check whether the dispatch introduces overhead.
     - **b=64**: Special case — full 64-bit values, no masking needed. Check if our template has unnecessary mask/shift operations for B=64.
     - **Anomalous result**: If a single b is slow, re-run it 3 times to confirm it's not noise: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --bw B`

6. **If fixes are needed** (unlikely based on prior measurements):
   - For b=64: add a dedicated fast path that's just `memcpy` for decode (values are already full-width) and raw stores for encode.
   - For b=33: check if there's dispatch overhead at the SIMD/scalar boundary. The `bitunpackD1_128v64()` function checks `if (b <= 32u)` — this should be fast.
   - For any other b: the template specialization should already be optimal. Check `objdump -d` for the specific B to see if the compiler generated suboptimal code.

7. Update `PROGRESS.md` with results.

8. Commit: `perf(simd128v64): verify scalar path b=33..64 outperforms TurboPFor C`

## Acceptance Criteria
- [ ] D1 decode shows ≥+1% for every b=33..64
- [ ] D1+EX decode shows ≥+1% for every b=33..64 (with 10% exceptions)
- [ ] Encode shows ≥+1% for every b=33..64
- [ ] `binary_compat_test` and `vbyte64_test` pass (sanity check)
- [ ] Results recorded in `PROGRESS.md`

## Testing
- **D1 decode**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 33-64`
- **D1+EX decode**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 33-64`
- **Encode**: `./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 33-64`

## Notes
- This task can run in parallel with Tasks 2-4 (SIMD decode optimization) since it touches different code paths.
- The scalar path was previously measured at +32-42% — this is a very comfortable margin. Unless something changed in the build system or compiler flags, all bit widths should easily exceed +1%.
- The D1+EX path for b > 32 may show different margins than D1-only because the exception handling adds overhead on both our side and TurboPFor C's side. But since our template-specialized unpack is the dominant cost, the margin should still be large.
- If you find b=63 behaves oddly: remember that the TurboPFor header format uses 6 bits for b, so b=63 is stored as 63 but decoded back to 63 (not 64). Double-check that the scalar template handles B=63 correctly (mask should be `(1ULL << 63) - 1`).
