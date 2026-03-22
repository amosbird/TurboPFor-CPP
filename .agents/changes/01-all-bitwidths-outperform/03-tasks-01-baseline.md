# Task 1: Baseline Measurement

**Depends on**: None
**Estimated complexity**: Low
**Type**: Testing

## Objective

Run comprehensive benchmarks across all three codepaths (D1 decode, D1+EX decode, encode) and all bit widths (b=1..64) to establish a fresh performance baseline before any optimization work begins.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `.agents/changes/01-all-bitwidths-outperform/PROGRESS.md` — Create/update with benchmark results

## Detailed Steps

1. Create `PROGRESS.md` in `.agents/changes/01-all-bitwidths-outperform/` and mark this task as In Progress.

2. Build all targets:
   ```bash
   cmake --build build --target ab_test binary_compat_test vbyte64_test -j$(nproc)
   ```

3. Verify correctness tests pass:
   ```bash
   ./build/binary_compat_test && ./build/vbyte64_test
   ```

4. Run D1 decode benchmark (no exceptions), b=1..64:
   ```bash
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-64
   ```
   Record the full output (both encode and decode columns).

5. Run D1+EX decode benchmark (10% exceptions), b=1..64:
   ```bash
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 1-64
   ```
   Record the full output.

6. Run encode + non-delta decode benchmark, b=1..64:
   ```bash
   ./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 1-64
   ```
   Record the full output.

7. Compile results into `PROGRESS.md` with three markdown tables:
   - **Table 1: D1 Decode (no exceptions)** — columns: BitWidth, Encode Diff%, Decode Diff%, Status (✅ ≥+1%, ⚠️ 0-1%, ❌ <0%)
   - **Table 2: D1+EX Decode (10% exceptions)** — same columns
   - **Table 3: Encode (from --simd128v64 run)** — columns: BitWidth, Encode Diff%, Status

   Mark bit widths below +1% with ❌ and add a summary count: "X of 64 bit widths below +1%".

8. Update `PROGRESS.md` to mark this task as Completed.

9. Commit with message: `perf(simd128v64): baseline benchmark measurement for all-bitwidths optimization`

## Acceptance Criteria
- [ ] All three benchmark suites (D1, D1+EX, Encode) have been run for b=1..64
- [ ] Results are recorded in `PROGRESS.md` with clear tables
- [ ] Bit widths below +1% are clearly identified for each codepath
- [ ] `binary_compat_test` and `vbyte64_test` pass
- [ ] A summary count of failing bit widths per codepath is included

## Testing
- **Correctness**: `./build/binary_compat_test && ./build/vbyte64_test`
- **Benchmarks**: Three `ab_test` runs as described above

## Notes
- Each benchmark run (300K iters × 7 runs × 64 bit widths) takes several minutes. Be patient.
- The benchmark output includes both encode and decode columns. For the `--simd128v64d1` run, the encode column is also relevant (it runs the same encode path).
- If any benchmark shows anomalous results (e.g., >50% diff in either direction), re-run that specific bit width with `--bw B` to confirm.
- The scalar path (b>32) should show +30-40% consistently. If not, flag it as unusual.
