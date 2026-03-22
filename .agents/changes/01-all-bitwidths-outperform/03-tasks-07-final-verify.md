# Task 7: Final Comprehensive Verification and Commit Message

**Depends on**: Tasks 2, 3, 4, 5, 6 (ALL optimization and verification tasks complete)
**Estimated complexity**: Low
**Type**: Testing / Documentation

## Objective

Run the complete benchmark suite across all three codepaths (D1 decode, D1+EX decode, encode) and all bit widths (b=1..64) to confirm every single bit width meets the ≥+1% target. Run correctness tests. Generate the final summary and commit message.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create

- `.agents/changes/01-all-bitwidths-outperform/PROGRESS.md` — Final results
- `.agents/changes/01-all-bitwidths-outperform/04-commit-msg.md` — Squash commit message

## Detailed Steps

1. Update `PROGRESS.md` to mark this task as In Progress.

2. **Build everything**:
   ```bash
   cmake --build build --target ab_test binary_compat_test vbyte64_test -j$(nproc)
   ```

3. **Run correctness tests**:
   ```bash
   ./build/binary_compat_test && ./build/vbyte64_test
   ```
   Both MUST pass. If either fails, DO NOT proceed — go back and fix.

4. **Run full D1 decode benchmark (no exceptions), b=1..64**:
   ```bash
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-64
   ```
   Record full output.

5. **Run full D1+EX decode benchmark (10% exceptions), b=1..64**:
   ```bash
   ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 1-64
   ```
   Record full output.

6. **Run full encode benchmark, b=1..64**:
   ```bash
   ./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 1-64
   ```
   Record full output.

7. **Compile final results table** in `PROGRESS.md`:

   Create a comprehensive summary table with ALL bit widths:
   ```markdown
   ## Final Results

   | BitWidth | D1 Decode | D1+EX Decode | Encode | Status |
   |----------|-----------|--------------|--------|--------|
   | 1        | +X.X%     | +X.X%        | +X.X%  | ✅/❌  |
   | 2        | +X.X%     | +X.X%        | +X.X%  | ✅/❌  |
   | ...      | ...       | ...          | ...    | ...    |
   | 64       | +X.X%     | +X.X%        | +X.X%  | ✅/❌  |

   ### Summary
   - D1 Decode: X/64 bit widths ≥+1% (min: +X.X% at b=Y, avg: +X.X%)
   - D1+EX Decode: X/64 bit widths ≥+1% (min: +X.X% at b=Y, avg: +X.X%)
   - Encode: X/64 bit widths ≥+1% (min: +X.X% at b=Y, avg: +X.X%)
   - Overall: ALL PASS / X failures
   ```

   Mark status as ✅ if ALL THREE codepaths show ≥+1% for that bit width, ❌ otherwise.

8. **If any bit width fails** (ANY codepath below +1%):
   - Re-run that specific bit width 3 times to confirm it's not noise:
     ```bash
     ./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --bw B
     ```
   - If confirmed below +1%: document in PROGRESS.md and note that this needs further investigation. Do NOT mark the overall task as complete — mark it as BLOCKED and describe what's failing.
   - If it's borderline (e.g., +0.8% to +1.2% across runs): document the variance and note it's noise-sensitive. Consider it a pass if the median of 3 runs is ≥+1%.

9. **Generate `04-commit-msg.md`**:

   Write the squash commit message following conventional commit format:
   ```markdown
   perf(simd128v64): outperform TurboPFor C by ≥+1% on all bit widths b=1..64

   Optimize the 128v64 SIMD codec so every bit width achieves at least +1%
   improvement over the TurboPFor C reference across D1 decode, D1+EX decode,
   and encode codepaths.

   Key changes:
   - Refined hybrid dispatch threshold for D1/D1+EX decode (periodic vs fully-unrolled)
   - Dedicated b=32 fast path for D1 decode
   - Fused IP32 shuffle into bitpack for single-pass encode
   - [Any other changes made during implementation]

   Average improvements:
   - D1 Decode: +X.X% (min +X.X% at b=Y)
   - D1+EX Decode: +X.X% (min +X.X% at b=Y)
   - Encode: +X.X% (min +X.X% at b=Y)
   ```

   Fill in the actual numbers from the benchmark results.

10. Update `PROGRESS.md` to mark this task as Completed.

11. Commit all changes with the message from `04-commit-msg.md`.

## Acceptance Criteria
- [ ] `binary_compat_test` and `vbyte64_test` pass
- [ ] D1 decode shows ≥+1% for EVERY b=1..64
- [ ] D1+EX decode shows ≥+1% for EVERY b=1..64 (10% exceptions)
- [ ] Encode shows ≥+1% for EVERY b=1..64
- [ ] Final results table in `PROGRESS.md` with all 64 × 3 = 192 measurements
- [ ] `04-commit-msg.md` generated with actual benchmark numbers
- [ ] All changes committed

## Testing
- **Correctness**: `./build/binary_compat_test && ./build/vbyte64_test`
- **D1 Decode**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 0 --bw-range 1-64`
- **D1+EX Decode**: `./build/ab_test --simd128v64d1 --iters 300000 --runs 7 --exc-pct 10 --bw-range 1-64`
- **Encode**: `./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 1-64`

## Notes
- This task should be the very last task executed. It's the final gate before declaring the change request complete.
- Each full benchmark run (64 bit widths × 300K iterations × 7 runs) takes several minutes. Budget ~15 minutes for all three suites.
- If the overall results are good but 1-2 bit widths are marginal (+0.5% to +1.0%), discuss with the user whether to accept or continue iterating.
- The commit message should be factual about what was achieved. If some bit widths are borderline, say so.
- Remember: the `--simd128v64` benchmark (encode) also reports a decode column — that's the non-D1 plain decode. It's not one of our three target codepaths but record it anyway for completeness.
