# Task 07: Optimize 256v64 Decode to Beat Scalar64 Baseline Across Scenarios

**Depends on**: Task 06
**Estimated complexity**: High
**Type**: Feature

## Objective
Optimize 256v64 delta decode so it meets the new target: decode throughput is >= scalar64 baseline
across key scenarios (random, 10%, 30%, 50%, 80% exceptions) for core bitwidths.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `src/simd/p4d1dec256v64.cpp`
- `src/simd/p4d1dec128v64.cpp`
- `src/simd/p4_simd_internal_128v.h`
- `src/simd/p4_simd_internal_256v.h`
- `src/simd/p4_simd_internal_256v.cpp`
- `benchmarks/ab_test.cpp` (only if measurement/reporting needs small support)

## Detailed Steps
1. Update `PROGRESS.md` to mark this task as In Progress.
2. Profile/identify decode hotspots for 256v64 random and exception scenarios.
3. Apply SIMD-focused optimizations while preserving exact decode semantics.
4. Validate with full compatibility tests after each meaningful optimization.
5. Benchmark with `--simd256v64d1` using representative bitwidths (1,8,16,32,48,60).
6. Iterate until target is met or a hard blocker is proven with evidence.
7. Update `PROGRESS.md` with results table and final status.
8. Commit changes.

## Acceptance Criteria
- [ ] Decode >= scalar64 baseline for random/10/30/50/80% scenarios on target bitwidth set.
- [ ] Compatibility tests remain fully passing.
- [ ] Results and rationale documented in PROGRESS.

## Testing
- **Test file**: `tests/binary_compat_test.cpp`
- **Benchmark**: `./build/ab_test --simd256v64d1 --count 200000 --reps 5 --bw <bw>`
