# Task 04: SIMD 256v64 Gate or Optimize

**Depends on**: Task 03
**Estimated complexity**: High
**Type**: Feature

## Objective
Decide and implement the safe backend strategy: either keep SIMD gated off (scalar default) or fix
SIMD semantics/performance sufficiently to enable it.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `src/simd/p4enc256v64.cpp`
- `src/simd/p4dec256v64.cpp`
- `src/simd/p4d1dec256v64.cpp`
- `src/simd/p4_simd.h`
- `src/dispatch.cpp`

## Detailed Steps
1. Update `PROGRESS.md` to mark this task as In Progress.
2. Validate SIMD wrapper semantics vs scalar/reference expectations.
3. Benchmark SIMD path in `--simd256v64d1` and compare against scalar64 baseline.
4. If SIMD is not both correct and favorable, keep scalar route as default and document rationale.
5. Update `PROGRESS.md` to mark this task as Completed.
6. Commit changes.

## Acceptance Criteria
- [ ] Selected backend is correctness-safe.
- [ ] Decision (enable/gate) is benchmark-justified and documented.
- [ ] Dispatch reflects the chosen strategy.

## Testing
- **Test file**: `tests/binary_compat_test.cpp`
- **Test cases**: scalar vs dispatch behavior parity for 256v64.
