# Task 06: Fix SIMD 64-bit Decode Correctness (Ungate Candidate)

**Depends on**: Task 05
**Estimated complexity**: High
**Type**: Feature

## Objective
Fix known SIMD 64-bit decode correctness bugs (including STO64 pair ordering), then validate whether
64-bit decode can be safely ungated from scalar fallback.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `src/simd/p4_simd_internal_128v.h`
- `src/simd/p4_simd_internal_128v.cpp`
- `src/simd/p4d1dec128v64.cpp`
- `src/simd/p4dec128v64.cpp`
- `src/dispatch.cpp`
- `tests/binary_compat_test.cpp` (if needed for targeted assertions)

## Detailed Steps
1. Update `PROGRESS.md` to mark this task as In Progress.
2. Reproduce and root-cause STO64 pair-swap / lane-ordering issues in SIMD decode paths.
3. Implement correctness fix for non-delta and delta decode in SIMD 64-bit path.
4. Keep/adjust large-start fallback logic so 64-bit start values remain correct.
5. Run full build/tests to confirm 0 regressions.
6. If correctness is proven, enable SIMD decode dispatch for relevant 64-bit paths.
7. Update `PROGRESS.md` to Completed with detailed notes.
8. Commit changes.

## Acceptance Criteria
- [ ] SIMD 64-bit decode passes all compatibility tests.
- [ ] No 64-bit ordering/overflow regressions remain.
- [ ] Dispatch uses correctness-safe routing (SIMD only if proven).

## Testing
- **Test file**: `tests/binary_compat_test.cpp`
- **Test cases**: 128v64/256v64 roundtrip + delta/non-delta + non-zero start values.
