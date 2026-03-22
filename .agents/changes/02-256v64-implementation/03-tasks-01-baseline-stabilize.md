# Task 01: Stabilize 256v64 Scalar Correctness Baseline

**Depends on**: None
**Estimated complexity**: Medium
**Type**: Feature

## Objective
Ensure scalar 256v64 implementation is complete and semantically correct for encode/decode paths,
forming a stable baseline for API exposure and benchmarking.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `src/scalar/bitpack256v64_scalar.cpp`
- `src/scalar/p4enc256v64_scalar.cpp`
- `src/scalar/p4d1dec256v64_scalar.cpp`
- `src/scalar/p4_scalar.h`
- `src/scalar/p4_scalar_internal.h`
- `src/scalar/p4_scalar_internal.cpp`

## Detailed Steps
1. Update `PROGRESS.md` to mark this task as In Progress
2. Verify scalar 256v64 functions compile and handle expected bitwidth/delta behavior.
3. Validate helper glue (including delta application helpers) and function signatures.
4. Run targeted tests/build checks relevant to scalar correctness.
5. Update `PROGRESS.md` to mark this task as Completed.
6. Commit changes.

## Acceptance Criteria
- [ ] Scalar 256v64 sources build cleanly.
- [ ] Delta/non-delta behavior is semantically correct.
- [ ] No regressions introduced in existing scalar interfaces.

## Testing
- **Test file**: `tests/binary_compat_test.cpp`
- **Test cases**: existing compat suites touching v64 paths.

## Notes
If unresolved semantic ambiguity appears, document it in `PROGRESS.md` notes before moving on.
