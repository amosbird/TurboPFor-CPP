# Task 03: Add/Adjust 256v64 Tests and Benchmark Mode

**Depends on**: Task 02
**Estimated complexity**: Medium
**Type**: Testing

## Objective
Ensure 256v64 behavior is validated in compatibility tests and benchmarked against scalar64 baseline.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `tests/binary_compat_test.cpp`
- `benchmarks/ab_test.cpp`

## Detailed Steps
1. Update `PROGRESS.md` to mark this task as In Progress.
2. Add/adjust 256v64 test coverage with semantically correct assertions.
3. Add/verify `--simd256v64d1` benchmark mode.
4. Run tests and benchmark sanity checks.
5. Update `PROGRESS.md` to mark this task as Completed.
6. Commit changes.

## Acceptance Criteria
- [ ] New/updated 256v64 tests pass.
- [ ] Benchmark mode executes and reports expected comparison.
- [ ] No regressions in existing benchmark modes.

## Testing
- **Test file**: `tests/binary_compat_test.cpp`
- **Test cases**: 256v64 roundtrip paths and dispatch-routed behavior.
