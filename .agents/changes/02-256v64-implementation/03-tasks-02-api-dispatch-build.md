# Task 02: Wire Public API, Dispatch, and Build for 256v64

**Depends on**: Task 01
**Estimated complexity**: Medium
**Type**: Feature

## Objective
Expose 256v64 through public API and dispatch, and ensure build system includes required sources.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Files to Modify/Create
- `include/turbopfor.h`
- `src/dispatch.cpp`
- `CMakeLists.txt`

## Detailed Steps
1. Update `PROGRESS.md` to mark this task as In Progress.
2. Add/verify public declarations for `p4Enc256v64`, `p4Dec256v64`, `p4D1Dec256v64`.
3. Add/verify dispatch routing to selected backend.
4. Ensure all relevant 256v64 files are included in build.
5. Validate compile/link.
6. Update `PROGRESS.md` to mark this task as Completed.
7. Commit changes.

## Acceptance Criteria
- [ ] Public API symbols exist and compile.
- [ ] Dispatch routes correctly.
- [ ] CMake includes required files without duplicates/conflicts.

## Testing
- **Test file**: `tests/binary_compat_test.cpp`
- **Test cases**: API reachability and roundtrip through top-level calls.
