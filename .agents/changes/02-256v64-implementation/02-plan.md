# Implementation Plan: 256v64 Implementation and Optimization

## Overview
Use a phased approach: (1) lock correctness and API wiring with a scalar baseline, (2) validate via
tests and benchmark harness, (3) iterate on SIMD/perf path only after semantics are stable.

## Architecture Changes
- Introduce dedicated 256v64 scalar codec units analogous to existing scalar modules.
- Add 256v64 SIMD-facing entry points; initial form may wrap existing 128v64 building blocks.
- Extend dispatch/API wiring for new public symbols.
- Extend benchmark/test coverage for 256v64 scenarios.

## Implementation Steps

### Step 1: PRD + Progress Bootstrap
**Files to modify/create**:
- `.agents/changes/02-256v64-implementation/PROGRESS.md`
- `.agents/changes/02-256v64-implementation/03-tasks-*`

**Technical approach**:
Create task breakdown and progress tracking for Auto-mode Ralph loop execution.

**Dependencies**: None

### Step 2: Correctness Baseline Stabilization
**Files to modify/create**:
- `src/scalar/bitpack256v64_scalar.cpp`
- `src/scalar/p4enc256v64_scalar.cpp`
- `src/scalar/p4d1dec256v64_scalar.cpp`
- `src/scalar/p4_scalar.h`
- `src/scalar/p4_scalar_internal.h`
- `src/scalar/p4_scalar_internal.cpp`

**Technical approach**:
Ensure scalar 256v64 implementation is complete, consistent, and semantically correct for delta/non-
delta modes, including helper plumbing.

**Dependencies**: Step 1

### Step 3: API / Dispatch / Build Wiring
**Files to modify/create**:
- `include/turbopfor.h`
- `src/dispatch.cpp`
- `CMakeLists.txt`

**Technical approach**:
Expose new symbols publicly and route them in dispatch to selected backend. Ensure build includes all
required new source units.

**Dependencies**: Step 2

### Step 4: Tests + Benchmark Integration
**Files to modify/create**:
- `tests/binary_compat_test.cpp`
- `benchmarks/ab_test.cpp`

**Technical approach**:
Add 256v64 coverage and benchmark mode (`--simd256v64d1`) for staged perf comparison against
`p4enc64/p4d1dec64`.

**Dependencies**: Step 3

### Step 5: SIMD Path Iteration (Optional/Gated)
**Files to modify/create**:
- `src/simd/p4enc256v64.cpp`
- `src/simd/p4dec256v64.cpp`
- `src/simd/p4d1dec256v64.cpp`
- `src/simd/p4_simd.h`
- `src/dispatch.cpp` (if toggling backend)

**Technical approach**:
Enable SIMD path only when correctness is proven and benchmark signal is favorable; otherwise keep
scalar route as stable default.

**Dependencies**: Step 4

## Testing Strategy
- **Unit/compat tests**: run `binary_compat_test` including new 256v64 coverage.
- **Integration**: verify API + dispatch symbols link and execute.
- **Performance**: run `ab_test --simd256v64d1` in consistent settings; compare to scalar64 baseline.

## Risks and Mitigations
- **Risk**: Non-delta semantics mismatch in wrapper approach.
  -> **Mitigation**: prioritize correctness checks and explicit roundtrip expectations.
- **Risk**: Wrapper SIMD underperforms scalar baseline.
  -> **Mitigation**: gate/disable SIMD by default until optimized.
- **Risk**: Bench noise masks small improvements.
  -> **Mitigation**: repeat runs and look for stable directional changes.

## Rollout Considerations
- Keep API additions backward compatible.
- Prefer phased commits: correctness first, optimization second.
- Do not switch default backend to a path with unresolved correctness/perf concerns.
