# Specification: 256v64 Implementation and Optimization

**ID**: 02-256v64-implementation

## Overview
Add end-to-end 256v64 support in this codebase, mirroring the existing 256v32 design where
appropriate and remaining consistent with current 128v64 behavior and public API conventions.

This change must first establish a correct, benchmarkable baseline path and then iteratively
improve performance. The immediate performance target for this stream is to beat the internal
scalar64 baseline (`p4enc64` / `p4d1dec64`) before broader optimization against TurboPFor C.

## Functional Requirements
### Core Functionality
- Add public API entries for 256v64 encode/decode:
  - `p4Enc256v64`
  - `p4Dec256v64`
  - `p4D1Dec256v64`
- Provide a correct 256v64 implementation path for:
  - plain encode/decode
  - delta1 decode path
- Integrate 256v64 into dispatch so exported API routes to active implementation.
- Ensure build system includes all required scalar/SIMD sources.
- Add benchmark mode to compare 256v64 path against internal scalar64 baseline.

### Edge Cases
- Bit widths across supported range must decode correctly.
- Delta and non-delta semantics must match existing project behavior.
- Roundtrip tests must account for ordering/format nuances already present in 128v64 hybrid paths.

## Non-Functional Requirements
- **Performance**: Stage 1 target is non-regressive and then faster than internal scalar64 baseline for
  designated 256v64 benchmark mode.
- **Security**: No unsafe memory behavior; preserve bounds and pointer arithmetic conventions.
- **Compatibility**: Do not break existing API symbols or current binary compatibility tests.
- **Maintainability**: Prefer reusable/generic structure over fragile per-bitwidth special-casing.

## Integration Points
- Public header: `include/turbopfor.h`
- Dispatch layer: `src/dispatch.cpp`
- Scalar internals: `src/scalar/*`
- SIMD internals: `src/simd/*`
- Build + tests + benches: `CMakeLists.txt`, `tests/binary_compat_test.cpp`, `benchmarks/ab_test.cpp`

## Constraints and Assumptions
### Constraints
- Toolchain and project constraints remain as-is (clang-focused environment).
- Avoid introducing non-portable dependencies outside current project conventions.

### Assumptions
- Existing 256v32 and 128v64 codepaths provide acceptable implementation patterns.
- Benchmark harness provides stable enough signal for staged optimization decisions.

## Out of Scope
- Full redesign of all v64 codec internals unrelated to 256v64.
- Immediate guaranteed dominance over TurboPFor in every scenario in first pass.

## Success Criteria
- 256v64 APIs exist and are wired through dispatch/build.
- Binary compatibility and relevant roundtrip tests pass.
- Benchmark mode `--simd256v64d1` exists and runs cleanly.
- Performance meets or exceeds internal scalar64 baseline in the agreed staging benchmark.

## Open Questions
- For first stable merge in this stream, should SIMD wrapper remain enabled by default or gated behind
  proven correctness/performance?
