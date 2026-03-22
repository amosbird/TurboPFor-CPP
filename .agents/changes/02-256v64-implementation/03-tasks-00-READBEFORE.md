# Critical Context for All Tasks

## Change Request Summary
Implement and stabilize 256v64 support end-to-end (API, dispatch, scalar/SIMD backend, tests,
benchmark mode), then iterate toward better performance with correctness guarded first.

## Specification Reference
See `01-specification.md`.

## Key Design Decisions
- Prioritize correctness and deterministic behavior before optimization.
- Incomplete tasks in `PROGRESS.md` have highest priority over Not Started tasks.
- SIMD path is optional/gated until validated; scalar fallback is acceptable as stable default.

## Coding Standards
- Follow existing project style and naming conventions.
- Keep changes localized and maintainable.
- Avoid risky rewrites outside scope.

## Testing Requirements
- Run relevant compatibility tests after each task.
- Use benchmark mode `--simd256v64d1` for staged perf validation.
- Ensure builds/tests are green before marking a task complete.

## Common Pitfalls
- Assuming non-delta ordering semantics without verifying existing v64 behavior.
- Enabling SIMD backend prematurely before semantic parity and benchmark sanity are proven.
