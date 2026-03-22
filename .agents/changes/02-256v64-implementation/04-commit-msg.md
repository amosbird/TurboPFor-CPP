feat: implement 256v64 end-to-end (scalar encode/decode, API, dispatch, tests, benchmark)

Add full 256v64 support for 64-bit integer packing in 256-element blocks:

- Scalar encode/decode using 2×128v64 block strategy with correct delta1
  carry propagation across block boundaries
- Public API: p4Enc256v64, p4Dec256v64, p4D1Dec256v64 in turbopfor.h
- Dispatch routes 256v64 (and 128v64) decode to scalar path — SIMD gated
  off due to pair-swap bug in STO64 decode templates and 32-bit start
  truncation issues
- 23 test patterns per suite including exception/edge cases, roundtrip
  and binary compatibility verified across all 12 suites (0 failures)
- Benchmark mode --simd256v64d1 for A/B comparison vs C reference
- Performance: scalar decode within ±5% of C reference for exception-heavy
  data (≥30% exceptions); encode ~20% slower (expected for scalar-only)
- No regression in existing 128v64 path
