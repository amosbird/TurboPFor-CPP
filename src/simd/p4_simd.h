#pragma once

#include <cstdint>

namespace turbopfor::simd
{

// 128-element vertical bitpacking format (128v32) - SSE4.2 SIMD implementation
// 256-element vertical bitpacking format (256v32) - AVX2 SIMD implementation
// Standard horizontal bitpacking format (p4enc32/p4d1dec32) uses scalar implementation

/// P4 encoding (128-element vertical bitpacking format)
unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out);

/// P4 decoding with delta1 (128-element vertical bitpacking format)
unsigned char * p4D1Dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

/// P4 encoding (256-element vertical bitpacking format, AVX2)
unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out);

/// P4 decoding with delta1 (256-element vertical bitpacking format, AVX2)
unsigned char * p4D1Dec256v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

/// P4 encoding (128-element hybrid 64-bit format, SSE4.1)
/// Uses 128v32 SIMD when b<=32, scalar bitpack64 when b>32
unsigned char * p4Enc128v64(uint64_t * in, unsigned n, unsigned char * out);

/// P4 decoding without delta (128-element hybrid 64-bit format, SSE4.1)
/// Matches TurboPFor C's p4dec128v64 — no delta prefix sum applied
unsigned char * p4Dec128v64(unsigned char * in, unsigned n, uint64_t * out);

/// P4 decoding with delta1 (128-element hybrid 64-bit format, SSE4.1)
unsigned char * p4D1Dec128v64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

/// P4 encoding (256-element hybrid 64-bit format, AVX2/scalar-hybrid)
unsigned char * p4Enc256v64(uint64_t * in, unsigned n, unsigned char * out);

/// P4 decoding without delta (256-element hybrid 64-bit format)
unsigned char * p4Dec256v64(unsigned char * in, unsigned n, uint64_t * out);

/// P4 decoding with delta1 (256-element hybrid 64-bit format)
unsigned char * p4D1Dec256v64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

} // namespace turbopfor::simd
