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

} // namespace turbopfor::simd
