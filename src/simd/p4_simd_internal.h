#pragma once

#include "scalar/p4_scalar_internal.h"

#include <smmintrin.h> // SSE4.1

// Define ALWAYS_INLINE for the SIMD implementation
#ifndef ALWAYS_INLINE
#    ifdef _MSC_VER
#        define ALWAYS_INLINE __forceinline
#    else
#        define ALWAYS_INLINE inline __attribute__((always_inline))
#    endif
#endif

// Define ALIGNAS for aligned data
#ifndef ALIGNAS
#    ifdef _MSC_VER
#        define ALIGNAS(x) __declspec(align(x))
#    else
#        define ALIGNAS(x) __attribute__((aligned(x)))
#    endif
#endif

namespace turbopfor::simd::detail
{

// Import constants and utilities from scalar namespace
using scalar::detail::bitWidth32;
using scalar::detail::bsr32;
using scalar::detail::loadU32;
using scalar::detail::loadU64;
using scalar::detail::maskBits;
using scalar::detail::MAX_BITS;
using scalar::detail::MAX_VALUES;
using scalar::detail::pad8;
using scalar::detail::storeU32;

/// SSE4.1 128v vertical bitpacking (4-lane interleaved, 128 elements)
unsigned char * bitpack128v32(const uint32_t * in, unsigned char * out, unsigned b);
const unsigned char * bitunpack128v32(const unsigned char * in, uint32_t * out, unsigned b);

/// SSE4.1 128v vertical bitunpacking with delta1 decode (fused operation)
const unsigned char * bitd1unpack128v32(const unsigned char * in, uint32_t * out, unsigned b, uint32_t start);

/// SSE4.1 128v vertical bitunpacking with delta1 decode and exception patching (fused)
const unsigned char *
bitd1unpack128v32_ex(const unsigned char * in, uint32_t * out, unsigned b, uint32_t start, const uint64_t * bitmap, const uint32_t *& pex);

/// Variable-byte encoding/decoding (reuse from scalar - not SIMD critical path)
unsigned char * vbEnc32(const uint32_t * in, unsigned n, unsigned char * out);
const unsigned char * vbDec32(const unsigned char * in, unsigned n, uint32_t * out);

/// P4 bit width selection
unsigned p4Bits32(const uint32_t * in, unsigned n, unsigned * pbx);

/// Write P4 header
void writeHeader(unsigned char *& out, unsigned b, unsigned bx);

/// Apply delta1 decoding with SIMD
/// NOT inlined to match TurboPFor's bitd1dec32 structure - improves icache behavior
void applyDelta1(uint32_t * out, unsigned n, uint32_t start);

} // namespace turbopfor::simd::detail
