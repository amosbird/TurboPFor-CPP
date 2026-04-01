#pragma once

#include "scalar/p4_scalar_internal.h"

#include <smmintrin.h> // SSE4.1

/// MSan support: unpoison memory that will be partially written then read via SIMD.
/// SIMD loads may read beyond the actually-written elements, which is safe
/// (unused lanes are masked out) but triggers MSan false positives.
#if defined(__clang__) && defined(__has_feature)
#    if __has_feature(memory_sanitizer)
#        include <sanitizer/msan_interface.h>
#        define TURBOPFOR_MSAN_UNPOISON(ptr, size) __msan_unpoison(ptr, size)
#    endif
#endif

#ifndef TURBOPFOR_MSAN_UNPOISON
#    define TURBOPFOR_MSAN_UNPOISON(ptr, size) static_cast<void>(0)
#endif

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
using scalar::detail::bitWidth64;
using scalar::detail::bsr32;
using scalar::detail::bsr64;
using scalar::detail::loadU32;
using scalar::detail::loadU64;
using scalar::detail::loadU64Fast;
using scalar::detail::maskBits;
using scalar::detail::maskBits64;
using scalar::detail::MAX_BITS;
using scalar::detail::MAX_BITS_64;
using scalar::detail::MAX_VALUES;
using scalar::detail::pad8;
using scalar::detail::storeU32;
using scalar::detail::storeU64Fast;

/// SSE4.1 128v vertical bitpacking (4-lane interleaved, 128 elements)
unsigned char * bitpack128v32(const uint32_t * in, unsigned char * out, unsigned b);
const unsigned char * bitunpack128v32(const unsigned char * in, uint32_t * out, unsigned b);

/// SSE4.1 128v vertical bitunpacking with delta1 decode (fused operation)
const unsigned char * bitd1unpack128v32(const unsigned char * in, uint32_t * out, unsigned b, uint32_t start);

/// SSE4.1 128v vertical bitunpacking with delta1 decode and exception patching (fused)
const unsigned char *
bitd1unpack128v32_ex(const unsigned char * in, uint32_t * out, unsigned b, uint32_t start, const uint64_t * bitmap, const uint32_t *& pex);

/// SSE4.1 128v64 hybrid bitpacking (128 x 64-bit values)
/// b <= 32: IP32 shuffle + bitpack128v32 SIMD
/// b > 32: scalar bitpack64
unsigned char * bitpack128v64(const uint64_t * in, unsigned char * out, unsigned b);
unsigned char * bitunpack128v64(const unsigned char * in, uint64_t * out, unsigned b);

/// Fused 128v64 unpack + delta1 decode (saves one full pass over output)
unsigned char * bitunpackD1_128v64(const unsigned char * in, uint64_t * out, unsigned b, uint64_t start);

/// Fused 128v64 unpack + delta1 + exception patching (single-pass SSSE3, b+bx <= 32 only)
/// Exceptions must be pre-unpacked as uint32_t[] into pex; pex is advanced past consumed exceptions.
const unsigned char *
bitd1unpack128v64_ex(const unsigned char * in, uint64_t * out, unsigned b, uint64_t start, const uint64_t * bitmap, const uint32_t *& pex);

/// Variable-byte encoding/decoding (reuse from scalar - not SIMD critical path)
unsigned char * vbEnc32(const uint32_t * in, unsigned n, unsigned char * out);
const unsigned char * vbDec32(const unsigned char * in, unsigned n, uint32_t * out);

/// P4 bit width selection
unsigned p4Bits32(const uint32_t * in, unsigned n, unsigned * pbx);

/// Write P4 header (32-bit)
void writeHeader(unsigned char *& out, unsigned b, unsigned bx);

/// Write P4 header (64-bit: max base bits = 64, with 63->64 bit quirk)
void writeHeader64(unsigned char *& out, unsigned b, unsigned bx);

/// Apply delta1 decoding with SIMD (32-bit)
/// NOT inlined to match TurboPFor's bitd1dec32 structure - improves icache behavior
void applyDelta1(uint32_t * out, unsigned n, uint32_t start);

/// Apply delta1 decoding for 64-bit output (scalar — SIMD is possible but not critical path)
void applyDelta1_64(uint64_t * out, unsigned n, uint64_t start);

} // namespace turbopfor::simd::detail
