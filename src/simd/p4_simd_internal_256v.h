#pragma once

#include "scalar/p4_scalar_internal.h"

#include <immintrin.h> // AVX2

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

/// AVX2 256v vertical bitpacking (8-lane interleaved, 256 elements)
unsigned char * bitpack256v32(const uint32_t * in, unsigned char * out, unsigned b);
const unsigned char * bitunpack256v32(const unsigned char * in, uint32_t * out, unsigned b);

/// AVX2 256v vertical bitunpacking with delta1 decode (fused operation)
const unsigned char * bitd1unpack256v32(const unsigned char * in, uint32_t * out, unsigned b, uint32_t start);

/// AVX2 256v vertical bitunpacking with delta1 decode and exception patching (fused)
const unsigned char *
bitd1unpack256v32_ex(const unsigned char * in, uint32_t * out, unsigned b, uint32_t start, const uint64_t * bitmap, const uint32_t *& pex);

/// Variable-byte encoding/decoding (reuse from scalar - not SIMD critical path)
unsigned char * vbEnc32_256v(const uint32_t * in, unsigned n, unsigned char * out);
const unsigned char * vbDec32_256v(const unsigned char * in, unsigned n, uint32_t * out);

/// P4 bit width selection
unsigned p4Bits32_256v(const uint32_t * in, unsigned n, unsigned * pbx);

/// Write P4 header
void writeHeader_256v(unsigned char *& out, unsigned b, unsigned bx);

/// Apply delta1 decoding with AVX2
/// NOT inlined to match TurboPFor's bitd1dec32 structure - improves icache behavior
void applyDelta1_256v(uint32_t * out, unsigned n, uint32_t start);

// Exact copy of TurboPFor's MM256_HDEC_EPI32 pattern - this is the key optimization!
// mm256_scan_epi32: AVX2 inclusive prefix sum (scan)
// Input: v = [a, b, c, d, e, f, g, h], sv = previous vector (we use sv[7] as carry)
// Output: [sv[7]+a, sv[7]+a+b, sv[7]+a+b+c, sv[7]+a+b+c+d,
//          sv[7]+a+b+c+d+e, sv[7]+a+b+c+d+e+f, sv[7]+a+b+c+d+e+f+g, sv[7]+a+b+c+d+e+f+g+h]
ALWAYS_INLINE __m256i mm256_scan_epi32(__m256i v, __m256i sv)
{
    // In-lane prefix sum within each 128-bit lane
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 4));
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 8));
    // Now: v = [a, a+b, a+b+c, a+b+c+d, e, e+f, e+f+g, e+f+g+h]

    // Cross-lane carry propagation (TurboPFor's MM256_HDEC_EPI32 pattern)
    // This is the magic: we need to add sv[7] to all elements AND v[3] to upper lane
    //
    // permute2x128(shuffle(sv, 0xFF), sv, 0x11):
    //   shuffle(sv, 0xFF) = [sv[3],sv[3],sv[3],sv[3], sv[7],sv[7],sv[7],sv[7]]
    //   permute2x128(..., sv, 0x11) selects [high(sv),high(sv)] = [sv[7] broadcast, sv[7] broadcast]
    //
    // permute2x128(zero, shuffle(v, 0xFF), 0x20):
    //   shuffle(v, 0xFF) = [v[3],v[3],v[3],v[3], v[7],v[7],v[7],v[7]]
    //   permute2x128(zero, ..., 0x20) selects [zero, low(...)] = [0,0,0,0, v[3],v[3],v[3],v[3]]
    //
    // Final: sv_broadcast + (v + lane_sum)
    return _mm256_add_epi32(
        _mm256_permute2x128_si256(_mm256_shuffle_epi32(sv, _MM_SHUFFLE(3, 3, 3, 3)), sv, 0x11),
        _mm256_add_epi32(v, _mm256_permute2x128_si256(_mm256_setzero_si256(), _mm256_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 3, 3)), 0x20)));
}

// mm256_scani_epi32: scan + add increment vector (for delta1: adds 1,2,3,4,5,6,7,8)
ALWAYS_INLINE __m256i mm256_scani_epi32(__m256i v, __m256i sv, __m256i vi)
{
    return _mm256_add_epi32(mm256_scan_epi32(v, sv), vi);
}

} // namespace turbopfor::simd::detail
