// SIMD implementation of 128v64 bitpacking (SSE4.1)
//
// 128v64 is a HYBRID format for 128 x 64-bit values:
//   - When b <= 32: Uses IP32 SIMD shuffle to extract low 32 bits from uint64_t
//     values (with pair-swap reordering), fused directly into bitpacking.
//   - When b > 32: Falls back to scalar bitpack64 (horizontal format).
//
// ENCODE: Fused single-pass IP32+bitpack using periodic-unroll templates.
//   Each bit width B gets compile-time shift constants via template
//   instantiation, matching TurboPFor C's fused macro architecture.
//   Eliminates the temp buffer that the old two-pass approach required.
//
// Binary compatible with TurboPFor's bitpack128v64 / bitunpack128v64.
//
// DISPATCH STRATEGY for D1 decode:
//   Uses monolithic switch(b) dispatching with fully-inlined per-bitwidth code.
//   The __m128i sv (carry register) is passed by value to avoid register spills.
//   Total function is ~86KB (33 cases × ~2.7KB each), which exceeds L1i (32KB),
//   but the monolithic approach avoids call overhead and achieves the best average
//   performance (+3.5%). Some individual bit widths may show outliers due to
//   L1i alignment effects — this is an inherent property of the large switch.

#include "bitunpack_sse_templates.h"
#include "p4_simd_internal.h"

#include <smmintrin.h> // SSE4.1

namespace turbopfor::simd::detail
{

// Pack 128 x 64-bit values using 128v64 hybrid format (SIMD)
//
// When b <= 32: fused IP32 shuffle + bitpack in a single pass via template
//   dispatch. Each bit width B is a compile-time constant, giving immediate-
//   shift instructions and eliminating the temp buffer overhead.
// When b > 32: delegate to scalar bitpack64
unsigned char * bitpack128v64(const uint64_t * in, unsigned char * out, unsigned b)
{
    if (b <= 32u)
    {
        unsigned char * pout = out + (V128_64_BLOCK_SIZE * b + 7u) / 8u;

#define CALL_FUSED_PACK(B) bitpack128v64_fused<B>(in, out)
        STO64_SWITCH(b, CALL_FUSED_PACK);
#undef CALL_FUSED_PACK

        return pout;
    }

    // When b > 32, use scalar 64-bit horizontal bitpacking
    return scalar::detail::bitpack64Scalar(in, V128_64_BLOCK_SIZE, out, b);
}

// Compile-time selector: periodic for P > 2, fully-unrolled otherwise
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_hybrid_entry(const unsigned char * in, uint64_t * out)
{
    constexpr unsigned P = PeriodLen<B>::value;
    if constexpr (B == 0 || P <= 2)
        return bitunpack_sse_sto64_entry<B, Count>(in, out);
    else
        return bitunpack_sse_sto64_periodic_entry<B, Count>(in, out);
}

unsigned char * bitunpack128v64(const unsigned char * in, uint64_t * out, unsigned b)
{
    if (b <= 32u)
    {
        const unsigned char * end = in + (V128_64_BLOCK_SIZE * b + 7u) / 8u;

#define CALL_STO64(B) bitunpack_sse_sto64_hybrid_entry<B, V128_64_BLOCK_SIZE>(in, out)
        STO64_SWITCH(b, CALL_STO64);
#undef CALL_STO64

        return const_cast<unsigned char *>(end);
    }

    // When b > 32, use scalar 64-bit horizontal bitunpacking
    return scalar::detail::bitunpack64Scalar(const_cast<unsigned char *>(in), V128_64_BLOCK_SIZE, out, b);
}

// ============================================================================
// Hybrid D1+EX decode dispatch: periodic-unroll for most B, fully-unrolled
// for B with very short periods (P <= 2, i.e. B=16 and B=32).
// ============================================================================

// Compile-time selector for D1+EX
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char *
bitunpack_sse_sto64_d1_ex_hybrid_entry(const unsigned char * in, uint64_t * out, __m128i sv, const uint64_t * bitmap, const uint32_t *& pex)
{
    constexpr unsigned P = PeriodLen<B>::value;
    if constexpr (B == 0 || P <= 2)
        return bitunpack_sse_sto64_d1_ex_entry<B, Count>(in, out, sv, bitmap, pex);
    else
        return bitunpack_sse_sto64_d1_ex_periodic_entry<B, Count>(in, out, sv, bitmap, pex);
}

// Fused unpack 128v64 + delta1 + exception patching (SIMD)
//
// For b+bx <= 32: hybrid dispatch (periodic or fully-unrolled per bitwidth).
// Returns: pointer past the consumed base-value bitstream.
// Caller must have already unpacked exceptions into pex as uint32_t[].
//
// When the accumulated carry could overflow 32 bits, returns nullptr to signal
// the caller to fall back to multi-phase (SIMD unpack + scalar patch + scalar delta1).
// The caller should check overflow conditions before calling this function.
const unsigned char *
bitd1unpack128v64_ex(const unsigned char * in, uint64_t * out, unsigned b, uint64_t start, const uint64_t * bitmap, const uint32_t *& pex)
{
    if (start > UINT32_MAX)
        return nullptr; // Signal caller to use scalar fallback

    __m128i sv = _mm_set1_epi32(static_cast<uint32_t>(start));
    const unsigned char * result;

#define CALL_STO64_D1_EX(B) result = bitunpack_sse_sto64_d1_ex_hybrid_entry<B, V128_64_BLOCK_SIZE>(in, out, sv, bitmap, pex)
    STO64_SWITCH(b, CALL_STO64_D1_EX);
#undef CALL_STO64_D1_EX

    return result;
}

} // namespace turbopfor::simd::detail
