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

// Number of elements in 128v64 block
constexpr unsigned V128_64_BLOCK_SIZE = 128;

// ============================================================================
// Switch-based dispatch macro for bitwidth dispatch.
// Used for both encode (fused IP32 bitpack) and decode (STO64 unpack).
// ============================================================================

#define STO64_SWITCH(b, CALL)       \
    switch (b)                      \
    {                               \
        case 0: CALL(0); break;     \
        case 1: CALL(1); break;     \
        case 2: CALL(2); break;     \
        case 3: CALL(3); break;     \
        case 4: CALL(4); break;     \
        case 5: CALL(5); break;     \
        case 6: CALL(6); break;     \
        case 7: CALL(7); break;     \
        case 8: CALL(8); break;     \
        case 9: CALL(9); break;     \
        case 10: CALL(10); break;   \
        case 11: CALL(11); break;   \
        case 12: CALL(12); break;   \
        case 13: CALL(13); break;   \
        case 14: CALL(14); break;   \
        case 15: CALL(15); break;   \
        case 16: CALL(16); break;   \
        case 17: CALL(17); break;   \
        case 18: CALL(18); break;   \
        case 19: CALL(19); break;   \
        case 20: CALL(20); break;   \
        case 21: CALL(21); break;   \
        case 22: CALL(22); break;   \
        case 23: CALL(23); break;   \
        case 24: CALL(24); break;   \
        case 25: CALL(25); break;   \
        case 26: CALL(26); break;   \
        case 27: CALL(27); break;   \
        case 28: CALL(28); break;   \
        case 29: CALL(29); break;   \
        case 30: CALL(30); break;   \
        case 31: CALL(31); break;   \
        case 32: CALL(32); break;   \
        default: __builtin_unreachable(); \
    }

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
// Hybrid D1 decode dispatch: periodic-unroll for most bit widths, fully-
// unrolled for bit widths where the period is very short (P <= 2).
//
// Periodic-unroll reduces code size from ~86KB to ~60KB, eliminating L1i
// cache pressure that caused per-bitwidth outliers (e.g. b=28 was -11%).
// However, for B=32 (P=1) and B=16 (P=2), the loop overhead of many
// single/double-group iterations dominates, so fully-unrolled is faster.
// ============================================================================

// Fused unpack 128v64 + delta1 decode (SIMD)
//
// Hybrid dispatch: uses periodic-unroll for most bit widths (smaller code,
// fits in L1i), but falls back to fully-unrolled for bit widths where the
// period is very short (P <= 2), making the loop overhead dominant.
//   B=32: P=1 (32 loop iterations of 1 group each — unacceptable overhead)
//   B=16: P=2 (16 loop iterations of 2 groups each — marginal)
// For these, the fully-unrolled ~2.7KB case code is used instead.
//
// For b > 32: scalar bitunpack64 + scalar delta1.

// Compile-time selector: periodic for P > 2, fully-unrolled otherwise
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_d1_hybrid_entry(const unsigned char * in, uint64_t * out, __m128i sv)
{
    constexpr unsigned P = PeriodLen<B>::value;
    if constexpr (B == 0 || P <= 2)
        return bitunpack_sse_sto64_d1_entry<B, Count>(in, out, sv);
    else
        return bitunpack_sse_sto64_d1_periodic_entry<B, Count>(in, out, sv);
}

// Overflow-safe path: two-pass approach to avoid scalar extraction overhead.
// Pass 1: SIMD non-delta unpack to 64-bit output (fast, pure SIMD).
// Pass 2: tight sequential 64-bit prefix sum over L1-hot data.
__attribute__((noinline)) static void
bitunpackD1_128v64_overflow(const unsigned char * in, uint64_t * out, unsigned b, uint64_t start)
{
#define CALL_STO64_NODELTA(B) bitunpack_sse_sto64_entry<B, V128_64_BLOCK_SIZE>(in, out)
    STO64_SWITCH(b, CALL_STO64_NODELTA);
#undef CALL_STO64_NODELTA

    uint64_t acc = start;
    for (unsigned i = 0; i < V128_64_BLOCK_SIZE; ++i)
    {
        acc += out[i] + 1;
        out[i] = acc;
    }
}

unsigned char * bitunpackD1_128v64(const unsigned char * in, uint64_t * out, unsigned b, uint64_t start)
{
    if (b <= 32u)
    {
        const unsigned char * ip_end = in + (V128_64_BLOCK_SIZE * b + 7u) / 8u;

        // The fused SIMD D1 path uses 32-bit prefix sum (_mm_add_epi32).
        // The accumulated carry can overflow 32 bits if start + sum_of_deltas + count > UINT32_MAX.
        // Maximum possible sum: start + 128 * ((1 << b) - 1) + 128 (delta1 +1 per element).
        // Use the fused path only when this cannot overflow.
        const uint64_t max_sum = start + static_cast<uint64_t>(V128_64_BLOCK_SIZE) * (b == 32 ? 0xFFFFFFFFULL : ((1ULL << b) - 1ULL))
                                 + V128_64_BLOCK_SIZE;
        if (max_sum <= UINT32_MAX)
        {
            // Fast path: fused SIMD D1 prefix scan is exact (no 32-bit overflow)
            __m128i sv = _mm_set1_epi32(static_cast<uint32_t>(start));

#define CALL_STO64_D1(B) bitunpack_sse_sto64_d1_hybrid_entry<B, V128_64_BLOCK_SIZE>(in, out, sv)
            STO64_SWITCH(b, CALL_STO64_D1);
#undef CALL_STO64_D1
        }
        else
        {
            bitunpackD1_128v64_overflow(in, out, b, start);
        }

        return const_cast<unsigned char *>(ip_end);
    }

    // b > 32: data is in horizontal (scalar) format, use fused scalar delta1+unpack
    // (single pass over data instead of separate unpack + delta1)
    return scalar::detail::bitunpackd1_64Scalar(
        const_cast<unsigned char *>(in), V128_64_BLOCK_SIZE, out, start, b);
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

#undef STO64_SWITCH

} // namespace turbopfor::simd::detail
