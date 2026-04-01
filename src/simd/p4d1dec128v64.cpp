// SIMD implementation of P4 decoding with delta1 for 128v64 format
//
// Uses fused SIMD approach matching TurboPFor C's p4d1dec128v64:
// - No exceptions (b <= 32): fused bitunpack + delta1 prefix scan + STO64 zero-extend
// - Bitmap exceptions (b+bx <= 32): fused bitunpack + SSSE3 exception merge + delta1 + STO64
// - Fallback (b > 32 or b+bx > 32): scalar multi-phase approach

#include "p4_simd.h"
#include "p4_simd_internal.h"
#include "bitunpack_sse_templates.h"

#include <smmintrin.h> // SSE4.1

namespace turbopfor::simd
{

namespace detail
{

template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_d1_hybrid_entry(const unsigned char * in, uint64_t * out, __m128i sv)
{
    constexpr unsigned P = PeriodLen<B>::value;
    if constexpr (B == 0 || P <= 2)
        return bitunpack_sse_sto64_d1_entry<B, Count>(in, out, sv);
    else
        return bitunpack_sse_sto64_d1_periodic_entry<B, Count>(in, out, sv);
}

__attribute__((noinline)) static void
bitunpack128v64_noinline(const unsigned char * in, uint64_t * out, unsigned b)
{
#define CALL_STO64_NODELTA(B) bitunpack_sse_sto64_entry<B, V128_64_BLOCK_SIZE>(in, out)
    STO64_SWITCH(b, CALL_STO64_NODELTA);
#undef CALL_STO64_NODELTA
}

unsigned char * bitunpackD1_128v64(const unsigned char * in, uint64_t * out, unsigned b, uint64_t start)
{
    if (b <= 32u)
    {
        const unsigned char * ip_end = in + (V128_64_BLOCK_SIZE * b + 7u) / 8u;

        const bool overflow = start > UINT32_MAX
            || start + static_cast<uint64_t>(V128_64_BLOCK_SIZE) * (b == 32 ? 0xFFFFFFFFULL : ((1ULL << b) - 1ULL))
                   + V128_64_BLOCK_SIZE > UINT32_MAX;
        if (__builtin_expect(overflow, 0))
        {
            bitunpack128v64_noinline(in, out, b);

            uint64_t acc = start;
            for (unsigned i = 0; i < V128_64_BLOCK_SIZE; i += 4)
            {
                acc += out[i + 0] + 1; out[i + 0] = acc;
                acc += out[i + 1] + 1; out[i + 1] = acc;
                acc += out[i + 2] + 1; out[i + 2] = acc;
                acc += out[i + 3] + 1; out[i + 3] = acc;
            }

            return const_cast<unsigned char *>(ip_end);
        }

        __m128i sv = _mm_set1_epi32(static_cast<uint32_t>(start));

#define CALL_STO64_D1(B) bitunpack_sse_sto64_d1_hybrid_entry<B, V128_64_BLOCK_SIZE>(in, out, sv)
        STO64_SWITCH(b, CALL_STO64_D1);
#undef CALL_STO64_D1

        return const_cast<unsigned char *>(ip_end);
    }

    return scalar::detail::bitunpackd1_64Scalar(
        const_cast<unsigned char *>(in), V128_64_BLOCK_SIZE, out, start, b);
}

} // namespace detail

namespace
{

// Decode P4 payload with bitmap exceptions for 128v64 format
//
// Two paths:
//   b+bx <= 32: Fused single-pass — unpack exceptions as uint32_t, then fused
//               unpack + SSSE3 exception merge + delta1 + STO64 (matches TurboPFor C).
//   b+bx > 32:  Multi-phase — unpack exceptions as uint64_t, unpack base values,
//               apply patches, apply delta1 (scalar fallback).
__attribute__((noinline)) const unsigned char *
p4D1Dec128v64PayloadBitmap(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

    // Phase 1: Read bitmap + popcount
    uint64_t bitmap[MAX_VALUES / 64] = {0};
    const unsigned words = (n + 63u) / 64u;
    unsigned exception_count = 0;

    for (unsigned i = 0; i < words; ++i)
    {
        uint64_t word = loadU64Fast(in + i * sizeof(uint64_t));
        if (i == words - 1u && (n & 0x3Fu))
            word &= (1ull << (n & 0x3Fu)) - 1ull;

        bitmap[i] = word;
#if defined(__GNUC__) || defined(__clang__)
        exception_count += static_cast<unsigned>(__builtin_popcountll(word));
#else
        uint64_t tmp = word;
        while (tmp)
        {
            ++exception_count;
            tmp &= tmp - 1ull;
        }
#endif
    }

    const unsigned char * ip = in + pad8(n);

    if (b + bx <= 32u)
    {
        alignas(16) uint32_t ex[MAX_VALUES + 64];
        const unsigned char * ex_end = scalar::detail::bitunpack32Scalar(const_cast<unsigned char *>(ip), exception_count, ex, bx);

        // Try fused SIMD path: bitunpack + SSSE3 exception merge + delta1 + STO64
        // Uses periodic template (smaller L1i footprint than fully-unrolled non-delta).
        // Only works when start fits in 32-bit carry register.
        const uint32_t * pex = ex;
        const unsigned char * base_end = bitd1unpack128v64_ex(ex_end, out, b, start, bitmap, pex);

        if (base_end != nullptr)
            return base_end;

        // Overflow fallback: SIMD unpack + scalar patch + scalar delta1
        ip = bitunpack128v64(ex_end, out, b);

        unsigned k = 0;
        for (unsigned i = 0; i < words; ++i)
        {
            uint64_t word = bitmap[i];
            while (word)
            {
                unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
                const unsigned idx = i * 64u + bit;
                out[idx] |= static_cast<uint64_t>(ex[k++]) << b;
                word &= word - 1ull;
            }
        }

        applyDelta1_64(out, n, start);

        return ip;
    }

    // SLOW PATH: b+bx > 32, scalar multi-phase approach

    // Phase 2: Unpack exception values (scalar bitpack64)
    uint64_t exceptions[MAX_VALUES + 64];
    ip = scalar::detail::bitunpack64Scalar(const_cast<unsigned char *>(ip), exception_count, exceptions, bx);

    // Phase 3: Unpack base values (SIMD bitunpack128v64)
    ip = bitunpack128v64(ip, out, b);

    // Phase 4: Apply patches
    unsigned k = 0;
    for (unsigned i = 0; i < words; ++i)
    {
        uint64_t word = bitmap[i];
        while (word)
        {
#if defined(__GNUC__) || defined(__clang__)
            unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
#else
            unsigned bit = 0;
            while (((word >> bit) & 1ull) == 0ull)
                ++bit;
#endif
            const unsigned idx = i * 64u + bit;
            out[idx] |= exceptions[k++] << b;
            word &= word - 1ull;
        }
    }

    // Phase 5: Apply delta1 decoding
    applyDelta1_64(out, n, start);

    return ip;
}

} // namespace

unsigned char * p4D1Dec128v64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    using namespace turbopfor::simd::detail;

    if (n == 0u)
        return in;

    unsigned char * ip = in;
    unsigned b = *ip++;

    // Case 1: Constant block
    if ((b & 0xC0u) == 0xC0u)
    {
        b &= 0x3Fu;
        if (b == 63u)
            b = 64u;

        const unsigned bytes_stored = (b + 7u) / 8u;
        uint64_t value = loadU64Fast(ip);
        if (b < 64u)
            value &= (1ull << b) - 1ull;

        for (unsigned i = 0; i < n; ++i)
            out[i] = (start += value) + (i + 1u);

        return ip + bytes_stored;
    }

    // Case 2: Standard bitpacking (possibly with bitmap exceptions)
    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        if (b & 0x80u)
        {
            bx = *ip++;
            b &= 0x7Fu;
        }

        if (b == 63u)
            b = 64u;

        if (bx == 0u)
        {
            ip = bitunpackD1_128v64(ip, out, b, start);
            return ip;
        }

        return const_cast<unsigned char *>(p4D1Dec128v64PayloadBitmap(ip, n, out, start, b, bx));
    }

    // Case 3: Variable-byte exceptions
    b &= 0x3Fu;
    if (b == 63u)
        b = 64u;

    const unsigned exception_count = *ip++;

    ip = bitunpack128v64(ip, out, b);

    uint64_t exceptions[MAX_VALUES + 64];
    ip = scalar::detail::vbDec64(ip, exception_count, exceptions);

    // Apply patches
    unsigned i = 0;
    const unsigned ec8 = exception_count & ~7u;
    for (; i < ec8; i += 8)
    {
        out[ip[i + 0]] |= exceptions[i + 0] << b;
        out[ip[i + 1]] |= exceptions[i + 1] << b;
        out[ip[i + 2]] |= exceptions[i + 2] << b;
        out[ip[i + 3]] |= exceptions[i + 3] << b;
        out[ip[i + 4]] |= exceptions[i + 4] << b;
        out[ip[i + 5]] |= exceptions[i + 5] << b;
        out[ip[i + 6]] |= exceptions[i + 6] << b;
        out[ip[i + 7]] |= exceptions[i + 7] << b;
    }
    for (; i < exception_count; ++i)
        out[ip[i]] |= exceptions[i] << b;

    ip += exception_count;
    applyDelta1_64(out, n, start);
    return ip;
}

} // namespace turbopfor::simd
