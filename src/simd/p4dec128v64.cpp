// SIMD implementation of P4 decoding (non-delta) for 128v64 format
//
// Uses SIMD bitunpack128v64 for base values (128v32 SIMD + STO64 zero-extend when b<=32).
// Exception handling reuses scalar utilities since exceptions are a cold path.
//
// This is the non-delta variant, matching TurboPFor C's p4dec128v64.
// No delta prefix sum is applied — output contains raw decoded values.

#include "p4_simd.h"
#include "p4_simd_internal.h"

#include <smmintrin.h> // SSE4.1

namespace turbopfor::simd
{

namespace
{

// Decode P4 payload with bitmap exceptions for 128v64 format (non-delta)
//
// Format: [bitmap][patch bits (bitpack64)][base bits (bitpack128v64)]
__attribute__((noinline)) const unsigned char *
p4Dec128v64PayloadBitmap(const unsigned char * in, unsigned n, uint64_t * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

    // Phase 1: Read bitmap
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

    // Phase 2: Unpack exception values (scalar bitpack64)
    uint64_t exceptions[MAX_VALUES + 64];
    ip = scalar::detail::bitunpack64Scalar(ip, exception_count, exceptions, bx);

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

    return ip;
}

} // namespace

const unsigned char * p4Dec128v64(const unsigned char * in, unsigned n, uint64_t * out)
{
    using namespace turbopfor::simd::detail;

    if (n == 0u)
        return in;

    const unsigned char * ip = in;
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
            out[i] = value;

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
            ip = bitunpack128v64(ip, out, b);
            return ip;
        }

        return p4Dec128v64PayloadBitmap(ip, n, out, b, bx);
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
    return ip;
}

} // namespace turbopfor::simd
