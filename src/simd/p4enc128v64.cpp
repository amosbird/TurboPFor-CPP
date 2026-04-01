// SIMD implementation of P4 encoding for 128v64 format
//
// Uses SIMD bitpack128v64 for base values (IP32 shuffle + 128v32 SIMD when b<=32).
// Exception handling reuses scalar utilities since exceptions are a cold path.
//
// Binary compatible with TurboPFor's p4enc128v64().

#include "p4_simd.h"
#include "p4_simd_internal.h"

#include <smmintrin.h> // SSE4.1

namespace turbopfor::simd
{

namespace
{

// Encode P4 block payload with exceptions for 128v64 format
//
// Two exception strategies:
// 1. Bitmap patching (bx <= 64): [bitmap][patch bits (bitpack64)][base bits (bitpack128v64 SIMD)]
// 2. Variable-byte (bx == 65): [count][base bits (bitpack128v64 SIMD)][vbyte64][positions]
unsigned char * p4Enc128v64PayloadExceptions(uint64_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

    const uint64_t base_mask = maskBits64(b);

    uint64_t base[MAX_VALUES];
    uint64_t exceptions[MAX_VALUES];
    unsigned exception_positions[MAX_VALUES];

    // Phase 1: Split values into base and exceptions (branchless)
    unsigned exception_count = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        exception_positions[exception_count] = i;
        exception_count += (in[i] > base_mask) ? 1u : 0u;
        base[i] = in[i] & base_mask;
    }

    // Phase 2: Build exception data
    uint64_t bitmap[MAX_VALUES / 64] = {0};
    for (unsigned i = 0; i < exception_count; ++i)
    {
        const unsigned pos = exception_positions[i];
        bitmap[pos >> 6] |= 1ull << (pos & 0x3Fu);
        exceptions[i] = in[pos] >> b;
    }

    // Phase 3: Encode
    if (bx <= MAX_BITS_64)
    {
        // Bitmap patching: [bitmap][patch bits][base bits]
        const unsigned bitmap_words = (n + 63u) / 64u;
        for (unsigned i = 0; i < bitmap_words; ++i)
            storeU64Fast(out + i * sizeof(uint64_t), bitmap[i]);

        out += pad8(n);

        // Patch bits always use scalar bitpack64
        out = scalar::detail::bitpack64Scalar(exceptions, exception_count, out, bx);

        // Base values use SIMD bitpack128v64
        out = bitpack128v64(base, out, b);

        return out;
    }

    // Variable-byte exceptions: [count][base bits][vbyte64][positions]
    *out++ = static_cast<unsigned char>(exception_count);

    out = bitpack128v64(base, out, b);
    out = scalar::detail::vbEnc64(exceptions, exception_count, out);

    for (unsigned i = 0; i < exception_count; ++i)
        *out++ = static_cast<unsigned char>(exception_positions[i]);

    return out;
}

unsigned char * p4Enc128v64Payload(uint64_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

    if (bx == 0u)
        return bitpack128v64(in, out, b);

    if (bx == MAX_BITS_64 + 2u)
    {
        storeU64Fast(out, in[0]);
        return out + (b + 7u) / 8u;
    }

    return p4Enc128v64PayloadExceptions(in, n, out, b, bx);
}

} // namespace

unsigned char * p4Enc128v64(uint64_t * in, unsigned n, unsigned char * out)
{
    using namespace turbopfor::simd::detail;

    if (n == 0u)
        return out;

    unsigned bx = 0;
    unsigned b = scalar::detail::p4Bits64(in, n, &bx);

    writeHeader64(out, b, bx);
    return p4Enc128v64Payload(in, n, out, b, bx);
}

} // namespace turbopfor::simd
