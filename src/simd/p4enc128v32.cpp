#include "p4_simd.h"
#include "p4_simd_internal.h"
#include "p4bits128_simd.h"

namespace turbopfor::simd
{

namespace
{

// Optimized payload encoding for 128 elements with exceptions
// Uses single-pass algorithm to build base values, exceptions, and bitmap simultaneously
unsigned char * p4Enc128PayloadExceptions(uint32_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

    const uint32_t msk = (b >= 32u) ? 0xFFFFFFFFu : ((1u << b) - 1u);

    // Aligned arrays for better cache performance
    alignas(16) uint32_t base[128];
    alignas(16) uint32_t ex[128];
    alignas(16) uint64_t bitmap[2] = {0, 0};  // 128 bits = 2 x 64-bit words
    alignas(16) uint8_t positions[128];

    // Single-pass: extract base values and track exceptions
    // Use SIMD comparison to find exceptions, then scalar to collect them
    unsigned xn = 0;

    // Process with SIMD for base extraction and exception detection
    const __m128i msk_vec = _mm_set1_epi32(static_cast<int>(msk));

    for (unsigned i = 0; i < n; i += 4)
    {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
        __m128i base_v = _mm_and_si128(v, msk_vec);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(base + i), base_v);

        // Check for exceptions: v > msk means high bits are set
        // cmpgt is signed, so we use unsigned comparison trick:
        // For unsigned: a > b iff (a ^ 0x80000000) > (b ^ 0x80000000) (signed)
        // But simpler: just check if (v & ~msk) != 0
        __m128i high_bits = _mm_andnot_si128(msk_vec, v);
        __m128i has_exc = _mm_cmpeq_epi32(high_bits, _mm_setzero_si128());
        // has_exc is 0xFFFFFFFF where NO exception, 0 where exception exists

        int exc_mask = _mm_movemask_ps(_mm_castsi128_ps(has_exc));
        // exc_mask bit is 1 where NO exception, 0 where exception

        // Process exceptions (inverted mask)
        unsigned exc_bits = static_cast<unsigned>(~exc_mask) & 0xFu;
        while (exc_bits)
        {
            unsigned bit = static_cast<unsigned>(__builtin_ctz(exc_bits));
            unsigned idx = i + bit;
            positions[xn] = static_cast<uint8_t>(idx);
            ex[xn] = in[idx] >> b;
            bitmap[idx >> 6] |= 1ull << (idx & 0x3Fu);
            ++xn;
            exc_bits &= exc_bits - 1u;
        }
    }

    if (bx <= MAX_BITS)
    {
        // Bitmap exception encoding
        // Format: [bitmap][exception bits][base bits]
        std::memcpy(out, bitmap, 16);  // 128 bits = 16 bytes
        out += 16;

        out = scalar::detail::bitpack32Scalar(ex, xn, out, bx);
        out = bitpack128v32(base, out, b);
        return out;
    }

    // Variable-byte exception encoding
    // Format: [count][base bits][vbyte exceptions][positions]
    *out++ = static_cast<unsigned char>(xn);
    out = bitpack128v32(base, out, b);
    out = vbEnc32(ex, xn, out);

    std::memcpy(out, positions, xn);
    out += xn;

    return out;
}

unsigned char * p4Enc128Payload(uint32_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

    // No exceptions - simple bitpacking
    if (bx == 0u)
        return bitpack128v32(in, out, b);

    // Constant block - all values equal
    if (bx == MAX_BITS + 2u)
    {
        storeU32(out, in[0]);
        return out + (b + 7u) / 8u;
    }

    // Exception handling
    return p4Enc128PayloadExceptions(in, n, out, b, bx);
}

}  // namespace

unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out)
{
    using namespace turbopfor::simd::detail;

    if (n == 0u)
        return out;

    unsigned bx = 0;
    unsigned b;

    // Use optimized SIMD version for full 128-element blocks
    if (n == 128u)
        b = p4Bits128(in, &bx);
    else
        b = p4Bits32(in, n, &bx);

    // Fast path for all-zeros: just write header byte
    if (b == 0u && bx == 0u) [[unlikely]]
    {
        *out++ = 0;
        return out;
    }

    writeHeader(out, b, bx);
    return p4Enc128Payload(in, n, out, b, bx);
}

}  // namespace turbopfor::simd

