#include "p4_simd.h"
#include "p4_simd_internal_256v.h"

namespace turbopfor::simd
{

namespace
{

// Optimized payload encoding for 256 elements with exceptions
// Uses single-pass algorithm to build base values, exceptions, and bitmap simultaneously
unsigned char * p4Enc256PayloadExceptions(uint32_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

    const uint32_t msk = (b >= 32u) ? 0xFFFFFFFFu : ((1u << b) - 1u);

    // Aligned arrays for better cache performance
    alignas(32) uint32_t base[256];
    alignas(32) uint32_t ex[256];
    alignas(32) uint64_t bitmap[4] = {0, 0, 0, 0};  // 256 bits = 4 x 64-bit words
    alignas(32) uint8_t positions[256];

    // Single-pass: extract base values and track exceptions
    unsigned xn = 0;

    // Process with AVX2 SIMD for base extraction and exception detection
    const __m256i msk_vec = _mm256_set1_epi32(static_cast<int>(msk));

    for (unsigned i = 0; i < n; i += 8)
    {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
        __m256i base_v = _mm256_and_si256(v, msk_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(base + i), base_v);

        // Check for exceptions: v > msk means high bits are set
        __m256i high_bits = _mm256_andnot_si256(msk_vec, v);
        __m256i has_exc = _mm256_cmpeq_epi32(high_bits, _mm256_setzero_si256());
        // has_exc is 0xFFFFFFFF where NO exception, 0 where exception exists

        int exc_mask = _mm256_movemask_ps(_mm256_castsi256_ps(has_exc));
        // exc_mask bit is 1 where NO exception, 0 where exception

        // Process exceptions (inverted mask)
        unsigned exc_bits = static_cast<unsigned>(~exc_mask) & 0xFFu;
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
        std::memcpy(out, bitmap, 32);  // 256 bits = 32 bytes
        out += 32;

        out = scalar::detail::bitpack32Scalar(ex, xn, out, bx);
        out = bitpack256v32(base, out, b);
        return out;
    }

    // Variable-byte exception encoding
    // Format: [count][base bits][vbyte exceptions][positions]
    *out++ = static_cast<unsigned char>(xn);
    out = bitpack256v32(base, out, b);
    out = vbEnc32_256v(ex, xn, out);

    std::memcpy(out, positions, xn);
    out += xn;

    return out;
}

unsigned char * p4Enc256Payload(uint32_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

    // No exceptions - simple bitpacking
    if (bx == 0u)
        return bitpack256v32(in, out, b);

    // Constant block - all values equal
    if (bx == MAX_BITS + 2u)
    {
        storeU32(out, in[0]);
        return out + (b + 7u) / 8u;
    }

    // Exception handling
    return p4Enc256PayloadExceptions(in, n, out, b, bx);
}

}  // namespace

unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out)
{
    using namespace turbopfor::simd::detail;

    if (n == 0u)
        return out;

    unsigned bx = 0;
    unsigned b = p4Bits32_256v(in, n, &bx);

    // Fast path for all-zeros: just write header byte
    if (b == 0u && bx == 0u) [[unlikely]]
    {
        *out++ = 0;
        return out;
    }

    writeHeader_256v(out, b, bx);
    return p4Enc256Payload(in, n, out, b, bx);
}

}  // namespace turbopfor::simd
