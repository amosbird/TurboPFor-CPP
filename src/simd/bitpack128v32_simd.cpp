#include "p4_simd_internal.h"

namespace turbopfor::simd::detail
{

// Optimized general implementation with 4x unrolling
// Minimizes branch overhead and improves instruction-level parallelism
static unsigned char * bitpack128v32_general(const uint32_t * in, unsigned char * out, unsigned b)
{
    const __m128i mv = _mm_set1_epi32(static_cast<int>((1u << b) - 1u));
    __m128i * op = reinterpret_cast<__m128i *>(out);

    __m128i ov = _mm_setzero_si128();
    unsigned shift = 0;

    // Process 4 groups at a time (16 elements per iteration)
    for (unsigned g = 0; g < 32u; g += 4u)
    {
        __m128i iv0 = _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(in + g * 4u)), mv);
        __m128i iv1 = _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (g + 1) * 4u)), mv);
        __m128i iv2 = _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (g + 2) * 4u)), mv);
        __m128i iv3 = _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (g + 3) * 4u)), mv);

        // Process iv0
        if (shift == 0u)
            ov = iv0;
        else
            ov = _mm_or_si128(ov, _mm_slli_epi32(iv0, static_cast<int>(shift)));
        shift += b;
        if (shift >= 32u)
        {
            _mm_storeu_si128(op++, ov);
            shift -= 32u;
            ov = (shift > 0u) ? _mm_srli_epi32(iv0, static_cast<int>(b - shift)) : _mm_setzero_si128();
        }

        // Process iv1
        ov = _mm_or_si128(ov, _mm_slli_epi32(iv1, static_cast<int>(shift)));
        shift += b;
        if (shift >= 32u)
        {
            _mm_storeu_si128(op++, ov);
            shift -= 32u;
            ov = (shift > 0u) ? _mm_srli_epi32(iv1, static_cast<int>(b - shift)) : _mm_setzero_si128();
        }

        // Process iv2
        ov = _mm_or_si128(ov, _mm_slli_epi32(iv2, static_cast<int>(shift)));
        shift += b;
        if (shift >= 32u)
        {
            _mm_storeu_si128(op++, ov);
            shift -= 32u;
            ov = (shift > 0u) ? _mm_srli_epi32(iv2, static_cast<int>(b - shift)) : _mm_setzero_si128();
        }

        // Process iv3
        ov = _mm_or_si128(ov, _mm_slli_epi32(iv3, static_cast<int>(shift)));
        shift += b;
        if (shift >= 32u)
        {
            _mm_storeu_si128(op++, ov);
            shift -= 32u;
            ov = (shift > 0u) ? _mm_srli_epi32(iv3, static_cast<int>(b - shift)) : _mm_setzero_si128();
        }
    }

    if (shift > 0u)
    {
        _mm_storeu_si128(op++, ov);
    }

    return out + (128u * b + 7u) / 8u;
}

// Specialized implementation for b=1 - fully unrolled
static unsigned char * bitpack128v32_b1(const uint32_t * in, unsigned char * out)
{
    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i mv = _mm_set1_epi32(1);
    __m128i ov = _mm_setzero_si128();

#define PACK1(g) \
    { \
        __m128i iv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (g) * 4u)); \
        iv = _mm_and_si128(iv, mv); \
        ov = _mm_or_si128(ov, _mm_slli_epi32(iv, g)); \
    }

    PACK1(0);
    PACK1(1);
    PACK1(2);
    PACK1(3);
    PACK1(4);
    PACK1(5);
    PACK1(6);
    PACK1(7);
    PACK1(8);
    PACK1(9);
    PACK1(10);
    PACK1(11);
    PACK1(12);
    PACK1(13);
    PACK1(14);
    PACK1(15);
    PACK1(16);
    PACK1(17);
    PACK1(18);
    PACK1(19);
    PACK1(20);
    PACK1(21);
    PACK1(22);
    PACK1(23);
    PACK1(24);
    PACK1(25);
    PACK1(26);
    PACK1(27);
    PACK1(28);
    PACK1(29);
    PACK1(30);
    PACK1(31);

#undef PACK1

    _mm_storeu_si128(op, ov);
    return out + 16;
}

// Specialized implementation for b=2 - fully unrolled
static unsigned char * bitpack128v32_b2(const uint32_t * in, unsigned char * out)
{
    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i mv = _mm_set1_epi32(3);
    __m128i ov;

#define PACK2(g, sh) \
    { \
        __m128i iv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (g) * 4u)); \
        iv = _mm_and_si128(iv, mv); \
        if (sh == 0) \
            ov = iv; \
        else \
            ov = _mm_or_si128(ov, _mm_slli_epi32(iv, sh)); \
    }

    // First output (groups 0-15)
    PACK2(0, 0);
    PACK2(1, 2);
    PACK2(2, 4);
    PACK2(3, 6);
    PACK2(4, 8);
    PACK2(5, 10);
    PACK2(6, 12);
    PACK2(7, 14);
    PACK2(8, 16);
    PACK2(9, 18);
    PACK2(10, 20);
    PACK2(11, 22);
    PACK2(12, 24);
    PACK2(13, 26);
    PACK2(14, 28);
    PACK2(15, 30);
    _mm_storeu_si128(op++, ov);

    // Second output (groups 16-31)
    PACK2(16, 0);
    PACK2(17, 2);
    PACK2(18, 4);
    PACK2(19, 6);
    PACK2(20, 8);
    PACK2(21, 10);
    PACK2(22, 12);
    PACK2(23, 14);
    PACK2(24, 16);
    PACK2(25, 18);
    PACK2(26, 20);
    PACK2(27, 22);
    PACK2(28, 24);
    PACK2(29, 26);
    PACK2(30, 28);
    PACK2(31, 30);
    _mm_storeu_si128(op++, ov);

#undef PACK2

    return out + 32;
}

// Specialized implementation for b=4 - fully unrolled
static unsigned char * bitpack128v32_b4(const uint32_t * in, unsigned char * out)
{
    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i mv = _mm_set1_epi32(0xF);
    __m128i ov;

#define PACK4(g, sh) \
    { \
        __m128i iv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (g) * 4u)); \
        iv = _mm_and_si128(iv, mv); \
        if (sh == 0) \
            ov = iv; \
        else \
            ov = _mm_or_si128(ov, _mm_slli_epi32(iv, sh)); \
    }

    // 4 outputs, 8 groups each
    PACK4(0, 0);
    PACK4(1, 4);
    PACK4(2, 8);
    PACK4(3, 12);
    PACK4(4, 16);
    PACK4(5, 20);
    PACK4(6, 24);
    PACK4(7, 28);
    _mm_storeu_si128(op++, ov);

    PACK4(8, 0);
    PACK4(9, 4);
    PACK4(10, 8);
    PACK4(11, 12);
    PACK4(12, 16);
    PACK4(13, 20);
    PACK4(14, 24);
    PACK4(15, 28);
    _mm_storeu_si128(op++, ov);

    PACK4(16, 0);
    PACK4(17, 4);
    PACK4(18, 8);
    PACK4(19, 12);
    PACK4(20, 16);
    PACK4(21, 20);
    PACK4(22, 24);
    PACK4(23, 28);
    _mm_storeu_si128(op++, ov);

    PACK4(24, 0);
    PACK4(25, 4);
    PACK4(26, 8);
    PACK4(27, 12);
    PACK4(28, 16);
    PACK4(29, 20);
    PACK4(30, 24);
    PACK4(31, 28);
    _mm_storeu_si128(op++, ov);

#undef PACK4

    return out + 64;
}

// Specialized implementation for b=8 - fully unrolled
static unsigned char * bitpack128v32_b8(const uint32_t * in, unsigned char * out)
{
    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i mv = _mm_set1_epi32(0xFF);
    __m128i ov;

#define PACK8(g, sh) \
    { \
        __m128i iv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (g) * 4u)); \
        iv = _mm_and_si128(iv, mv); \
        if (sh == 0) \
            ov = iv; \
        else \
            ov = _mm_or_si128(ov, _mm_slli_epi32(iv, sh)); \
    }

    // 8 outputs, 4 groups each
    for (unsigned i = 0; i < 8; ++i)
    {
        unsigned base = i * 4;
        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (base + 0) * 4u));
        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (base + 1) * 4u));
        __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (base + 2) * 4u));
        __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (base + 3) * 4u));
        v0 = _mm_and_si128(v0, mv);
        v1 = _mm_and_si128(v1, mv);
        v2 = _mm_and_si128(v2, mv);
        v3 = _mm_and_si128(v3, mv);
        ov = v0;
        ov = _mm_or_si128(ov, _mm_slli_epi32(v1, 8));
        ov = _mm_or_si128(ov, _mm_slli_epi32(v2, 16));
        ov = _mm_or_si128(ov, _mm_slli_epi32(v3, 24));
        _mm_storeu_si128(op++, ov);
    }

#undef PACK8

    return out + 128;
}

// Specialized implementation for b=16 - fully unrolled
static unsigned char * bitpack128v32_b16(const uint32_t * in, unsigned char * out)
{
    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i mv = _mm_set1_epi32(0xFFFF);

    // 16 outputs, 2 groups each
    for (unsigned i = 0; i < 16; ++i)
    {
        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (i * 2 + 0) * 4u));
        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + (i * 2 + 1) * 4u));
        v0 = _mm_and_si128(v0, mv);
        v1 = _mm_and_si128(v1, mv);
        __m128i ov = _mm_or_si128(v0, _mm_slli_epi32(v1, 16));
        _mm_storeu_si128(op++, ov);
    }

    return out + 256;
}

unsigned char * bitpack128v32(const uint32_t * in, unsigned char * out, unsigned b)
{
    if (b == 0u)
        return out;

    if (b == 32u)
    {
        std::memcpy(out, in, 128u * sizeof(uint32_t));
        return out + 128u * sizeof(uint32_t);
    }

    // Use specialized functions for power-of-2 bit widths
    switch (b)
    {
        case 1:
            return bitpack128v32_b1(in, out);
        case 2:
            return bitpack128v32_b2(in, out);
        case 4:
            return bitpack128v32_b4(in, out);
        case 8:
            return bitpack128v32_b8(in, out);
        case 16:
            return bitpack128v32_b16(in, out);
        default:
            return bitpack128v32_general(in, out, b);
    }
}

} // namespace turbopfor::simd::detail
