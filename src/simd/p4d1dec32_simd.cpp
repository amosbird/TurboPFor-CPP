#include "p4_simd.h"
#include "p4_simd_internal.h"

namespace turbopfor::simd
{

namespace
{

const unsigned char * p4D1DecPayload(
    const unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

    if ((b & 0x80u) == 0u)
    {
        // No exceptions - bitunpack then apply delta1
        in = bitunpack32(in, n, out, b);
        applyDelta1(out, n, start);
        return in;
    }

    b &= 0x7Fu;
    if (bx == 0u)
    {
        // Bitmap says no exceptions - bitunpack then apply delta1
        in = bitunpack32(in, n, out, b);
        applyDelta1(out, n, start);
        return in;
    }

    uint64_t bitmap[MAX_VALUES / 64] = {0};
    const unsigned words = (n + 63u) / 64u;
    unsigned num = 0;

    for (unsigned i = 0; i < words; ++i)
    {
        uint64_t word = loadU64(in + i * sizeof(uint64_t));
        if (i == words - 1 && (n & 0x3Fu))
            word &= (1ull << (n & 0x3Fu)) - 1ull;

        bitmap[i] = word;
#if defined(__GNUC__) || defined(__clang__)
        num += static_cast<unsigned>(__builtin_popcountll(word));
#else
        while (word)
        {
            ++num;
            word &= word - 1u;
        }
#endif
    }

    const unsigned char * ip = in + pad8(n);

    uint32_t ex[MAX_VALUES + 64] = {0};
    ip = bitunpack32(ip, num, ex, bx);
    ip = bitunpack32(ip, n, out, b);

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
            out[idx] |= static_cast<uint32_t>(ex[k++] << b);
            word &= word - 1ull;
        }
    }

    applyDelta1(out, n, start);
    return ip;
}

}  // namespace

unsigned char * p4D1Dec32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
{
    using namespace turbopfor::simd::detail;

    if (n == 0u)
        return in;

    unsigned char * ip = in;
    unsigned b = *ip++;

    if ((b & 0xC0u) == 0xC0u)
    {
        b &= 0x3Fu;
        uint32_t value = loadU32(ip);
        if (b < MAX_BITS)
            value &= maskBits(b);

        for (unsigned i = 0; i < n; ++i)
            out[i] = value;

        applyDelta1(out, n, start);
        return ip + (b + 7u) / 8u;
    }

    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        if (b & 0x80u)
            bx = *ip++;

        return const_cast<unsigned char *>(p4D1DecPayload(ip, n, out, start, b, bx));
    }

    unsigned bx = *ip++;
    b &= 0x3Fu;
    uint32_t ex[MAX_VALUES + 64] = {0};
    ip = const_cast<unsigned char *>(bitunpack32(ip, n, out, b));
    ip = const_cast<unsigned char *>(vbDec32(ip, bx, ex));

    for (unsigned i = 0; i < bx; ++i)
        out[ip[i]] |= static_cast<uint32_t>(ex[i] << b);

    ip += bx;
    applyDelta1(out, n, start);
    return ip;
}

}  // namespace turbopfor::simd

