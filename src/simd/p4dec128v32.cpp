#include "p4_simd.h"
#include "p4_simd_internal.h"

namespace turbopfor::simd
{

namespace
{

__attribute__((noinline)) const unsigned char * p4Dec128Exceptions(
    const unsigned char * __restrict__ in, unsigned n, uint32_t * __restrict__ out, unsigned b, unsigned bx)
{
    using namespace turbopfor::simd::detail;

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

    alignas(16) uint32_t ex[MAX_VALUES + 64];
    TURBOPFOR_MSAN_UNPOISON(ex, sizeof(ex));
    ip = scalar::detail::bitunpack32Scalar(ip, num, ex, bx);

    const uint32_t * pex = ex;
    ip = bitunpack128v32_ex(ip, out, b, bitmap, pex);

    return ip;
}

} // namespace

const unsigned char * p4Dec128v32(const unsigned char * in, unsigned n, uint32_t * out)
{
    using namespace turbopfor::simd::detail;

    if (n == 0u)
        return in;

    const unsigned char * ip = in;
    unsigned b = *ip++;

    if ((b & 0xC0u) == 0xC0u)
    {
        b &= 0x3Fu;
        uint32_t value = loadU32(ip);
        if (b < MAX_BITS)
            value &= maskBits(b);

        for (unsigned i = 0; i < n; ++i)
            out[i] = value;

        return ip + (b + 7u) / 8u;
    }

    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        if (b & 0x80u)
            bx = *ip++;

        if (!(b & 0x80u))
        {
            return bitunpack128v32(ip, out, b);
        }

        b &= 0x7Fu;
        if (bx == 0u)
        {
            return bitunpack128v32(ip, out, b);
        }

        return p4Dec128Exceptions(ip, n, out, b, bx);
    }

    unsigned bx = *ip++;
    b &= 0x3Fu;
    uint32_t ex[MAX_VALUES + 64];
    ip = bitunpack128v32(ip, out, b);
    ip = vbDec32(ip, bx, ex);

    unsigned i = 0;
    const unsigned bx8 = bx & ~7u;
    for (; i < bx8; i += 8)
    {
        out[ip[i + 0]] |= static_cast<uint32_t>(ex[i + 0] << b);
        out[ip[i + 1]] |= static_cast<uint32_t>(ex[i + 1] << b);
        out[ip[i + 2]] |= static_cast<uint32_t>(ex[i + 2] << b);
        out[ip[i + 3]] |= static_cast<uint32_t>(ex[i + 3] << b);
        out[ip[i + 4]] |= static_cast<uint32_t>(ex[i + 4] << b);
        out[ip[i + 5]] |= static_cast<uint32_t>(ex[i + 5] << b);
        out[ip[i + 6]] |= static_cast<uint32_t>(ex[i + 6] << b);
        out[ip[i + 7]] |= static_cast<uint32_t>(ex[i + 7] << b);
    }
    for (; i < bx; ++i)
        out[ip[i]] |= static_cast<uint32_t>(ex[i] << b);

    ip += bx;
    return ip;
}

} // namespace turbopfor::simd
