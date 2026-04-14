#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

const unsigned char * p4Dec256PayloadBitmap(
    const unsigned char * in, unsigned n, uint32_t * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

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
        while (word)
        {
            ++exception_count;
            word &= word - 1ull;
        }
#endif
    }

    const unsigned char * ip = in + pad8(n);

    uint32_t exceptions[MAX_VALUES + 64];
    ip = bitunpack32Scalar(ip, exception_count, exceptions, bx);

    ip = bitunpack256v32Scalar(ip, out, b);

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
            out[idx] |= static_cast<uint32_t>(exceptions[k++] << b);
            word &= word - 1ull;
        }
    }

    return ip;
}

const unsigned char * p4Dec256Payload(
    const unsigned char * in, unsigned n, uint32_t * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    if ((b & 0x80u) == 0u)
    {
        return bitunpack256v32Scalar(in, out, b);
    }

    b &= 0x7Fu;

    if (bx == 0u)
    {
        return bitunpack256v32Scalar(in, out, b);
    }

    return p4Dec256PayloadBitmap(in, n, out, b, bx);
}

} // namespace

const unsigned char * p4Dec256v32(const unsigned char * in, unsigned n, uint32_t * out)
{
    using namespace turbopfor::scalar::detail;

    if (n == 0u)
        return in;

    const unsigned char * ip = in;
    unsigned b = *ip++;

    if ((b & 0xC0u) == 0xC0u)
    {
        b &= 0x3Fu;

        uint32_t value = loadU32Fast(ip);
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

        return p4Dec256Payload(ip, n, out, b, bx);
    }

    unsigned bx = *ip++;
    b &= 0x3Fu;

    ip = bitunpack256v32Scalar(ip, out, b);

    uint32_t exceptions[MAX_VALUES + 64];
    ip = vbDec32(ip, bx, exceptions);

    for (unsigned i = 0; i < bx; ++i)
        out[ip[i]] |= static_cast<uint32_t>(exceptions[i] << b);

    ip += bx;

    return ip;
}

} // namespace turbopfor::scalar
