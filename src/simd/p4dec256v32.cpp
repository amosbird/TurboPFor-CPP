#include "bitunpack_avx2_templates.h"
#include "p4_simd.h"
#include "p4_simd_internal.h"
#include "p4_simd_internal_256v.h"

namespace turbopfor::simd
{

namespace
{

template <unsigned Count>
struct BitUnpackAVX2TableNoDelta
{
    using Func = const unsigned char * (*)(const unsigned char *, uint32_t *);

    template <unsigned B>
    static const unsigned char * impl(const unsigned char * in, uint32_t * out)
    {
        __m256i sv = _mm256_setzero_si256();
        return detail::bitunpack_avx2_entry<B, Count, false, false>(in, out, sv, nullptr, nullptr);
    }

    static const Func * get()
    {
        static const Func table[]
            = {impl<0>,  impl<1>,  impl<2>,  impl<3>,  impl<4>,  impl<5>,  impl<6>,  impl<7>,  impl<8>,  impl<9>,  impl<10>,
               impl<11>, impl<12>, impl<13>, impl<14>, impl<15>, impl<16>, impl<17>, impl<18>, impl<19>, impl<20>, impl<21>,
               impl<22>, impl<23>, impl<24>, impl<25>, impl<26>, impl<27>, impl<28>, impl<29>, impl<30>, impl<31>, impl<32>};
        return table;
    }
};

ALWAYS_INLINE const unsigned char * p4Dec256Exceptions(
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

    alignas(32) uint32_t ex[MAX_VALUES + 64];
    TURBOPFOR_MSAN_UNPOISON(ex, sizeof(ex));
    uint32_t * ex_ptr = ex;
    unsigned rem = num;

    static const auto * table128 = BitUnpackAVX2TableNoDelta<128>::get();
    static const auto * table32 = BitUnpackAVX2TableNoDelta<32>::get();

    while (rem >= 128)
    {
        ip = table128[bx](ip, ex_ptr);
        ex_ptr += 128;
        rem -= 128;
    }

    while (rem >= 32)
    {
        ip = table32[bx](ip, ex_ptr);
        ex_ptr += 32;
        rem -= 32;
    }

    if (rem > 0)
    {
        ip = scalar::detail::bitunpack32Scalar(ip, rem, ex_ptr, bx);
    }

    const uint32_t * pex = ex;
    ip = bitunpack256v32_ex(ip, out, b, bitmap, pex);

    return ip;
}

} // namespace

const unsigned char * p4Dec256v32(const unsigned char * __restrict__ in, unsigned n, uint32_t * __restrict__ out)
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
            return bitunpack256v32(ip, out, b);
        }

        b &= 0x7Fu;
        if (bx == 0u)
        {
            return bitunpack256v32(ip, out, b);
        }

        return p4Dec256Exceptions(ip, n, out, b, bx);
    }

    unsigned bx = *ip++;
    b &= 0x3Fu;

    if (bx == 0u)
    {
        return bitunpack256v32(ip, out, b);
    }

    uint32_t ex[MAX_VALUES + 64];
    ip = bitunpack256v32(ip, out, b);
    ip = vbDec32_256v(ip, bx, ex);

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
