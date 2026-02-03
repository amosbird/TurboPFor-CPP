#include "bitunpack_avx2_templates.h"
#include "p4_simd.h"
#include "p4_simd_internal.h"
#include "p4_simd_internal_256v.h"

namespace turbopfor::simd
{

namespace
{

template <unsigned Count>
struct BitUnpackAVX2Table
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

/// Handle exceptions path (b & 0x80 set, bx > 0)
/// This is a COLD path - but inlining helps performance
ALWAYS_INLINE const unsigned char * p4D1Dec256Exceptions(
    const unsigned char * __restrict__ in, unsigned n, uint32_t * __restrict__ out, uint32_t start, unsigned b, unsigned bx)
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
    /// SIMD loads read 4/8 elements at a time; unpoison to avoid false positives
    /// when num is not aligned (unused lanes are masked out by shuffle)
    TURBOPFOR_MSAN_UNPOISON(ex, sizeof(ex));
    uint32_t * ex_ptr = ex;
    unsigned rem = num;

    static const auto * table128 = BitUnpackAVX2Table<128>::get();
    static const auto * table32 = BitUnpackAVX2Table<32>::get();

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
        ip = scalar::detail::bitunpack32Scalar(const_cast<unsigned char *>(ip), rem, ex_ptr, bx);
    }

    const uint32_t * pex = ex;
    ip = bitd1unpack256v32_ex(ip, out, b, start, bitmap, pex);

    // Tail handling
    const unsigned vec_count = n / 8u;
    if (n > vec_count * 8u)
    {
        // Read the last written vector to get running sum
        if (vec_count > 0)
        {
            __m256i last_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(out + (vec_count - 1) * 8));
            start = static_cast<uint32_t>(_mm256_extract_epi32(last_v, 7));
        }

        for (unsigned j = vec_count * 8u; j < n; ++j)
        {
            unsigned word_idx = j / 64;
            unsigned bit_idx = j % 64;
            if (bitmap[word_idx] & (1ull << bit_idx))
            {
                out[j] |= static_cast<uint32_t>((*pex++) << b);
            }

            start += out[j] + 1u;
            out[j] = start;
        }
    }

    return ip;
}

} // namespace

unsigned char * p4D1Dec256v32(unsigned char * __restrict__ in, unsigned n, uint32_t * __restrict__ out, uint32_t start)
{
    using namespace turbopfor::simd::detail;

    if (n == 0u)
        return in;

    unsigned char * ip = in;
    unsigned b = *ip++;

    // Case 1: All values equal (b & 0xC0 == 0xC0)
    if ((b & 0xC0u) == 0xC0u)
    {
        b &= 0x3Fu;
        uint32_t value = loadU32(ip);
        if (b < MAX_BITS)
            value &= maskBits(b);

        for (unsigned i = 0; i < n; ++i)
            out[i] = value;

        applyDelta1_256v(out, n, start);
        return ip + (b + 7u) / 8u;
    }

    // Case 2: PFOR format (b & 0x40 == 0)
    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        if (b & 0x80u)
            bx = *ip++;

        // FAST PATH: No exceptions (most common case)
        // Direct tail call to bitd1unpack256v32 - matches TurboPFor structure
        if (!(b & 0x80u))
        {
            return const_cast<unsigned char *>(bitd1unpack256v32(ip, out, b, start));
        }

        // Has exception bitmap but may have zero exceptions
        b &= 0x7Fu;
        if (bx == 0u)
        {
            return const_cast<unsigned char *>(bitd1unpack256v32(ip, out, b, start));
        }

        // SLOW PATH: Handle exceptions
        return const_cast<unsigned char *>(p4D1Dec256Exceptions(ip, n, out, start, b, bx));
    }

    // Case 3: Variable byte format (b & 0x40 != 0)
    unsigned bx = *ip++;
    b &= 0x3Fu;

    // FAST PATH: No exceptions - use fused delta1 unpack directly
    if (bx == 0u)
    {
        return const_cast<unsigned char *>(bitd1unpack256v32(ip, out, b, start));
    }

    // SLOW PATH: Has exceptions - must unpack, merge, then apply delta
    uint32_t ex[MAX_VALUES + 64]; // No initialization needed - vbDec32 writes all values we read
    ip = const_cast<unsigned char *>(bitunpack256v32(ip, out, b));
    ip = const_cast<unsigned char *>(vbDec32_256v(ip, bx, ex));

    // Unrolled exception merge loop (8x unrolled to match TurboPFor)
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
    applyDelta1_256v(out, n, start);
    return ip;
}

} // namespace turbopfor::simd
