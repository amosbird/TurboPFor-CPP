#include "p4_simd.h"
#include "p4_simd_internal.h"

namespace turbopfor::simd
{

namespace
{

/// Handle exceptions path (b & 0x80 set, bx > 0)
/// This is a COLD path - marked noinline to keep it out of the fast path
__attribute__((noinline)) const unsigned char * p4D1Dec128Exceptions(
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

    alignas(16) uint32_t ex[MAX_VALUES + 64];
    /// SIMD loads read 4 elements at a time; unpoison to avoid false positives
    /// when num is not a multiple of 4 (unused lanes are masked out by pshufb)
    TURBOPFOR_MSAN_UNPOISON(ex, sizeof(ex));
    ip = scalar::detail::bitunpack32Scalar(const_cast<unsigned char *>(ip), num, ex, bx);

    const uint32_t * pex = ex;
    ip = bitd1unpack128v32_ex(ip, out, b, start, bitmap, pex);

    return ip;
}

} // namespace

unsigned char * p4D1Dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
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

        applyDelta1(out, n, start);
        return ip + (b + 7u) / 8u;
    }

    // Case 2: PFOR format (b & 0x40 == 0)
    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        if (b & 0x80u)
            bx = *ip++;

        // FAST PATH: No exceptions (most common case)
        // Direct tail call to bitd1unpack128v32 - matches TurboPFor structure
        if (!(b & 0x80u))
        {
            return const_cast<unsigned char *>(bitd1unpack128v32(ip, out, b, start));
        }

        // Has exception bitmap but may have zero exceptions
        b &= 0x7Fu;
        if (bx == 0u)
        {
            return const_cast<unsigned char *>(bitd1unpack128v32(ip, out, b, start));
        }

        // SLOW PATH: Handle exceptions
        return const_cast<unsigned char *>(p4D1Dec128Exceptions(ip, n, out, start, b, bx));
    }

    // Case 3: Variable byte format (b & 0x40 != 0)
    unsigned bx = *ip++;
    b &= 0x3Fu;
    uint32_t ex[MAX_VALUES + 64]; // No initialization needed - vbDec32 writes all values we read
    ip = const_cast<unsigned char *>(bitunpack128v32(ip, out, b));
    ip = const_cast<unsigned char *>(vbDec32(ip, bx, ex));

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
    applyDelta1(out, n, start);
    return ip;
}

} // namespace turbopfor::simd
