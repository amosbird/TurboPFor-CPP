// SIMD internal functions - forward to scalar implementations
// The scalar implementations are already well-optimized and work correctly.
// SIMD-specific optimizations are in p4bits128_simd.h and bitpack128v32_simd.cpp.

#include "p4_simd_internal.h"

#include <smmintrin.h> // SSE4.1

namespace turbopfor::simd::detail
{

unsigned char * vbEnc32(const uint32_t * in, unsigned n, unsigned char * out)
{
    return scalar::detail::vbEnc32(in, n, out);
}

const unsigned char * vbDec32(const unsigned char * in, unsigned n, uint32_t * out)
{
    return scalar::detail::vbDec32(reinterpret_cast<unsigned char *>(const_cast<unsigned char *>(in)), n, out);
}

unsigned p4Bits32(const uint32_t * in, unsigned n, unsigned * pbx)
{
    return scalar::detail::p4Bits32(in, n, pbx);
}

void writeHeader(unsigned char *& out, unsigned b, unsigned bx)
{
    scalar::detail::writeHeader(out, b, bx);
}

__attribute__((noinline)) void applyDelta1(uint32_t * out, unsigned n, uint32_t start)
{
    if (n == 0u)
        return;

    /// SIMD prefix sum using SSE4.1
    /// Each __m128i holds 4 x 32-bit values
    /// prefix_sum(a,b,c,d) needs: a+start, a+b+start, a+b+c+start, a+b+c+d+start
    ///
    /// Algorithm:
    /// 1. Add +1 to each element (delta1 offset)
    /// 2. Do in-lane prefix sum within each 128-bit vector
    /// 3. Add carry from previous vector (broadcast last element)

    __m128i * op = reinterpret_cast<__m128i *>(out);
    __m128i ones = _mm_set1_epi32(1);
    __m128i carry = _mm_set1_epi32(static_cast<int>(start));

    unsigned vec_count = n / 4u;
    unsigned i = 0;

    for (; i < vec_count; ++i)
    {
        __m128i v = _mm_loadu_si128(op + i);

        // Add +1 to each element (delta1 offset)
        v = _mm_add_epi32(v, ones);

        // In-lane prefix sum:
        // v = [a, b, c, d]
        // Step 1: v + shift_left_4bytes(v) = [a, a+b, b+c, c+d]
        v = _mm_add_epi32(v, _mm_slli_si128(v, 4));
        // Step 2: v + shift_left_8bytes(v) = [a, a+b, a+b+c, a+b+c+d]
        v = _mm_add_epi32(v, _mm_slli_si128(v, 8));

        // Add carry from previous iteration
        v = _mm_add_epi32(v, carry);

        // Store result
        _mm_storeu_si128(op + i, v);

        // Broadcast last element as carry for next iteration
        carry = _mm_shuffle_epi32(v, 0xFF); // [d,d,d,d]
    }

    // Handle remaining elements
    uint32_t scalar_carry = (vec_count > 0) ? out[vec_count * 4u - 1u] : start;

    for (unsigned j = vec_count * 4u; j < n; ++j)
    {
        scalar_carry += out[j] + 1u;
        out[j] = scalar_carry;
    }
}

} // namespace turbopfor::simd::detail
