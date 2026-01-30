// AVX2 internal functions for 256v32 format
// Forward some functions to scalar implementations where not performance-critical.
// SIMD-specific optimizations are in bitunpack256v32_simd.cpp.

#include "p4_simd_internal_256v.h"

#include <immintrin.h> // AVX2

namespace turbopfor::simd::detail
{

unsigned char * vbEnc32_256v(const uint32_t * in, unsigned n, unsigned char * out)
{
    return scalar::detail::vbEnc32(in, n, out);
}

const unsigned char * vbDec32_256v(const unsigned char * in, unsigned n, uint32_t * out)
{
    return scalar::detail::vbDec32(const_cast<unsigned char *>(in), n, out);
}

unsigned p4Bits32_256v(const uint32_t * in, unsigned n, unsigned * pbx)
{
    return scalar::detail::p4Bits32(in, n, pbx);
}

void writeHeader_256v(unsigned char *& out, unsigned b, unsigned bx)
{
    scalar::detail::writeHeader(out, b, bx);
}

void applyDelta1_256v(uint32_t * out, unsigned n, uint32_t start)
{
    if (n == 0u)
        return;

    __m256i * op = reinterpret_cast<__m256i *>(out);
    __m256i vs = _mm256_set1_epi32(static_cast<int>(start));
    // cv = [1,2,3,4,5,6,7,8] for delta1 (each element adds its position + 1)
    __m256i cv = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);

    unsigned vec_count = n / 8u;

    for (unsigned i = 0; i < vec_count; ++i)
    {
        __m256i v = _mm256_loadu_si256(op + i);
        vs = mm256_scani_epi32(v, vs, cv);
        _mm256_storeu_si256(op + i, vs);
    }

    // Handle remaining elements
    start = static_cast<uint32_t>(_mm256_extract_epi32(vs, 7));
    for (unsigned j = vec_count * 8u; j < n; ++j)
    {
        start += out[j] + 1u;
        out[j] = start;
    }
}

} // namespace turbopfor::simd::detail
