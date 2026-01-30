#include <immintrin.h>
#include "bitunpack_avx2_templates.h"
#include "p4_simd.h"
#include "p4_simd_internal_256v.h"

namespace turbopfor::simd::detail
{
namespace
{

using namespace turbopfor::simd::detail;

// Wrappers for 256v32 (Count = 256)
template <unsigned B>
ALWAYS_INLINE const unsigned char * bitunpack_256v_wrapper(const unsigned char * __restrict in, uint32_t * __restrict out)
{
    __m256i sv = _mm256_setzero_si256();
    return bitunpack_avx2_entry<B, 256, false, false>(in, out, sv, nullptr, nullptr);
}

template <unsigned B>
ALWAYS_INLINE const unsigned char * bitd1unpack_256v_wrapper(const unsigned char * __restrict in, uint32_t * __restrict out, __m256i sv)
{
    return bitunpack_avx2_entry<B, 256, true, false>(in, out, sv, nullptr, nullptr);
}

template <unsigned B>
ALWAYS_INLINE const unsigned char * bitd1unpack_256v_ex_wrapper(
    const unsigned char * __restrict in, uint32_t * __restrict out, __m256i sv, const uint64_t * bitmap, const uint32_t *& pex)
{
    return bitunpack_avx2_entry<B, 256, true, true>(in, out, sv, bitmap, pex);
}

} // namespace

// Dispatch Tables
#define GEN_TABLE_AVX2(FUNC) \
    FUNC<0>, FUNC<1>, FUNC<2>, FUNC<3>, FUNC<4>, FUNC<5>, FUNC<6>, FUNC<7>, FUNC<8>, FUNC<9>, FUNC<10>, FUNC<11>, FUNC<12>, FUNC<13>, \
        FUNC<14>, FUNC<15>, FUNC<16>, FUNC<17>, FUNC<18>, FUNC<19>, FUNC<20>, FUNC<21>, FUNC<22>, FUNC<23>, FUNC<24>, FUNC<25>, FUNC<26>, \
        FUNC<27>, FUNC<28>, FUNC<29>, FUNC<30>, FUNC<31>, FUNC<32>

typedef const unsigned char * (*bitd1unpack256v32_func)(const unsigned char *, uint32_t *, __m256i);
static const bitd1unpack256v32_func bitd1unpack_table_256v[33] = {GEN_TABLE_AVX2(bitd1unpack_256v_wrapper)};

typedef const unsigned char * (*bitunpack256v32_func)(const unsigned char *, uint32_t *);
static const bitunpack256v32_func bitunpack_table_256v[33] = {GEN_TABLE_AVX2(bitunpack_256v_wrapper)};

typedef const unsigned char * (*bitd1unpack256v32_ex_func)(const unsigned char *, uint32_t *, __m256i, const uint64_t *, const uint32_t *&);
static const bitd1unpack256v32_ex_func bitd1unpack_ex_table_256v[33] = {GEN_TABLE_AVX2(bitd1unpack_256v_ex_wrapper)};

const unsigned char * bitd1unpack256v32(const unsigned char * in, uint32_t * out, unsigned b, uint32_t start)
{
    __m256i sv = _mm256_set1_epi32(static_cast<int>(start));
    return bitd1unpack_table_256v[b](in, out, sv);
}

const unsigned char * bitunpack256v32(const unsigned char * in, uint32_t * out, unsigned b)
{
    return bitunpack_table_256v[b](in, out);
}

const unsigned char *
bitd1unpack256v32_ex(const unsigned char * in, uint32_t * out, unsigned b, uint32_t start, const uint64_t * bitmap, const uint32_t *& pex)
{
    __m256i sv = _mm256_set1_epi32(static_cast<int>(start));
    return bitd1unpack_ex_table_256v[b](in, out, sv, bitmap, pex);
}

} // namespace turbopfor::simd::detail
