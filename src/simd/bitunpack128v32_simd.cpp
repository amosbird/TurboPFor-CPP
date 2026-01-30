#include <cstdint>
#include <immintrin.h>
#include "bitunpack_sse_templates.h"
#include "p4_simd.h"
#include "p4_simd_internal.h"

namespace turbopfor::simd::detail
{

// --- bitunpack128v32 (Standard) ---
using BitUnpackFn = const unsigned char * (*)(const unsigned char *, uint32_t *);

template <unsigned B>
const unsigned char * bitunpack128v32_impl(const unsigned char * in, uint32_t * out)
{
    __m128i sv = _mm_setzero_si128(); // Unused
    return bitunpack_sse_entry<B, 128, false, false>(in, out, sv, nullptr, nullptr);
}

// Helper macro to generate dispatch table
#define GEN_TABLE(FUNC) \
    FUNC<0>, FUNC<1>, FUNC<2>, FUNC<3>, FUNC<4>, FUNC<5>, FUNC<6>, FUNC<7>, FUNC<8>, FUNC<9>, FUNC<10>, FUNC<11>, FUNC<12>, FUNC<13>, \
        FUNC<14>, FUNC<15>, FUNC<16>, FUNC<17>, FUNC<18>, FUNC<19>, FUNC<20>, FUNC<21>, FUNC<22>, FUNC<23>, FUNC<24>, FUNC<25>, FUNC<26>, \
        FUNC<27>, FUNC<28>, FUNC<29>, FUNC<30>, FUNC<31>, FUNC<32>

const unsigned char * bitunpack128v32(const unsigned char * __restrict in, uint32_t * __restrict out, unsigned b)
{
    static const BitUnpackFn funcs[] = {GEN_TABLE(bitunpack128v32_impl)};
    return funcs[b](in, out);
}

// --- bitd1unpack128v32 (Delta) ---
using BitD1UnpackFn = const unsigned char * (*)(const unsigned char *, uint32_t *, __m128i);

template <unsigned B>
const unsigned char * bitd1unpack128v32_impl(const unsigned char * in, uint32_t * out, __m128i sv)
{
    return bitunpack_sse_entry<B, 128, true, false>(in, out, sv, nullptr, nullptr);
}

const unsigned char * bitd1unpack128v32(const unsigned char * __restrict in, uint32_t * __restrict out, unsigned b, uint32_t start)
{
    static const BitD1UnpackFn funcs[] = {GEN_TABLE(bitd1unpack128v32_impl)};
    __m128i sv = _mm_set1_epi32(start);
    return funcs[b](in, out, sv);
}

// --- bitd1unpack128v32_ex (Fused Delta + Exceptions) ---
// New function for fused exception handling
using BitD1UnpackExFn = const unsigned char * (*)(const unsigned char *, uint32_t *, __m128i, const uint64_t *, const uint32_t *&);

template <unsigned B>
const unsigned char *
bitd1unpack128v32_ex_impl(const unsigned char * in, uint32_t * out, __m128i sv, const uint64_t * bitmap, const uint32_t *& pex)
{
    return bitunpack_sse_entry<B, 128, true, true>(in, out, sv, bitmap, pex);
}

const unsigned char * bitd1unpack128v32_ex(
    const unsigned char * __restrict in,
    uint32_t * __restrict out,
    unsigned b,
    uint32_t start,
    const uint64_t * bitmap,
    const uint32_t *& pex)
{
    static const BitD1UnpackExFn funcs[] = {GEN_TABLE(bitd1unpack128v32_ex_impl)};
    __m128i sv = _mm_set1_epi32(start);
    return funcs[b](in, out, sv, bitmap, pex);
}

} // namespace turbopfor::simd::detail
