#include "turbopfor.h"

#include "scalar/p4_scalar.h"
#include "simd/p4_simd.h"

#include <cstdint>

namespace turbopfor
{

// p4enc32 and p4d1dec32 always use scalar versions
unsigned char * p4Enc32(uint32_t * in, unsigned n, unsigned char * out)
{
    return turbopfor::scalar::p4Enc32(in, n, out);
}

unsigned char * p4D1Enc32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start)
{
    return turbopfor::scalar::p4D1Enc32(in, n, out, start);
}

const unsigned char * p4Dec32(const unsigned char * in, unsigned n, uint32_t * out)
{
    return turbopfor::scalar::p4Dec32(in, n, out);
}

const unsigned char * p4D1Dec32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
{
    return turbopfor::scalar::p4D1Dec32(in, n, out, start);
}

// p4enc128v32 and p4dec128v32/p4d1dec128v32 use SIMD if available, otherwise scalar
unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4Enc128v32(in, n, out);
#else
    return turbopfor::scalar::p4Enc128v32(in, n, out);
#endif
}

unsigned char * p4D1Enc128v32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4D1Enc128v32(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Enc128v32(in, n, out, start);
#endif
}

const unsigned char * p4Dec128v32(const unsigned char * in, unsigned n, uint32_t * out)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4Dec128v32(in, n, out);
#else
    return turbopfor::scalar::p4Dec128v32(in, n, out);
#endif
}

const unsigned char * p4D1Dec128v32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4D1Dec128v32(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Dec128v32(in, n, out, start);
#endif
}

// p4enc256v32 and p4dec256v32/p4d1dec256v32 use SIMD (AVX2) if available, otherwise scalar
unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out)
{
#ifdef ENABLE_AVX2
    return turbopfor::simd::p4Enc256v32(in, n, out);
#else
    return turbopfor::scalar::p4Enc256v32(in, n, out);
#endif
}

unsigned char * p4D1Enc256v32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start)
{
#ifdef ENABLE_AVX2
    return turbopfor::simd::p4D1Enc256v32(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Enc256v32(in, n, out, start);
#endif
}

const unsigned char * p4Dec256v32(const unsigned char * in, unsigned n, uint32_t * out)
{
#ifdef ENABLE_AVX2
    return turbopfor::simd::p4Dec256v32(in, n, out);
#else
    return turbopfor::scalar::p4Dec256v32(in, n, out);
#endif
}

const unsigned char * p4D1Dec256v32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
{
#ifdef ENABLE_AVX2
    return turbopfor::simd::p4D1Dec256v32(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Dec256v32(in, n, out, start);
#endif
}

// p4enc64 and p4d1dec64 always use scalar versions
unsigned char * p4Enc64(uint64_t * in, unsigned n, unsigned char * out)
{
    return turbopfor::scalar::p4Enc64(in, n, out);
}

unsigned char * p4D1Enc64(uint64_t * in, unsigned n, unsigned char * out, uint64_t start)
{
    return turbopfor::scalar::p4D1Enc64(in, n, out, start);
}

const unsigned char * p4D1Dec64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    return turbopfor::scalar::p4D1Dec64(in, n, out, start);
}

// p4enc128v64 and p4dec128v64/p4d1dec128v64 use SIMD if available.
// The STO64 pair-swap bug has been fixed (IP32 reordering is now reversed
// in all decode templates via _mm_shuffle_epi32 before output).
// The 64-bit start value handling uses a safe fallback: when start > UINT32_MAX,
// the fused D1 path (which uses 32-bit prefix sum) falls back to SIMD unpack +
// scalar delta1, avoiding truncation.
unsigned char * p4Enc128v64(uint64_t * in, unsigned n, unsigned char * out)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4Enc128v64(in, n, out);
#else
    return turbopfor::scalar::p4Enc128v64(in, n, out);
#endif
}

unsigned char * p4D1Enc128v64(uint64_t * in, unsigned n, unsigned char * out, uint64_t start)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4D1Enc128v64(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Enc128v64(in, n, out, start);
#endif
}

const unsigned char * p4Dec128v64(const unsigned char * in, unsigned n, uint64_t * out)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4Dec128v64(in, n, out);
#else
    return turbopfor::scalar::p4Dec128v64(in, n, out);
#endif
}

const unsigned char * p4D1Dec128v64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4D1Dec128v64(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Dec128v64(in, n, out, start);
#endif
}

// 256v64 functions use SIMD (SSE4.2) if available, otherwise scalar.
// The 256v64 decode wraps 128v64 decode (2× blocks), which now uses the
// corrected SIMD path with the pair-swap fix.
unsigned char * p4Enc256v64(uint64_t * in, unsigned n, unsigned char * out)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4Enc256v64(in, n, out);
#else
    return turbopfor::scalar::p4Enc256v64(in, n, out);
#endif
}

unsigned char * p4D1Enc256v64(uint64_t * in, unsigned n, unsigned char * out, uint64_t start)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4D1Enc256v64(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Enc256v64(in, n, out, start);
#endif
}

const unsigned char * p4Dec256v64(const unsigned char * in, unsigned n, uint64_t * out)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4Dec256v64(in, n, out);
#else
    return turbopfor::scalar::p4Dec256v64(in, n, out);
#endif
}

const unsigned char * p4D1Dec256v64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4D1Dec256v64(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Dec256v64(in, n, out, start);
#endif
}

} // namespace turbopfor
