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

unsigned char * p4D1Dec32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
{
    return turbopfor::scalar::p4D1Dec32(in, n, out, start);
}

// p4enc128v32 and p4d1dec128v32 use SIMD if available, otherwise scalar
unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4Enc128v32(in, n, out);
#else
    return turbopfor::scalar::p4Enc128v32(in, n, out);
#endif
}

unsigned char * p4D1Dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4D1Dec128v32(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Dec128v32(in, n, out, start);
#endif
}

// p4enc256v32 and p4d1dec256v32 use SIMD (AVX2) if available, otherwise scalar
unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out)
{
#ifdef ENABLE_AVX2
    return turbopfor::simd::p4Enc256v32(in, n, out);
#else
    return turbopfor::scalar::p4Enc256v32(in, n, out);
#endif
}

unsigned char * p4D1Dec256v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
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

unsigned char * p4D1Dec64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    return turbopfor::scalar::p4D1Dec64(in, n, out, start);
}

// p4enc128v64 and p4d1dec128v64 use SIMD if available, otherwise scalar
// (128v64 is a hybrid format: 128v32 SIMD when b<=32, scalar64 when b>32)
unsigned char * p4Enc128v64(uint64_t * in, unsigned n, unsigned char * out)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4Enc128v64(in, n, out);
#else
    return turbopfor::scalar::p4Enc128v64(in, n, out);
#endif
}

unsigned char * p4Dec128v64(unsigned char * in, unsigned n, uint64_t * out)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4Dec128v64(in, n, out);
#else
    return turbopfor::scalar::p4Dec128v64(in, n, out);
#endif
}

unsigned char * p4D1Dec128v64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
#ifdef ENABLE_SSE42
    return turbopfor::simd::p4D1Dec128v64(in, n, out, start);
#else
    return turbopfor::scalar::p4D1Dec128v64(in, n, out, start);
#endif
}

} // namespace turbopfor
