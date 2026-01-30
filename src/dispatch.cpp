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

}  // namespace turbopfor

