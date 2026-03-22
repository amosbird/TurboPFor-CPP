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

// p4enc128v64 uses SIMD if available (encode is correct and byte-compatible).
// p4dec128v64 and p4d1dec128v64 use scalar only — the SIMD decode path has a
// pre-existing pair-swap bug: the STO64 fused unpack does NOT reverse the IP32
// reordering, producing output in [v2,v3,v0,v1] order instead of [v0,v1,v2,v3].
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
    return turbopfor::scalar::p4Dec128v64(in, n, out);
}

unsigned char * p4D1Dec128v64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    return turbopfor::scalar::p4D1Dec128v64(in, n, out, start);
}

// 256v64 functions always use scalar implementation.
//
// SIMD cannot be used here due to a pre-existing architectural issue in the
// SIMD 128v64 decode path: the STO64 fused unpack does NOT reverse the IP32
// pair-swap reordering ([v2,v3,v0,v1] → [v0,v1,v2,v3]). This causes:
//   1. p4Dec128v64 (SIMD) returns pair-swapped output vs scalar
//   2. p4D1Dec128v64 (SIMD) prefix sums pair-swapped deltas, producing wrong results
//
// The SIMD 128v64 *encode* is correct (verified byte-identical with scalar/C).
// The bug only affects decode paths. Until the STO64 templates are fixed to
// reverse the pair-swap, all 256v64 (and 128v64) decode must go through scalar.
//
// Additionally, the SIMD D1 path truncates the 64-bit start value to 32 bits
// via _mm_set1_epi32. This was partially fixed (fallback to scalar when
// start > UINT32_MAX), but the pair-swap issue is the primary blocker.
unsigned char * p4Enc256v64(uint64_t * in, unsigned n, unsigned char * out)
{
    return turbopfor::scalar::p4Enc256v64(in, n, out);
}

unsigned char * p4Dec256v64(unsigned char * in, unsigned n, uint64_t * out)
{
    return turbopfor::scalar::p4Dec256v64(in, n, out);
}

unsigned char * p4D1Dec256v64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    return turbopfor::scalar::p4D1Dec256v64(in, n, out, start);
}

} // namespace turbopfor
