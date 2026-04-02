#include "p4_simd.h"

#include <algorithm>

namespace turbopfor::simd
{

const unsigned char * p4Dec256v64(const unsigned char * in, unsigned n, uint64_t * out)
{
    unsigned remaining = n;
    const unsigned char * ip = in;
    uint64_t * op = out;

    while (remaining > 0u)
    {
        const unsigned chunk = std::min(remaining, 128u);
        ip = p4Dec128v64(ip, chunk, op);
        op += chunk;
        remaining -= chunk;
    }

    return ip;
}

} // namespace turbopfor::simd
