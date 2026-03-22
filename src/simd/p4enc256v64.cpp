#include "p4_simd.h"

#include <algorithm>

namespace turbopfor::simd
{

unsigned char * p4Enc256v64(uint64_t * in, unsigned n, unsigned char * out)
{
    unsigned remaining = n;
    uint64_t * ip = in;
    unsigned char * op = out;

    while (remaining > 0u)
    {
        const unsigned chunk = std::min(remaining, 128u);
        op = p4Enc128v64(ip, chunk, op);
        ip += chunk;
        remaining -= chunk;
    }

    return op;
}

} // namespace turbopfor::simd
