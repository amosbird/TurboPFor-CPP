#include "p4_simd.h"

#include <algorithm>

namespace turbopfor::simd
{

unsigned char * p4D1Dec256v64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    unsigned remaining = n;
    unsigned char * ip = in;
    uint64_t * op = out;
    uint64_t carry = start;

    while (remaining > 0u)
    {
        const unsigned chunk = std::min(remaining, 128u);
        ip = p4D1Dec128v64(ip, chunk, op, carry);
        carry = op[chunk - 1];
        op += chunk;
        remaining -= chunk;
    }

    return ip;
}

} // namespace turbopfor::simd
