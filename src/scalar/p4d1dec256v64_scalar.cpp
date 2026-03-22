// Scalar implementation of P4 decoding for 256v64 format (delta1 + non-delta)
//
// 256v64 format: two consecutive 128v64 P4 blocks.
// This matches TurboPFor's approach where p4dec256v64/p4d1dec256v64 processes
// two 128-element blocks sequentially.

#include "p4_scalar.h"
#include "p4_scalar_internal.h"

#include <algorithm>

namespace turbopfor::scalar
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

unsigned char * p4Dec256v64(unsigned char * in, unsigned n, uint64_t * out)
{
    unsigned remaining = n;
    unsigned char * ip = in;
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

} // namespace turbopfor::scalar
