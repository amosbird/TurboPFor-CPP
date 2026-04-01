// Scalar implementation of P4 encoding for 256v64 format
//
// 256v64 format: encodes 256 x 64-bit values as two consecutive 128v64 P4 blocks.
// This matches TurboPFor's approach where p4enc256v64 is implemented as two
// calls to p4enc128v64, each with its own header and payload.

#include "p4_scalar.h"
#include "p4_scalar_internal.h"

#include <algorithm>

namespace turbopfor::scalar
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

} // namespace turbopfor::scalar
