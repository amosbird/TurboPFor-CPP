#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

unsigned char * p4D1Enc256v64(uint64_t * in, unsigned n, unsigned char * out, uint64_t start)
{
    if (n == 0u)
        return out;

    uint64_t tmp[256 + 8];
    turbopfor::scalar::detail::deltaEnc1(in, n, tmp, start);
    return p4Enc256v64(tmp, n, out);
}

} // namespace turbopfor::scalar
