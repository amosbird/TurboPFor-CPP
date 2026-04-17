#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

unsigned char * p4D1Enc128v32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start)
{
    if (n == 0u)
        return out;

    uint32_t tmp[256 + 8];
    turbopfor::scalar::detail::deltaEnc1(in, n, tmp, start);
    return p4Enc128v32(tmp, n, out);
}

} // namespace turbopfor::scalar
