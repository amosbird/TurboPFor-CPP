#include "p4_scalar_bitunpack_impl.h"

namespace turbopfor::scalar::detail
{

const unsigned char * bitunpack32Scalar(const unsigned char * in, unsigned n, uint32_t * out, unsigned b)
{
    return Bitunpack32ScalarImpl<false>::dispatch(in, n, out, 0u, b);
}

} // namespace turbopfor::scalar::detail
