#include "p4_scalar_bitunpack64_impl.h"

namespace turbopfor::scalar::detail
{

const unsigned char * bitunpack64Scalar(const unsigned char * in, unsigned n, uint64_t * out, unsigned b)
{
    return Bitunpack64ScalarImpl<false>::dispatch(in, n, out, 0ull, b);
}

} // namespace turbopfor::scalar::detail
