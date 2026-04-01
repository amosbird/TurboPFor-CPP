#include "p4_scalar_bitpack64_impl.h"

namespace turbopfor::scalar::detail
{

unsigned char * bitpack64Scalar(const uint64_t * in, unsigned n, unsigned char * out, unsigned b)
{
    return Bitpack64ScalarImpl::dispatch(in, n, out, b);
}

} // namespace turbopfor::scalar::detail
