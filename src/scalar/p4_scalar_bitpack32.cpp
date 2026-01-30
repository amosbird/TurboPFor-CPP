#include "p4_scalar_bitpack_impl.h"

namespace turbopfor::scalar::detail
{

unsigned char * bitpack32Scalar(const uint32_t * in, unsigned n, unsigned char * out, unsigned b)
{
    return Bitpack32ScalarImpl::dispatch(in, n, out, b);
}

} // namespace turbopfor::scalar::detail
