#include "p4_scalar_bitunpack_impl.h"

namespace turbopfor::scalar::detail
{

unsigned char * bitunpackd1_32Scalar(unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b)
{
    return Bitunpack32ScalarImpl<true>::dispatch(in, n, out, start, b);
}

} // namespace turbopfor::scalar::detail
