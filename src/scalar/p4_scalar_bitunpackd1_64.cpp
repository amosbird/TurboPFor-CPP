#include "p4_scalar_bitunpack64_impl.h"

namespace turbopfor::scalar::detail
{

unsigned char * bitunpackd1_64Scalar(unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b)
{
    return Bitunpack64ScalarImpl<true>::dispatch(in, n, out, start, b);
}

} // namespace turbopfor::scalar::detail
