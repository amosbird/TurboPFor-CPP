// Scalar implementation of 256v64 bitpacking
//
// 256v64 format: HYBRID format for 256 x 64-bit values.
//
//   - When b <= 32: Delegates to bitpack256v32/bitunpack256v32.
//   - When b > 32:  Delegates to scalar bitpack64/bitunpack64 horizontal format.

#include "p4_scalar_internal.h"

namespace turbopfor::scalar::detail
{

constexpr unsigned V256_64_BLOCK_SIZE = 256;

namespace
{
constexpr unsigned HALF = 128;
}

unsigned char * bitpack256v64Scalar(const uint64_t * in, unsigned char * out, unsigned b)
{
    if (b <= 32u)
    {
        uint64_t tmp0[HALF];
        uint64_t tmp1[HALF];
        for (unsigned i = 0; i < HALF; ++i)
        {
            tmp0[i] = in[i];
            tmp1[i] = in[HALF + i];
        }

        out = bitpack128v64Scalar(tmp0, out, b);
        out = bitpack128v64Scalar(tmp1, out, b);
        return out;
    }

    return bitpack64Scalar(in, V256_64_BLOCK_SIZE, out, b);
}

const unsigned char * bitunpack256v64Scalar(const unsigned char * in, uint64_t * out, unsigned b)
{
    if (b <= 32u)
    {
        in = bitunpack128v64Scalar(in, out, b);
        in = bitunpack128v64Scalar(in, out + HALF, b);
        return in;
    }

    return bitunpack64Scalar(in, V256_64_BLOCK_SIZE, out, b);
}

} // namespace turbopfor::scalar::detail
