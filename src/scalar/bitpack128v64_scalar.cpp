// Scalar implementation of 128v64 bitpacking
//
// 128v64 format: HYBRID format for 128 x 64-bit values.
// This matches TurboPFor's bitpack128v64/bitunpack128v64 exactly:
//
//   - When b <= 32: Delegates to bitpack128v32/bitunpack128v32 (4-lane interleaved SIMD format),
//     treating the 64-bit values as 32-bit since they fit in 32 bits.
//     CRITICAL: The input is reordered to match the SIMD shuffle pattern.
//     TurboPFor's IP32 macro loads 4 uint64_t values via two __m128i loads and shuffles
//     them such that within each group of 4 values [v0, v1, v2, v3], the result is
//     [v2, v3, v0, v1] (pairs are swapped). This must be replicated in scalar code.
//   - When b > 32: Delegates to bitpack64/bitunpack64 (scalar horizontal format).
//
// This hybrid approach is used because the 128v32 SIMD format is faster when values
// fit in 32 bits, while for wider values we fall back to the scalar 64-bit format.
//
// Total output size: (128 * b + 7) / 8 bytes

#include "p4_scalar_internal.h"

namespace turbopfor::scalar::detail
{

// Number of elements in 128v64 block
constexpr unsigned V128_64_BLOCK_SIZE = 128;

// Pack 128 x 64-bit values using 128v64 hybrid format
//
// When b <= 32: reorders elements to match SIMD IP32 shuffle, then uses 128v32 format
// When b > 32: uses scalar 64-bit horizontal bitpacking
//
// Parameters:
//   in: Input array of 128 uint64_t values
//   out: Output buffer (must have space for (128 * b + 7) / 8 bytes)
//   b: Bit width (0-64)
//
// Returns: Pointer to end of packed data
unsigned char * bitpack128v64Scalar(const uint64_t * in, unsigned char * out, unsigned b)
{
    if (b <= 32u)
    {
        // When b <= 32, delegate to 128v32 format.
        // We need to replicate the IP32 SIMD shuffle:
        //   IP32 loads 4 uint64_t (32 bytes) via two __m128i loads, then shuffles to
        //   extract the low 32-bit halves, producing [v2, v3, v0, v1] from [v0, v1, v2, v3].
        //
        // For each group of 4 input uint64_t values, swap the two pairs:
        //   input[0],input[1],input[2],input[3] → tmp[0]=input[2], tmp[1]=input[3], tmp[2]=input[0], tmp[3]=input[1]
        uint32_t tmp[V128_64_BLOCK_SIZE];
        for (unsigned i = 0; i < V128_64_BLOCK_SIZE; i += 4)
        {
            tmp[i + 0] = static_cast<uint32_t>(in[i + 2]);
            tmp[i + 1] = static_cast<uint32_t>(in[i + 3]);
            tmp[i + 2] = static_cast<uint32_t>(in[i + 0]);
            tmp[i + 3] = static_cast<uint32_t>(in[i + 1]);
        }

        unsigned char * pout = out + (V128_64_BLOCK_SIZE * b + 7u) / 8u;
        bitpack128v32Scalar(tmp, out, b);
        return pout;
    }

    // When b > 32, use scalar 64-bit horizontal bitpacking
    return bitpack64Scalar(in, V128_64_BLOCK_SIZE, out, b);
}

// Unpack 128 x 64-bit values from 128v64 hybrid format
//
// When b <= 32: unpacks from 128v32 format into temp array, then reverses the SIMD shuffle
// When b > 32: uses scalar 64-bit horizontal bitunpacking
//
// Parameters:
//   in: Input buffer containing packed data
//   out: Output array for 128 uint64_t values
//   b: Bit width (0-64)
//
// Returns: Pointer to end of consumed input data
const unsigned char * bitunpack128v64Scalar(const unsigned char * in, uint64_t * out, unsigned b)
{
    if (b <= 32u)
    {
        // When b <= 32, unpack from 128v32 format into a temp 32-bit array,
        // then reverse the SIMD IP32 shuffle and widen to 64-bit.
        //
        // The 128v32 data was stored with the shuffle [v2,v3,v0,v1],
        // so after unpacking we have tmp = [v2,v3,v0,v1] for each group of 4.
        // We need to reverse: out[0]=tmp[2], out[1]=tmp[3], out[2]=tmp[0], out[3]=tmp[1]
        uint32_t tmp[V128_64_BLOCK_SIZE];
        const unsigned char * ip = bitunpack128v32Scalar(in, tmp, b);

        for (unsigned i = 0; i < V128_64_BLOCK_SIZE; i += 4)
        {
            out[i + 0] = static_cast<uint64_t>(tmp[i + 2]);
            out[i + 1] = static_cast<uint64_t>(tmp[i + 3]);
            out[i + 2] = static_cast<uint64_t>(tmp[i + 0]);
            out[i + 3] = static_cast<uint64_t>(tmp[i + 1]);
        }

        return ip;
    }

    // When b > 32, use scalar 64-bit horizontal bitunpacking
    return bitunpack64Scalar(in, V128_64_BLOCK_SIZE, out, b);
}

} // namespace turbopfor::scalar::detail
