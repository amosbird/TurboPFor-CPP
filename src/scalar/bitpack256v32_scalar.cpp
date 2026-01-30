// Scalar implementation of 256v32 bitpacking
//
// 256v32 format: 8-lane interleaved horizontal bitpacking for 256 x 32-bit values.
// This format is optimized for AVX2 SIMD processing where 8 values are packed in parallel.
//
// INPUT:
//   256 values in sequential order: v[0], v[1], v[2], ..., v[255]
//
// PROCESSING:
//   - Values are grouped into 32 groups of 8 consecutive values
//   - Group g contains: v[g*8], v[g*8+1], v[g*8+2], ..., v[g*8+7]
//   - Each value in a group maps to one of 8 AVX2 lanes (lane 0-7)
//   - Within each lane, bits from successive groups are packed horizontally
//
// OUTPUT LAYOUT:
//   - Packed bits are written in 32-byte chunks (8 x 32-bit lanes)
//   - Lane k contains bits from: v[k], v[k+8], v[k+16], v[k+24], v[k+32], ...
//   - Order: [lane0][lane1][lane2]...[lane7], repeating
//
// EXAMPLE (b=8, values 0-255):
//   Group 0: v[0-7]   -> lane 0-7 bits [0:7]
//   Group 1: v[8-15]  -> lane 0-7 bits [8:15]
//   Group 2: v[16-23] -> lane 0-7 bits [16:23]
//   Group 3: v[24-31] -> lane 0-7 bits [24:31]
//
//   Output bytes 0-31:
//     [0,8,16,24] [1,9,17,25] [2,10,18,26] ... [7,15,23,31]
//
// Total output size: (256 * b + 7) / 8 bytes

#include "p4_scalar_internal.h"

namespace turbopfor::scalar::detail
{

// Number of elements in 256v32 block
constexpr unsigned V256_BLOCK_SIZE = 256;

// Number of groups (each group has 8 elements for 8 SIMD lanes)
constexpr unsigned V256_GROUP_COUNT = 32;

// Number of SIMD lanes (AVX2 = 256 bits = 8 x 32-bit)
constexpr unsigned V256_LANE_COUNT = 8;

// Pack 256 x 32-bit values using AVX2-compatible bitpacking format
//
// This matches the AVX2 implementation: processes 32 groups of 8 values,
// packing b bits from each value horizontally, writing output when 32-bit
// boundaries are crossed.
//
// Parameters:
//   in: Input array of 256 uint32_t values
//   out: Output buffer (must have space for (256 * b + 7) / 8 bytes)
//   b: Bit width (0-32)
//
// Returns: Pointer to end of packed data
unsigned char * bitpack256v32Scalar(const uint32_t * in, unsigned char * out, unsigned b)
{
    // Special case: b=0 means all values are 0, no output needed
    if (b == 0u)
        return out;

    // Special case: b=32 means no compression, copy with endian conversion
    if (b == 32u)
    {
        copyU32ArrayToLe(out, in, V256_BLOCK_SIZE);
        return out + V256_BLOCK_SIZE * sizeof(uint32_t);
    }

    const uint32_t mask = (1u << b) - 1u;

    // Accumulator for 8 lanes, each lane accumulates bits until 32-bit boundary
    uint32_t ov[V256_LANE_COUNT] = {0};
    unsigned shift = 0;

    // Process 32 groups, each group has 8 values (one per lane)
    for (unsigned g = 0; g < V256_GROUP_COUNT; ++g)
    {
        // Read 8 values for this group (one per lane)
        uint32_t iv[V256_LANE_COUNT];
        for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
        {
            iv[lane] = in[g * V256_LANE_COUNT + lane] & mask;
        }

        // Pack bits into accumulator
        if (shift == 0u)
        {
            // Start fresh accumulator
            for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
            {
                ov[lane] = iv[lane];
            }
        }
        else
        {
            // OR into existing accumulator with shift
            for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
            {
                ov[lane] |= iv[lane] << shift;
            }
        }

        shift += b;

        // Check if we've filled 32 bits (need to write output)
        if (shift >= 32u)
        {
            // Write 8 lanes (32 bytes)
            for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
            {
                storeU32Fast(out, ov[lane]);
                out += sizeof(uint32_t);
            }

            shift -= 32u;

            if (shift > 0u)
            {
                // Carry over high bits that didn't fit
                for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
                {
                    ov[lane] = iv[lane] >> (b - shift);
                }
            }
            else
            {
                // Reset accumulator
                for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
                {
                    ov[lane] = 0;
                }
            }
        }
    }

    // Write any remaining bits
    if (shift > 0u)
    {
        for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
        {
            storeU32Fast(out, ov[lane]);
            out += sizeof(uint32_t);
        }
    }

    return out;
}

// Unpack 256 x 32-bit values from AVX2-compatible bitpacking format
//
// Parameters:
//   in: Input buffer containing packed data
//   out: Output array for 256 uint32_t values
//   b: Bit width (0-32)
//
// Returns: Pointer to end of consumed input data
unsigned char * bitunpack256v32Scalar(unsigned char * in, uint32_t * out, unsigned b)
{
    // Special case: b=0 means all values are 0
    if (b == 0u)
    {
        std::memset(out, 0, V256_BLOCK_SIZE * sizeof(uint32_t));
        return in;
    }

    // Special case: b=32 means no compression, copy with endian conversion
    if (b == 32u)
    {
        copyU32ArrayFromLe(out, in, V256_BLOCK_SIZE);
        return in + V256_BLOCK_SIZE * sizeof(uint32_t);
    }

    const uint32_t mask = (1u << b) - 1u;

    // Input value for 8 lanes
    uint32_t iv[V256_LANE_COUNT] = {0};
    unsigned shift = 0;

    // Track position in input stream
    unsigned char * inp = in;

    // Process 32 groups, each group produces 8 output values
    for (unsigned g = 0; g < V256_GROUP_COUNT; ++g)
    {
        // If shift is 0, we need to load new input
        if (shift == 0u)
        {
            for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
            {
                iv[lane] = loadU32Fast(inp);
                inp += sizeof(uint32_t);
            }
        }

        // Extract b bits for each lane
        uint32_t ov[V256_LANE_COUNT];
        for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
        {
            ov[lane] = (iv[lane] >> shift) & mask;
        }

        shift += b;

        // Check if we need more bits from next input word
        if (shift >= 32u)
        {
            shift -= 32u;

            if (shift > 0u)
            {
                // Load next input words and get remaining high bits
                for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
                {
                    iv[lane] = loadU32Fast(inp);
                    inp += sizeof(uint32_t);
                    // OR in high bits from new word
                    ov[lane] |= (iv[lane] << (b - shift)) & mask;
                }
            }
        }

        // Store 8 output values
        for (unsigned lane = 0; lane < V256_LANE_COUNT; ++lane)
        {
            out[g * V256_LANE_COUNT + lane] = ov[lane];
        }
    }

    return in + (V256_BLOCK_SIZE * b + 7u) / 8u;
}

} // namespace turbopfor::scalar::detail
