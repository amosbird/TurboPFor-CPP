// Scalar implementation of 128v32 bitpacking
//
// 128v32 format: 4-lane interleaved horizontal bitpacking for 128 x 32-bit values.
// This format is optimized for SSE SIMD processing where 4 values are packed in parallel.
//
// INPUT:
//   128 values in sequential order: v[0], v[1], v[2], ..., v[127]
//
// PROCESSING:
//   - Values are grouped into 32 groups of 4 consecutive values
//   - Group g contains: v[g*4], v[g*4+1], v[g*4+2], v[g*4+3]
//   - Each value in a group maps to one of 4 SIMD lanes (lane 0, 1, 2, 3)
//   - Within each lane, bits from successive groups are packed horizontally
//
// OUTPUT LAYOUT:
//   - Packed bits are written in 16-byte chunks (4 x 32-bit lanes)
//   - Lane k contains bits from: v[k], v[k+4], v[k+8], v[k+12], v[k+16], ...
//   - Order: [lane0][lane1][lane2][lane3], repeating
//
// EXAMPLE (b=8, values 0-127):
//   Group 0: v[0]=0, v[1]=1, v[2]=2, v[3]=3  -> lane 0,1,2,3 bits [0:7]
//   Group 1: v[4]=4, v[5]=5, v[6]=6, v[7]=7  -> lane 0,1,2,3 bits [8:15]
//   Group 2: v[8]=8, v[9]=9, v[10]=10, v[11]=11 -> lane 0,1,2,3 bits [16:23]
//   Group 3: v[12]=12, v[13]=13, v[14]=14, v[15]=15 -> lane 0,1,2,3 bits [24:31]
//
//   Output bytes 0-15:
//     [0,4,8,12] [1,5,9,13] [2,6,10,14] [3,7,11,15]
//
// Total output size: (128 * b + 7) / 8 bytes

#include "p4_scalar_internal.h"

namespace turbopfor::scalar::detail
{

// Number of elements in 128v32 block
constexpr unsigned V128_BLOCK_SIZE = 128;

// Number of groups (each group has 4 elements for 4 SIMD lanes)
constexpr unsigned V128_GROUP_COUNT = 32;

// Number of SIMD lanes
constexpr unsigned V128_LANE_COUNT = 4;

// Pack 128 x 32-bit values using SIMD-compatible bitpacking format
//
// This matches the SIMD implementation: processes 32 groups of 4 values,
// packing b bits from each value horizontally, writing output when 32-bit
// boundaries are crossed.
//
// Parameters:
//   in: Input array of 128 uint32_t values
//   out: Output buffer (must have space for (128 * b + 7) / 8 bytes)
//   b: Bit width (0-32)
//
// Returns: Pointer to end of packed data
unsigned char * bitpack128v32Scalar(const uint32_t * in, unsigned char * out, unsigned b)
{
    // Special case: b=0 means all values are 0, no output needed
    if (b == 0u)
        return out;

    // Special case: b=32 means no compression, copy with endian conversion
    if (b == 32u)
    {
        copyU32ArrayToLe(out, in, V128_BLOCK_SIZE);
        return out + V128_BLOCK_SIZE * sizeof(uint32_t);
    }

    const uint32_t mask = (1u << b) - 1u;

    // Accumulator for 4 lanes, each lane accumulates bits until 32-bit boundary
    uint32_t ov[V128_LANE_COUNT] = {0, 0, 0, 0};
    unsigned shift = 0;

    // Process 32 groups, each group has 4 values (one per lane)
    for (unsigned g = 0; g < V128_GROUP_COUNT; ++g)
    {
        // Read 4 values for this group (one per lane)
        uint32_t iv[V128_LANE_COUNT];
        for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
        {
            iv[lane] = in[g * V128_LANE_COUNT + lane] & mask;
        }

        // Pack bits into accumulator
        if (shift == 0u)
        {
            // Start fresh accumulator
            for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
            {
                ov[lane] = iv[lane];
            }
        }
        else
        {
            // OR into existing accumulator with shift
            for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
            {
                ov[lane] |= iv[lane] << shift;
            }
        }

        shift += b;

        // Check if we've filled 32 bits (need to write output)
        if (shift >= 32u)
        {
            // Write 4 lanes (16 bytes)
            for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
            {
                storeU32Fast(out, ov[lane]);
                out += sizeof(uint32_t);
            }

            shift -= 32u;

            if (shift > 0u)
            {
                // Carry over high bits that didn't fit
                for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
                {
                    ov[lane] = iv[lane] >> (b - shift);
                }
            }
            else
            {
                // Reset accumulator
                for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
                {
                    ov[lane] = 0;
                }
            }
        }
    }

    // Write any remaining bits
    if (shift > 0u)
    {
        for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
        {
            storeU32Fast(out, ov[lane]);
            out += sizeof(uint32_t);
        }
    }

    return out;
}

// Unpack 128 x 32-bit values from SIMD-compatible bitpacking format
//
// Parameters:
//   in: Input buffer containing packed data
//   out: Output array for 128 uint32_t values
//   b: Bit width (0-32)
//
// Returns: Pointer to end of consumed input data
unsigned char * bitunpack128v32Scalar(unsigned char * in, uint32_t * out, unsigned b)
{
    // Special case: b=0 means all values are 0
    if (b == 0u)
    {
        std::memset(out, 0, V128_BLOCK_SIZE * sizeof(uint32_t));
        return in;
    }

    // Special case: b=32 means no compression, copy with endian conversion
    if (b == 32u)
    {
        copyU32ArrayFromLe(out, in, V128_BLOCK_SIZE);
        return in + V128_BLOCK_SIZE * sizeof(uint32_t);
    }

    const uint32_t mask = (1u << b) - 1u;

    // Input value for 4 lanes
    uint32_t iv[V128_LANE_COUNT] = {0, 0, 0, 0};
    unsigned shift = 0;

    // Track position in input stream
    unsigned char * inp = in;

    // Process 32 groups, each group produces 4 output values
    for (unsigned g = 0; g < V128_GROUP_COUNT; ++g)
    {
        // If shift is 0, we need to load new input
        if (shift == 0u)
        {
            for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
            {
                iv[lane] = loadU32Fast(inp);
                inp += sizeof(uint32_t);
            }
        }

        // Extract b bits for each lane
        uint32_t ov[V128_LANE_COUNT];
        for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
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
                for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
                {
                    iv[lane] = loadU32Fast(inp);
                    inp += sizeof(uint32_t);
                    // OR in high bits from new word
                    ov[lane] |= (iv[lane] << (b - shift)) & mask;
                }
            }
        }

        // Store 4 output values
        for (unsigned lane = 0; lane < V128_LANE_COUNT; ++lane)
        {
            out[g * V128_LANE_COUNT + lane] = ov[lane];
        }
    }

    return in + (V128_BLOCK_SIZE * b + 7u) / 8u;
}

} // namespace turbopfor::scalar::detail
