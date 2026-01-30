// Scalar implementation of P4 encoding for 128v32 format
//
// P4 (Patched Frame-of-Reference) encoding with 128-element 4-lane interleaved bitpacking.
// This format is designed for SSE SIMD processing but this is a scalar reference implementation
// for correctness verification and fallback on non-SIMD platforms.
//
// Format overview:
//   [header][payload]
//
// Header (1-2 bytes):
//   - Encodes base bit width (b) and exception strategy (bx)
//   - Same format as standard P4 encoding
//
// Payload depends on encoding strategy:
//   - b=0, bx=0: All zeros, no payload
//   - bx=0: Simple bitpacking using 128v32 format
//   - bx=34 (constant): Single value stored
//   - bx=1-31: Bitwise patching (bitmap + patch bits + base bits)
//   - bx=33: Variable-byte exceptions (count + base bits + vbyte values + positions)
//
// Key difference from standard P4 (p4enc32):
//   - Uses 128v32 bitpacking (bitpack128v32Scalar) instead of scalar (bitpack32Scalar)
//   - 128v32 format: 4-lane interleaved horizontal packing (see bitpack128v32_scalar.cpp)
//   - Always processes exactly 128 elements (32 groups x 4 lanes)

#include "p4_scalar.h"
#include "p4_scalar_internal.h"
#include "p4bits128_scalar.h"

namespace turbopfor::scalar
{

namespace
{

// Encode P4 block payload with exception handling for 128v32 format
//
// This handles the two exception encoding strategies:
// 1. Bitwise patching (bx <= 31): bitmap + patch bits + base bits
// 2. Variable-byte (bx == 33): count + base bits + vbyte values + positions
//
// Parameters:
//   in: Input array of n values (n <= 128, typically 128)
//   n: Number of values
//   out: Output buffer
//   b: Base bit width (bits per value for base data)
//   bx: Exception strategy indicator
//
// Returns: Pointer to end of encoded data
unsigned char * p4Enc128v32PayloadExceptions(uint32_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Create mask for extracting base bits (low b bits of each value)
    const uint64_t base_mask = (b >= 32u) ? 0xFFFFFFFFull : ((1ull << b) - 1ull);

    // Temporary arrays for exception handling
    // base[]: stores low b bits of each value (to be bitpacked)
    // exceptions[]: stores high bits of exception values (bits above b)
    // exception_positions[]: stores indices of values that exceed b bits
    uint32_t base[MAX_VALUES + 32] = {0}; // Extra padding for SIMD safety
    uint32_t exceptions[MAX_VALUES + 32] = {0};
    uint64_t bitmap[MAX_VALUES / 64] = {0}; // 1 bit per value: 1 = has exception
    unsigned exception_positions[MAX_VALUES] = {0};

    // Phase 1: Scan input and separate base values from exceptions
    //
    // For each value:
    // - Extract low b bits as base value
    // - If value > base_mask, it's an exception: record position and high bits
    unsigned exception_count = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        // Record position first (branchless technique from TurboPFor)
        // This avoids branch misprediction by always writing position,
        // but only incrementing count when it's actually an exception
        exception_positions[exception_count] = i;
        exception_count += (in[i] > base_mask) ? 1u : 0u;

        // Store base value (low b bits) - always needed
        base[i] = static_cast<uint32_t>(in[i] & base_mask);
    }

    // Phase 2: Build exception data structures
    //
    // For bitwise patching, we need:
    // - bitmap: marks which positions have exceptions
    // - exceptions[]: the high bits (value >> b) for each exception
    for (unsigned i = 0; i < exception_count; ++i)
    {
        const unsigned pos = exception_positions[i];

        // Set bit in bitmap at exception position
        bitmap[pos >> 6] |= 1ull << (pos & 0x3Fu);

        // Store exception value (high bits above b)
        exceptions[i] = static_cast<uint32_t>(in[pos] >> b);
    }

    // Phase 3: Encode based on exception strategy
    if (bx <= MAX_BITS)
    {
        // Strategy 1: Bitwise patching
        //
        // Format: [bitmap][patch bits][base bits]
        //
        // bitmap: n bits (padded to byte boundary), 1 = position has exception
        // patch bits: exception_count values, each bx bits (horizontal bitpacking)
        // base bits: n values using 128v32 vertical bitpacking

        // Write exception bitmap
        const unsigned bitmap_words = (n + 63u) / 64u;
        for (unsigned i = 0; i < bitmap_words; ++i)
        {
            storeU64Fast(out + i * sizeof(uint64_t), bitmap[i]);
        }

        out += pad8(n); // Advance by bitmap size (padded to bytes)

        // Write patch bits using horizontal bitpacking
        // Note: exceptions use standard horizontal format, not 128v32
        out = bitpack32Scalar(exceptions, exception_count, out, bx);

        // Write base values using 128v32 bitpacking
        out = bitpack128v32Scalar(base, out, b);

        return out;
    }

    // Strategy 2: Variable-byte exception encoding
    //
    // Format: [exception_count][base bits][vbyte exceptions][position list]
    //
    // exception_count: 1 byte
    // base bits: n values using 128v32 vertical bitpacking
    // vbyte exceptions: variable-byte encoded exception values
    // position list: exception_count bytes, each is position in [0, n)

    *out++ = static_cast<unsigned char>(exception_count);

    // Write base values using 128v32 bitpacking
    out = bitpack128v32Scalar(base, out, b);

    // Write exception values using variable-byte encoding
    out = vbEnc32(exceptions, exception_count, out);

    // Write exception positions (1 byte each, since n <= 256)
    for (unsigned i = 0; i < exception_count; ++i)
        *out++ = static_cast<unsigned char>(exception_positions[i]);

    return out;
}

// Encode P4 block payload for 128v32 format
//
// Handles all encoding strategies based on (b, bx) determined by p4Bits32:
//   - b=0, bx=0: All zeros
//   - bx=0: Simple bitpacking (no exceptions)
//   - bx=34: Constant block (all values equal)
//   - bx=1-31: Bitwise patching
//   - bx=33: Variable-byte exceptions
//
// Parameters:
//   in: Input array of n values
//   n: Number of values (should be 128 for 128v32 format)
//   out: Output buffer
//   b: Base bit width
//   bx: Exception strategy
//
// Returns: Pointer to end of encoded data
unsigned char * p4Enc128v32Payload(uint32_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Case 1: No exceptions needed
    if (bx == 0u)
    {
        // Simple 128v32 bitpacking, no exception handling
        return bitpack128v32Scalar(in, out, b);
    }

    // Case 2: Constant block (all values are identical)
    // bx = MAX_BITS + 2 = 34 indicates constant encoding
    if (bx == MAX_BITS + 2u)
    {
        // Store single value using minimal bytes
        // For b=32: store full 4 bytes
        // For b<32: store only (b+7)/8 bytes
        storeU32(out, in[0]);
        return out + (b + 7u) / 8u;
    }

    // Case 3: Exception handling required
    return p4Enc128v32PayloadExceptions(in, n, out, b, bx);
}

} // namespace

// Main P4 encoding entry point for 128v32 format
//
// Encodes up to 128 uint32_t values using P4 encoding with 128v32 SIMD-parallel bitpacking.
// This format is optimized for SIMD decoding where 4 lanes process data in parallel.
//
// Encoding steps:
// 1. Analyze input to determine optimal bit width and exception strategy
// 2. Write header (1-2 bytes)
// 3. Write payload using appropriate strategy
//
// Parameters:
//   in: Input array of uint32_t values
//   n: Number of values (typically 128, max 256 for P4 format)
//   out: Output buffer (must have sufficient space)
//
// Returns: Pointer past end of encoded data
//
// Binary compatibility: Output is bit-identical to TurboPFor's p4enc128v32()
unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out)
{
    using namespace turbopfor::scalar::detail;

    // Empty input produces no output
    if (n == 0u)
        return out;

    // Analyze input to find optimal encoding parameters
    // b: base bit width (minimum bits to represent most values)
    // bx: exception strategy (0=none, 1-31=patching, 33=vbyte, 34=constant)
    unsigned bx = 0;
    unsigned b;

    // Use optimized version for full 128-element blocks
    if (n == 128u)
        b = p4Bits128(in, &bx);
    else
        b = p4Bits32(in, n, &bx);

    // Write P4 header (1-2 bytes encoding b and bx)
    writeHeader(out, b, bx);

    // Write payload using determined strategy
    return p4Enc128v32Payload(in, n, out, b, bx);
}

} // namespace turbopfor::scalar
