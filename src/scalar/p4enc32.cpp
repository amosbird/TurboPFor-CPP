#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

// Encode P4 block with exception handling (bitwise patching or variable-byte exceptions)
//
// P4 encoding strategy:
// 1. Split each value into base (low b bits) and exception (high bits if needed)
// 2. Store all base values using b bits per value (bitpacked)
// 3. Store exceptions using one of two strategies:
//    - Bitwise patching (bx < 32): bitmap + patch bits
//    - Variable-byte (bx == 33): vbyte encoded exceptions + position list
//
// Parameters:
//   in: Input array of n values
//   n: Number of values (max 256)
//   out: Output buffer
//   b: Base bit width (1-31)
//   bx: Exception strategy:
//       0: no exceptions
//       1-31: bitwise patching with patch_bits = bx
//       33: variable-byte exception encoding
//
// Returns: Pointer to end of encoded data
unsigned char * p4Enc32PayloadExceptions(uint32_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Create mask for extracting base bits (low b bits)
    const uint32_t base_mask = (1u << b) - 1u;

    // Temporary arrays for splitting values into base and exceptions
    uint32_t base[MAX_VALUES]; // Base values (low b bits)
    uint32_t exceptions[MAX_VALUES]; // Exception values (high bits)
    uint32_t exception_positions[MAX_VALUES]; // Positions of exception values

    // Phase 1: Split values into base and exceptions
    // Use unsigned for loop counter to avoid movzbl instructions
    unsigned exception_count = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        const uint32_t value = in[i];
        base[i] = static_cast<uint32_t>(value & base_mask);

        // Check if this value needs more than b bits (is an exception)
        if (value > base_mask)
        {
            exception_positions[exception_count] = i;
            exceptions[exception_count] = value >> b; // Store high bits
            ++exception_count;
        }
    }

    // Phase 2: Encode exceptions based on strategy

    if (bx <= MAX_BITS)
    {
        // Strategy: Bitwise patching
        // Format: [bitmap][patch bits][base bits]
        //
        // Bitmap: 1 bit per value indicating if it has an exception
        // Patch bits: bx bits per exception value
        // Base bits: b bits per value

        // Build exception bitmap (1 bit = has exception, 0 bit = no exception)
        uint64_t bitmap[MAX_VALUES / 64]; // 4 uint64_t for 256 bits
        std::memset(bitmap, 0, sizeof(bitmap));

        for (unsigned i = 0; i < exception_count; ++i)
        {
            const unsigned pos = exception_positions[i];
            bitmap[pos >> 6] |= 1ull << (pos & 0x3Fu); // Set bit at position
        }

        // Write bitmap (pad to byte boundary)
        const unsigned bitmap_words = (n + 63u) / 64u;
        for (unsigned i = 0; i < bitmap_words; ++i)
        {
            storeU64Fast(out + i * sizeof(uint64_t), bitmap[i]);
        }

        out += pad8(n); // Advance by bitmap size in bytes

        // Write patch bits for exceptions
        out = bitpack32Scalar(exceptions, exception_count, out, bx);

        // Write base bits for all values
        out = bitpack32Scalar(base, n, out, b);

        return out;
    }

    // Strategy: Variable-byte exception encoding
    // Format: [exception_count][base bits][vbyte exceptions][position list]
    //
    // exception_count: 1 byte, number of exceptions
    // base bits: b bits per value
    // vbyte exceptions: variable-byte encoded exception values
    // position list: 1 byte per exception indicating its position

    *out++ = static_cast<unsigned char>(exception_count);

    // Write base bits for all values
    out = bitpack32Scalar(base, n, out, b);

    // Write exception values using variable-byte encoding
    out = vbEnc32(exceptions, exception_count, out);

    // Write exception positions
    for (unsigned i = 0; i < exception_count; ++i)
        *out++ = exception_positions[i];

    return out;
}

// Encode P4 block payload (data after header)
//
// Handles different encoding strategies based on bx value:
// - b=0: All zeros, no data needed
// - bx=0: Simple bitpacking, no exceptions
// - bx=1-31: Bitwise patching (patch_bits = bx)
// - bx=33: Variable-byte exception encoding
// - bx=34: Constant block (all values equal)
// Note: bx=32 is reserved/unused (separates patching from special encodings)
//
// Parameters:
//   in: Input array
//   n: Number of values
//   out: Output buffer
//   b: Base bit width
//   bx: Exception strategy (see p4Bits32 for encoding)
//
// Returns: Pointer to end of encoded data
unsigned char * p4Enc32Payload(uint32_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Special case 1: All zeros
    // No data needed, header alone indicates all zeros
    if (b == 0u && bx == 0u)
        return out;

    // Special case 2: Simple bitpacking (no exceptions)
    // All values fit in b bits, just pack them
    if (bx == 0u)
        return bitpack32Scalar(in, n, out, b);

    // Special case 2: Constant block (all values equal, non-zero)
    // bx = 34 (MAX_BITS + 2) indicates constant block
    // Format: Just store the constant value once
    if (bx == MAX_BITS + 2u)
    {
        if (b == 32u) [[unlikely]]
        {
            storeU32Fast(out, in[0]);
            return out + 4;
        }

        // Advance by bytes needed for b bits
        // Store only the bytes needed for b bits (avoids garbage writes)
        // Mask the value to b bits to ensure clean encoding
        const uint32_t masked_value = in[0] & ((1u << b) - 1u);
        const uint32_t bytes_needed = (b + 7u) / 8u;

        // Write bytes in little-endian order (lowest byte first)
        for (unsigned i = 0; i < bytes_needed; ++i)
        {
            out[i] = static_cast<unsigned char>(masked_value >> (i * 8u));
        }
        return out + bytes_needed;
    }

    // General case: Exception handling (bitwise patching or variable-byte)
    return p4Enc32PayloadExceptions(in, n, out, b, bx);
}

} // namespace

// Main P4 encoding entry point for 32-bit integer arrays
//
// P4 (Patched Frame-of-Reference) encoding:
// 1. Analyze input to find optimal base bit width and exception strategy
// 2. Write header (1-2 bytes) indicating encoding parameters
// 3. Write payload (compressed data)
//
// Format: [header][payload]
// Header: 1-2 bytes encoding (b, bx) - see writeHeader
// Payload: depends on encoding strategy - see p4Enc32Payload
//
// Parameters:
//   in: Input array of uint32_t values
//   n: Number of values (0-256)
//   out: Output buffer (must have enough space)
//
// Returns: Pointer to end of encoded data
unsigned char * p4Enc32(uint32_t * in, unsigned n, unsigned char * out)
{
    using namespace turbopfor::scalar::detail;

    // Analyze input to determine optimal encoding parameters
    unsigned exception_bits = 0;
    unsigned base_bits = p4Bits32(in, n, &exception_bits);

    // Write encoding header (1-2 bytes)
    writeHeader(out, base_bits, exception_bits);

    // Write payload (compressed data)
    return p4Enc32Payload(in, n, out, base_bits, exception_bits);
}

} // namespace turbopfor::scalar
