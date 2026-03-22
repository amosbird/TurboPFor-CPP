#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

// Encode P4 block with exception handling for 64-bit values
// Same structure as p4Enc32PayloadExceptions but uses 64-bit types and bitpack64Scalar/vbEnc64
unsigned char * p4Enc64PayloadExceptions(uint64_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Create mask for extracting base bits (low b bits)
    const uint64_t base_mask = maskBits64(b);

    // Temporary arrays for splitting values into base and exceptions
    uint64_t base[MAX_VALUES]; // Base values (low b bits)
    uint64_t exceptions[MAX_VALUES]; // Exception values (high bits)
    uint32_t exception_positions[MAX_VALUES]; // Positions of exception values

    // Phase 1: Split values into base and exceptions
    unsigned exception_count = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        const uint64_t value = in[i];
        base[i] = value & base_mask;

        // Check if this value needs more than b bits (is an exception)
        if (value > base_mask)
        {
            exception_positions[exception_count] = i;
            exceptions[exception_count] = value >> b; // Store high bits
            ++exception_count;
        }
    }

    // Phase 2: Encode exceptions based on strategy

    if (bx <= MAX_BITS_64)
    {
        // Strategy: Bitwise patching
        // Format: [bitmap][patch bits][base bits]

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

        // Write patch bits for exceptions (64-bit bitpacking)
        out = bitpack64Scalar(exceptions, exception_count, out, bx);

        // Write base bits for all values (64-bit bitpacking)
        out = bitpack64Scalar(base, n, out, b);

        return out;
    }

    // Strategy: Variable-byte exception encoding
    // Format: [exception_count][base bits][vbyte exceptions][position list]

    *out++ = static_cast<unsigned char>(exception_count);

    // Write base bits for all values (64-bit bitpacking)
    out = bitpack64Scalar(base, n, out, b);

    // Write exception values using 64-bit variable-byte encoding
    out = vbEnc64(exceptions, exception_count, out);

    // Write exception positions
    for (unsigned i = 0; i < exception_count; ++i)
        *out++ = exception_positions[i];

    return out;
}

// Encode P4 block payload for 64-bit values (data after header)
unsigned char * p4Enc64Payload(uint64_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Special case 1: All zeros
    if (b == 0u && bx == 0u)
        return out;

    // Special case 2: Simple bitpacking (no exceptions)
    if (bx == 0u)
        return bitpack64Scalar(in, n, out, b);

    // Special case 3: Constant block (all values equal, non-zero)
    // bx = 66 (MAX_BITS_64 + 2) indicates constant block
    if (bx == MAX_BITS_64 + 2u)
    {
        // Store the constant value using full-width write, advance by needed bytes
        // This matches TurboPFor's: ctou64(out) = in[0]; return out + (b+7)/8;
        storeU64Fast(out, in[0]);
        return out + (b + 7u) / 8u;
    }

    // General case: Exception handling (bitwise patching or variable-byte)
    return p4Enc64PayloadExceptions(in, n, out, b, bx);
}

} // namespace

// Main P4 encoding entry point for 64-bit integer arrays
//
// Same structure as p4Enc32 but for 64-bit values, using:
// - p4Bits64 for analysis
// - writeHeader64 for header (with 63→64 quirk)
// - bitpack64Scalar for bitpacking
// - vbEnc64 for variable-byte exception encoding
unsigned char * p4Enc64(uint64_t * in, unsigned n, unsigned char * out)
{
    using namespace turbopfor::scalar::detail;

    // Analyze input to determine optimal encoding parameters
    unsigned exception_bits = 0;
    unsigned base_bits = p4Bits64(in, n, &exception_bits);

    // Write encoding header (1-2 bytes, with 63→64 clamping)
    writeHeader64(out, base_bits, exception_bits);

    // Write payload (compressed data)
    return p4Enc64Payload(in, n, out, base_bits, exception_bits);
}

} // namespace turbopfor::scalar
