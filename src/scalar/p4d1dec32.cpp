#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

// Decode P4 block with exceptions and apply delta1 decoding
//
// P4D1 (P4 + Delta1) combined decoding strategy:
// 1. Decode base values and exceptions from compressed format
// 2. Merge exceptions with base values (bitwise OR with shift)
// 3. Apply delta1 decoding: out[i] = start + cumsum(values[i] + 1)
//
// This function handles bitwise patching exceptions (bx <= 31):
// Format: [bitmap][patch bits][base bits]
//
// Parameters:
//   in: Input compressed data pointer
//   n: Number of values to decode (max 256)
//   out: Output array for decoded values
//   start: Initial value for delta1 decoding
//   b: Base bit width (0-31)
//   bx: Patch bit width (1-31)
//
// Returns: Pointer to next byte after decoded data
unsigned char * p4D1DecPayloadExceptions(unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Phase 1: Load and parse exception bitmap
    // Bitmap has 1 bit per value indicating if it has an exception
    uint64_t bitmap[MAX_VALUES / 64]; // 4 words for 256 bits
    const unsigned bitmap_words = (n + 63u) / 64u;
    unsigned exception_count = 0;

    for (unsigned i = 0; i < bitmap_words; ++i)
    {
        uint64_t word = loadU64Fast(in + i * sizeof(uint64_t));

        // Mask out unused bits in the last word
        if (i == bitmap_words - 1u && (n & 0x3Fu))
            word &= (1ull << (n & 0x3Fu)) - 1ull;

        bitmap[i] = word;

        // Count set bits (number of exceptions)
#if defined(__GNUC__) || defined(__clang__)
        exception_count += static_cast<unsigned>(__builtin_popcountll(word));
#else
        while (word)
        {
            ++exception_count;
            word &= word - 1ull;
        }
#endif
    }

    // Advance past bitmap
    unsigned char * input_ptr = in + pad8(n);

    // Phase 2: Unpack exception patch bits and base bits
    // exception_values: high bits for exceptions
    // out: base values (will be merged with exceptions)
    uint32_t exception_values[MAX_VALUES];
    input_ptr = bitunpack32Scalar(input_ptr, exception_count, exception_values, bx);
    input_ptr = bitunpack32Scalar(input_ptr, n, out, b);

    // Phase 3: Merge exceptions into base values
    // For each bit set in bitmap, OR the corresponding exception value into output
    unsigned exception_index = 0;
    for (unsigned word_index = 0; word_index < bitmap_words; ++word_index)
    {
        uint64_t word = bitmap[word_index];
        while (word)
        {
            // Find position of lowest set bit
#if defined(__GNUC__) || defined(__clang__)
            const unsigned bit_position = static_cast<unsigned>(__builtin_ctzll(word));
#else
            unsigned bit_position = 0;
            while (((word >> bit_position) & 1ull) == 0ull)
                ++bit_position;
#endif

            // Calculate array index and merge exception
            const unsigned value_index = word_index * 64u + bit_position;
            out[value_index] |= static_cast<uint32_t>(exception_values[exception_index++] << b);

            // Clear the lowest set bit
            word &= word - 1ull;
        }
    }

    // Phase 4: Apply delta1 decoding
    // Delta1: decode differences to reconstruct original sequence
    // Original encoding: deltas[i] = original[i] - original[i-1] - 1
    // Decoding: original[i] = original[i-1] + deltas[i] + 1
    for (unsigned i = 0; i < n; ++i)
        out[i] = (start += out[i]) + (i + 1u);

    return input_ptr;
}

} // namespace

// Main P4D1 decoding entry point (P4 decode + Delta1 decode fused)
//
// P4D1 combines P4 decompression with delta1 decoding for better performance.
// Delta1 encoding stores first-order differences (original[i] - original[i-1] - 1).
//
// This function:
// 1. Reads P4 header to determine encoding strategy
// 2. Decodes compressed data using appropriate strategy
// 3. Applies delta1 reconstruction to get original values
//
// Format: [header][payload]
// Header: 1-2 bytes encoding (b, bx) parameters
// Payload: compressed data (format depends on header flags)
//
// Header decoding:
// - 0x00-0x3F: Simple bitpacking (b bits, no exceptions)
// - 0x40-0x7F: Variable-byte exceptions (0x40 | b), then bx byte
// - 0x80-0xBF: Bitwise patching exceptions (0x80 | b), then bx byte
// - 0xC0-0xFF: Constant block (0xC0 | b), all values equal
//
// Parameters:
//   in: Input compressed data
//   n: Number of values to decode (0-256)
//   out: Output array (must have space for n values)
//   start: Initial value for delta1 decoding (previous value in sequence)
//
// Returns: Pointer to next byte after decoded data
unsigned char * p4D1Dec32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
{
    using namespace turbopfor::scalar::detail;

    unsigned char * input_ptr = in;
    unsigned base_bits = *input_ptr++;

    // Fast path: Simple bitpacking (no exceptions)
    // Header format: [0x00-0x3F] = base_bits only
    // Most common case (~70-80% of blocks)
    if ((base_bits & 0xC0u) == 0u) [[likely]]
        return bitunpackd1_32Scalar(input_ptr, n, out, start, base_bits);

    // Second fast path: Bitwise patching exceptions
    // Header format: [0x80 | base_bits][exception_bits]
    // Common case (~15-20% of blocks)
    if ((base_bits & 0x40u) == 0u)
    {
        unsigned exception_bits = *input_ptr++;
        base_bits &= 0x7Fu; // Remove exception flag

        // Special case: bx=0 means no actual exceptions
        if (exception_bits == 0u) [[likely]]
            return bitunpackd1_32Scalar(input_ptr, n, out, start, base_bits);

        return p4D1DecPayloadExceptions(input_ptr, n, out, start, base_bits, exception_bits);
    }

    // Rare path: Constant block
    // Header format: [0xC0 | base_bits], then constant value bytes
    // All values in block are the same (~5-10% of blocks)
    if ((base_bits & 0xC0u) == 0xC0u) [[unlikely]]
    {
        base_bits &= 0x3Fu; // Extract bit width
        const unsigned bytes_stored = (base_bits + 7u) / 8u;

        // Load constant value carefully to avoid over-reading
        // Only (base_bits + 7) / 8 bytes are stored
        uint32_t constant_value;
        switch (bytes_stored)
        {
            case 1u:
                constant_value = input_ptr[0];
                break;
            case 2u:
                constant_value = loadU16Fast(input_ptr);
                break;
            case 3u:
                constant_value = loadU24(input_ptr);
                break;
            case 4u:
                constant_value = loadU32Fast(input_ptr);
                break;
            default:
                __builtin_unreachable();
        }
        constant_value &= (1ull << base_bits) - 1u;

        // Fill output with constant + apply delta1
        // This is effectively: out[i] = start + i * (constant + 1)
        for (unsigned i = 0; i < n; ++i)
            out[i] = (start += constant_value) + (i + 1u);

        return input_ptr + bytes_stored;
    }

    // Rare path: Variable-byte exceptions
    // Header format: [0x40 | base_bits][exception_count]
    // Used when exceptions are large (~1-5% of blocks)
    base_bits &= 0x3Fu; // Extract base bits (remove 0x40 flag)
    const unsigned exception_count = *input_ptr++;

    // Decode base values for all elements
    uint32_t exception_values[MAX_VALUES];
    input_ptr = bitunpack32Scalar(input_ptr, n, out, base_bits);

    // Decode exception values using variable-byte encoding
    input_ptr = vbDec32(input_ptr, exception_count, exception_values);

    // Merge exceptions: position list follows exception values
    for (unsigned i = 0; i < exception_count; ++i)
    {
        const unsigned position = input_ptr[i];
        out[position] |= exception_values[i] << base_bits;
    }

    input_ptr += exception_count;

    // Apply delta1 decoding
    for (unsigned i = 0; i < n; ++i)
        out[i] = (start += out[i]) + (i + 1u);

    return input_ptr;
}

} // namespace turbopfor::scalar
