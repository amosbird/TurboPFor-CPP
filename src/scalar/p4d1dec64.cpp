#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

// Decode P4 block with bitwise patching exceptions and apply delta1 decoding (64-bit)
//
// Same structure as p4D1DecPayloadExceptions (32-bit) but uses 64-bit types
// and bitunpack64Scalar/bitunpackd1_64Scalar.
//
// Format: [bitmap][patch bits][base bits]
const unsigned char * p4D1Dec64PayloadExceptions(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Phase 1: Load and parse exception bitmap
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
    const unsigned char * input_ptr = in + pad8(n);

    // Phase 2: Unpack exception patch bits and base bits (both 64-bit)
    uint64_t exception_values[MAX_VALUES];
    input_ptr = bitunpack64Scalar(input_ptr, exception_count, exception_values, bx);
    input_ptr = bitunpack64Scalar(input_ptr, n, out, b);

    // Phase 3: Merge exceptions into base values
    unsigned exception_index = 0;
    for (unsigned word_index = 0; word_index < bitmap_words; ++word_index)
    {
        uint64_t word = bitmap[word_index];
        while (word)
        {
#if defined(__GNUC__) || defined(__clang__)
            const unsigned bit_position = static_cast<unsigned>(__builtin_ctzll(word));
#else
            unsigned bit_position = 0;
            while (((word >> bit_position) & 1ull) == 0ull)
                ++bit_position;
#endif

            const unsigned value_index = word_index * 64u + bit_position;
            out[value_index] |= exception_values[exception_index++] << b;

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

// Main P4D1 decoding entry point for 64-bit values (P4 decode + Delta1 decode fused)
//
// Same structure as p4D1Dec32 but for 64-bit values, with:
// - 63→64 bit mapping in header decoding
// - 64-bit constant values (up to 8 bytes)
// - bitunpackd1_64Scalar / bitunpack64Scalar for unpacking
// - vbDec64 for variable-byte exception decoding
const unsigned char * p4D1Dec64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    if (n == 0u)
        return in;

    using namespace turbopfor::scalar::detail;

    const unsigned char * input_ptr = in;
    unsigned base_bits = *input_ptr++;

    // Fast path: Simple bitpacking (no exceptions)
    // Header format: [0x00-0x3F] = base_bits only
    if ((base_bits & 0xC0u) == 0u) [[likely]]
    {
        // 64-bit quirk: 63 in header means 64
        if (base_bits == 63u)
            base_bits = 64u;
        return bitunpackd1_64Scalar(input_ptr, n, out, start, base_bits);
    }

    // Second fast path: Bitwise patching exceptions
    // Header format: [0x80 | base_bits][exception_bits]
    if ((base_bits & 0x40u) == 0u)
    {
        unsigned exception_bits = *input_ptr++;
        base_bits &= 0x7Fu; // Remove exception flag

        // 64-bit quirk: 63 in header means 64
        // When b==63 is mapped to 64, bx should be 0 (from p4Bits64),
        // but handle it defensively
        if (base_bits == 63u)
            base_bits = 64u;

        // Special case: bx=0 means no actual exceptions
        if (exception_bits == 0u) [[likely]]
            return bitunpackd1_64Scalar(input_ptr, n, out, start, base_bits);

        return p4D1Dec64PayloadExceptions(input_ptr, n, out, start, base_bits, exception_bits);
    }

    // Rare path: Constant block
    // Header format: [0xC0 | base_bits], then constant value bytes
    if ((base_bits & 0xC0u) == 0xC0u) [[unlikely]]
    {
        base_bits &= 0x3Fu; // Extract bit width

        // 64-bit quirk: 63 in header means 64
        if (base_bits == 63u)
            base_bits = 64u;

        const unsigned bytes_stored = (base_bits + 7u) / 8u;

        // Load constant value — TurboPFor reads a full 8-byte word then masks
        // This is safe because TurboPFor always overallocates
        uint64_t constant_value = loadU64Fast(input_ptr);
        if (base_bits < 64u)
            constant_value &= (1ull << base_bits) - 1ull;

        // Fill output with constant + apply delta1
        for (unsigned i = 0; i < n; ++i)
            out[i] = (start += constant_value) + (i + 1u);

        return input_ptr + bytes_stored;
    }

    // Rare path: Variable-byte exceptions
    // Header format: [0x40 | base_bits][exception_count]
    base_bits &= 0x3Fu; // Extract base bits (remove 0x40 flag)

    // 64-bit quirk: 63 in header means 64
    if (base_bits == 63u)
        base_bits = 64u;

    const unsigned exception_count = *input_ptr++;

    // Decode base values for all elements (64-bit bitunpack)
    uint64_t exception_values[MAX_VALUES];
    input_ptr = bitunpack64Scalar(input_ptr, n, out, base_bits);

    // Decode exception values using 64-bit variable-byte encoding
    input_ptr = vbDec64(input_ptr, exception_count, exception_values);

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
