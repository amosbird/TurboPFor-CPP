#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

const unsigned char * p4DecPayloadExceptions(const unsigned char * in, unsigned n, uint32_t * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    uint64_t bitmap[MAX_VALUES / 64];
    const unsigned bitmap_words = (n + 63u) / 64u;
    unsigned exception_count = 0;

    for (unsigned i = 0; i < bitmap_words; ++i)
    {
        uint64_t word = loadU64Fast(in + i * sizeof(uint64_t));

        if (i == bitmap_words - 1u && (n & 0x3Fu))
            word &= (1ull << (n & 0x3Fu)) - 1ull;

        bitmap[i] = word;

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

    const unsigned char * input_ptr = in + pad8(n);

    uint32_t exception_values[MAX_VALUES];
    input_ptr = bitunpack32Scalar(input_ptr, exception_count, exception_values, bx);
    input_ptr = bitunpack32Scalar(input_ptr, n, out, b);

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
            out[value_index] |= static_cast<uint32_t>(exception_values[exception_index++] << b);

            word &= word - 1ull;
        }
    }

    return input_ptr;
}

} // namespace

const unsigned char * p4Dec32(const unsigned char * in, unsigned n, uint32_t * out)
{
    if (n == 0u)
        return in;

    using namespace turbopfor::scalar::detail;

    const unsigned char * input_ptr = in;
    unsigned base_bits = *input_ptr++;

    if ((base_bits & 0xC0u) == 0u) [[likely]]
        return bitunpack32Scalar(input_ptr, n, out, base_bits);

    if ((base_bits & 0x40u) == 0u)
    {
        unsigned exception_bits = *input_ptr++;
        base_bits &= 0x7Fu;

        if (exception_bits == 0u) [[likely]]
            return bitunpack32Scalar(input_ptr, n, out, base_bits);

        return p4DecPayloadExceptions(input_ptr, n, out, base_bits, exception_bits);
    }

    if ((base_bits & 0xC0u) == 0xC0u) [[unlikely]]
    {
        base_bits &= 0x3Fu;
        const unsigned bytes_stored = (base_bits + 7u) / 8u;

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

        for (unsigned i = 0; i < n; ++i)
            out[i] = constant_value;

        return input_ptr + bytes_stored;
    }

    base_bits &= 0x3Fu;
    const unsigned exception_count = *input_ptr++;

    uint32_t exception_values[MAX_VALUES];
    input_ptr = bitunpack32Scalar(input_ptr, n, out, base_bits);

    input_ptr = vbDec32(input_ptr, exception_count, exception_values);

    for (unsigned i = 0; i < exception_count; ++i)
    {
        const unsigned position = input_ptr[i];
        out[position] |= exception_values[i] << base_bits;
    }

    input_ptr += exception_count;

    return input_ptr;
}

} // namespace turbopfor::scalar
