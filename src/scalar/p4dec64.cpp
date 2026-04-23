#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

const unsigned char * p4Dec64PayloadExceptions(const unsigned char * in, unsigned n, uint64_t * out, unsigned b, unsigned bx)
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

    uint64_t exception_values[MAX_VALUES];
    input_ptr = bitunpack64Scalar(input_ptr, exception_count, exception_values, bx);
    input_ptr = bitunpack64Scalar(input_ptr, n, out, b);

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

            word &= word - 1ull;
        }
    }

    return input_ptr;
}

} // namespace

const unsigned char * p4Dec64(const unsigned char * in, unsigned n, uint64_t * out)
{
    if (n == 0u)
        return in;

    using namespace turbopfor::scalar::detail;

    const unsigned char * input_ptr = in;
    unsigned base_bits = *input_ptr++;

    if ((base_bits & 0xC0u) == 0u) [[likely]]
    {
        if (base_bits == 63u)
            base_bits = 64u;
        return bitunpack64Scalar(input_ptr, n, out, base_bits);
    }

    if ((base_bits & 0x40u) == 0u)
    {
        unsigned exception_bits = *input_ptr++;
        base_bits &= 0x7Fu;

        if (base_bits == 63u)
            base_bits = 64u;

        if (exception_bits == 0u) [[likely]]
            return bitunpack64Scalar(input_ptr, n, out, base_bits);

        return p4Dec64PayloadExceptions(input_ptr, n, out, base_bits, exception_bits);
    }

    if ((base_bits & 0xC0u) == 0xC0u) [[unlikely]]
    {
        base_bits &= 0x3Fu;

        if (base_bits == 63u)
            base_bits = 64u;

        const unsigned bytes_stored = (base_bits + 7u) / 8u;

        uint64_t constant_value = loadU64Fast(input_ptr);
        if (base_bits < 64u)
            constant_value &= (1ull << base_bits) - 1ull;

        for (unsigned i = 0; i < n; ++i)
            out[i] = constant_value;

        return input_ptr + bytes_stored;
    }

    base_bits &= 0x3Fu;

    if (base_bits == 63u)
        base_bits = 64u;

    const unsigned exception_count = *input_ptr++;

    uint64_t exception_values[MAX_VALUES];
    input_ptr = bitunpack64Scalar(input_ptr, n, out, base_bits);

    input_ptr = vbDec64(input_ptr, exception_count, exception_values);

    for (unsigned i = 0; i < exception_count; ++i)
    {
        const unsigned position = input_ptr[i];
        out[position] |= exception_values[i] << base_bits;
    }

    input_ptr += exception_count;

    return input_ptr;
}

} // namespace turbopfor::scalar
