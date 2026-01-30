#pragma once

#include "p4_scalar_bitunpack_impl.h" // for choose_block_size, gcd_u32, etc.
#include "p4_scalar_internal.h"

#include <utility>

namespace turbopfor::scalar::detail
{

// Store partial bytes at the end of output
template <unsigned R>
static TURBOPFOR_ALWAYS_INLINE void store_partial(unsigned char *& op, uint64_t v)
{
    static_assert(R >= 1 && R <= 7);
    if constexpr (R >= 4)
    {
        storeU32Fast(op, static_cast<uint32_t>(v));
        op += 4u;
        if constexpr (R >= 6)
        {
            storeU16Fast(op, static_cast<uint16_t>(v >> 32));
            op += 2u;
            if constexpr (R == 7)
            {
                *op++ = static_cast<unsigned char>(v >> 48);
            }
        }
        else if constexpr (R == 5)
        {
            *op++ = static_cast<unsigned char>(v >> 32);
        }
    }
    else if constexpr (R >= 2)
    {
        storeU16Fast(op, static_cast<uint16_t>(v));
        op += 2u;
        if constexpr (R == 3)
        {
            *op++ = static_cast<unsigned char>(v >> 16);
        }
    }
    else
    {
        *op++ = static_cast<unsigned char>(v);
    }
}

// Pack one element into the word array at compile-time computed position
template <unsigned B, unsigned Base, size_t I>
static TURBOPFOR_ALWAYS_INLINE void pack_one(const uint32_t * __restrict in, uint64_t * __restrict w)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned wi = bitpos / 64u;
    constexpr unsigned sh = bitpos % 64u;

    w[wi] |= static_cast<uint64_t>(in[idx]) << sh;
    if constexpr (sh + B > 64u)
        w[wi + 1u] |= static_cast<uint64_t>(in[idx]) >> (64u - sh);
}

// Pack multiple elements using index sequence expansion
template <unsigned B, unsigned Base, size_t... I>
static TURBOPFOR_ALWAYS_INLINE void pack_all(const uint32_t * __restrict in, uint64_t * __restrict w, std::index_sequence<I...>)
{
    (pack_one<B, Base, I>(in, w), ...);
}

// Pack a block of K elements with bit width B
template <unsigned B, unsigned K, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE unsigned char * pack_block(const uint32_t * __restrict in, unsigned char * __restrict out)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;
    constexpr unsigned word_count = (total_bits + 63u) / 64u;
    constexpr unsigned last_bytes = total_bytes - (word_count - 1u) * 8u;

    uint64_t w[word_count] = {};
    pack_all<B, Base>(in, w, std::make_index_sequence<K>{});

    unsigned char * op = out;
    for (unsigned i = 0; i + 1u < word_count; ++i)
    {
        storeU64Fast(op, w[i]);
        op += 8u;
    }
    if constexpr (last_bytes == 8u)
    {
        storeU64Fast(op, w[word_count - 1u]);
        op += 8u;
    }
    else
    {
        store_partial<last_bytes>(op, w[word_count - 1u]);
    }
    return op;
}

// Recursively pack blocks of optimal size
template <unsigned B, unsigned N, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE unsigned char * pack_blocks(const uint32_t * __restrict in, unsigned char * __restrict out)
{
    if constexpr (N == 0u)
    {
        return out;
    }
    else
    {
        // Choose block size using same logic as unpack
        constexpr unsigned block = choose_block_size(B, N);

        unsigned char * op = pack_block<B, block, Base>(in, out);
        if constexpr (N == block)
            return op;
        else
            return pack_blocks<B, N - block, Base + block>(in, op);
    }
}

// Pack N elements with bit width B (N is compile-time constant)
template <unsigned B, unsigned N>
static TURBOPFOR_ALWAYS_INLINE unsigned char * pack_n_b(const uint32_t * __restrict in, unsigned char * __restrict out)
{
    static_assert(B >= 1 && B <= 32);
    static_assert(N >= 1 && N <= 32);

    // Special case: b=32 - copy with endian conversion
    if constexpr (B == 32)
    {
        copyU32ArrayToLe(out, in, N);
        return out + N * 4u;
    }
    // Special case: b=16 - simple 16-bit stores
    else if constexpr (B == 16)
    {
        unsigned char * op = out;
        for (unsigned i = 0; i < N; ++i)
        {
            storeU16Fast(op, static_cast<uint16_t>(in[i]));
            op += 2u;
        }
        return op;
    }
    // Special case: b=8 - byte stores
    else if constexpr (B == 8)
    {
        unsigned char * op = out;
        for (unsigned i = 0; i < N; ++i)
            *op++ = static_cast<unsigned char>(in[i]);
        return op;
    }
    else
    {
        return pack_blocks<B, N, 0u>(in, out);
    }
}

// Dispatch on runtime n (1-31) to compile-time N
template <unsigned B>
static TURBOPFOR_ALWAYS_INLINE unsigned char * pack_dispatch_n(const uint32_t * in, unsigned n, unsigned char * out)
{
    switch (n)
    {
        case 1u:
            return pack_n_b<B, 1>(in, out);
        case 2u:
            return pack_n_b<B, 2>(in, out);
        case 3u:
            return pack_n_b<B, 3>(in, out);
        case 4u:
            return pack_n_b<B, 4>(in, out);
        case 5u:
            return pack_n_b<B, 5>(in, out);
        case 6u:
            return pack_n_b<B, 6>(in, out);
        case 7u:
            return pack_n_b<B, 7>(in, out);
        case 8u:
            return pack_n_b<B, 8>(in, out);
        case 9u:
            return pack_n_b<B, 9>(in, out);
        case 10u:
            return pack_n_b<B, 10>(in, out);
        case 11u:
            return pack_n_b<B, 11>(in, out);
        case 12u:
            return pack_n_b<B, 12>(in, out);
        case 13u:
            return pack_n_b<B, 13>(in, out);
        case 14u:
            return pack_n_b<B, 14>(in, out);
        case 15u:
            return pack_n_b<B, 15>(in, out);
        case 16u:
            return pack_n_b<B, 16>(in, out);
        case 17u:
            return pack_n_b<B, 17>(in, out);
        case 18u:
            return pack_n_b<B, 18>(in, out);
        case 19u:
            return pack_n_b<B, 19>(in, out);
        case 20u:
            return pack_n_b<B, 20>(in, out);
        case 21u:
            return pack_n_b<B, 21>(in, out);
        case 22u:
            return pack_n_b<B, 22>(in, out);
        case 23u:
            return pack_n_b<B, 23>(in, out);
        case 24u:
            return pack_n_b<B, 24>(in, out);
        case 25u:
            return pack_n_b<B, 25>(in, out);
        case 26u:
            return pack_n_b<B, 26>(in, out);
        case 27u:
            return pack_n_b<B, 27>(in, out);
        case 28u:
            return pack_n_b<B, 28>(in, out);
        case 29u:
            return pack_n_b<B, 29>(in, out);
        case 30u:
            return pack_n_b<B, 30>(in, out);
        case 31u:
            return pack_n_b<B, 31>(in, out);
        default:
            __builtin_unreachable();
    }
}

// Main bitpack implementation struct with function table
struct Bitpack32ScalarImpl
{
    using Fn = unsigned char * (*)(const uint32_t *, unsigned, unsigned char *);

    template <unsigned B>
    static TURBOPFOR_ALWAYS_INLINE unsigned char * bitpack_b(const uint32_t * in, unsigned n, unsigned char * out)
    {
        // Process 32-element blocks
        const uint32_t * end = in + (n & ~31u);
        while (in < end)
        {
            out = pack_n_b<B, 32>(in, out);
            in += 32;
        }

        n &= 31u;
        if (n == 0u)
            return out;
        return pack_dispatch_n<B>(in, n, out);
    }

    static unsigned char * dispatch(const uint32_t * in, unsigned n, unsigned char * out, unsigned b)
    {
        if (b == 0u) [[unlikely]]
            return out;

        if (b == 32u) [[unlikely]]
        {
            copyU32ArrayToLe(out, in, n);
            return out + n * 4u;
        }

        return table[b](in, n, out);
    }

    static inline const Fn table[33] = {
        nullptr,        &bitpack_b<1>,  &bitpack_b<2>,  &bitpack_b<3>,  &bitpack_b<4>,  &bitpack_b<5>,  &bitpack_b<6>,
        &bitpack_b<7>,  &bitpack_b<8>,  &bitpack_b<9>,  &bitpack_b<10>, &bitpack_b<11>, &bitpack_b<12>, &bitpack_b<13>,
        &bitpack_b<14>, &bitpack_b<15>, &bitpack_b<16>, &bitpack_b<17>, &bitpack_b<18>, &bitpack_b<19>, &bitpack_b<20>,
        &bitpack_b<21>, &bitpack_b<22>, &bitpack_b<23>, &bitpack_b<24>, &bitpack_b<25>, &bitpack_b<26>, &bitpack_b<27>,
        &bitpack_b<28>, &bitpack_b<29>, &bitpack_b<30>, &bitpack_b<31>, &bitpack_b<32>,
    };
};

} // namespace turbopfor::scalar::detail
