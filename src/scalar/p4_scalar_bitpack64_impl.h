#pragma once

#include "p4_scalar_internal.h"

#include <utility>

namespace turbopfor::scalar::detail
{

// Write-through 64-bit bitpacking: streams packed words directly to output.
// Instead of building into a temporary buffer, we maintain a single accumulator
// and store each completed word immediately. This reduces register pressure
// and eliminates the temporary buffer zero-init + copy overhead for large block sizes.
//
// For each element at compile-time-known position:
//   - If it fits entirely in the current word: shift and OR into accumulator
//   - If it spans two words: OR high bits into accumulator, store it, start next
//     word with the low remainder bits
//   - If it starts at bit 0 of a new word: assign directly (no OR)

// State threaded through the compile-time fold: tracks current output pointer
// and the accumulator for the word being built.
struct PackState64
{
    unsigned char * op;
    uint64_t w; // Current word being accumulated
};

// Pack one 64-bit element, streaming completed words to output.
// Returns updated state.
template <unsigned B, unsigned Base, size_t I>
static TURBOPFOR_ALWAYS_INLINE PackState64 pack_one_stream64(const uint64_t * __restrict in, PackState64 s)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned sh = bitpos % 64u;
    constexpr unsigned end_bit = sh + B;

    if constexpr (sh == 0u)
    {
        // Starting a fresh word — assign, no OR needed
        s.w = in[idx];
        if constexpr (B == 64u)
        {
            // Full word — store immediately
            storeU64Fast(s.op, s.w);
            s.op += 8u;
            s.w = 0;
        }
    }
    else if constexpr (end_bit <= 64u)
    {
        // Fits entirely within current word
        s.w |= in[idx] << sh;
        if constexpr (end_bit == 64u)
        {
            // Exactly fills the word — store it
            storeU64Fast(s.op, s.w);
            s.op += 8u;
            s.w = 0;
        }
    }
    else
    {
        // Spans two words: high part goes in current word, low part starts next
        s.w |= in[idx] << sh;
        storeU64Fast(s.op, s.w);
        s.op += 8u;
        // Start next word with the remainder (sh is constexpr 1..63, so 64-sh is valid)
        s.w = in[idx] >> (64u - sh);
    }

    return s;
}

// Fold over all elements using index sequence
template <unsigned B, unsigned Base, size_t I0, size_t... Is>
static TURBOPFOR_ALWAYS_INLINE PackState64 pack_fold_stream64(const uint64_t * __restrict in, PackState64 s, std::index_sequence<I0, Is...>)
{
    s = pack_one_stream64<B, Base, I0>(in, s);
    if constexpr (sizeof...(Is) > 0)
        return pack_fold_stream64<B, Base>(in, s, std::index_sequence<Is...>{});
    else
        return s;
}

// Pack K 64-bit elements with bit width B, streaming to output.
template <unsigned B, unsigned K, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE unsigned char * pack_block_stream64(const uint64_t * __restrict in, unsigned char * __restrict out)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;
    constexpr unsigned tail_bits = total_bits % 64u;
    constexpr unsigned tail_bytes = tail_bits == 0u ? 0u : (tail_bits + 7u) / 8u;

    PackState64 s{out, 0};
    s = pack_fold_stream64<B, Base>(in, s, std::make_index_sequence<K>{});

    // Store any remaining partial word
    if constexpr (tail_bits > 0u)
    {
        if constexpr (tail_bytes == 8u)
        {
            storeU64Fast(s.op, s.w);
            s.op += 8u;
        }
        else
        {
            store_partial<tail_bytes>(s.op, s.w);
        }
    }

    // Return the canonical end pointer (computed from total bytes, not from streaming)
    return out + total_bytes;
}

// Pack N 64-bit elements with bit width B (N is compile-time constant)
template <unsigned B, unsigned N>
static TURBOPFOR_ALWAYS_INLINE unsigned char * pack64_n_b(const uint64_t * __restrict in, unsigned char * __restrict out)
{
    static_assert(B >= 1 && B <= 64);
    static_assert(N >= 1 && N <= 32);

    // Special case: b=64 - copy with endian conversion
    if constexpr (B == 64)
    {
        copyU64ArrayToLe(out, in, N);
        return out + N * 8u;
    }
    // Special case: all N values fit in a single 32-bit word (e.g. b=1 n=32)
    else if constexpr (N * B <= 32 && N * B > 0)
    {
        uint32_t w = 0;
        for (unsigned i = 0; i < N; ++i)
            w |= static_cast<uint32_t>(in[i]) << (i * B);
        constexpr unsigned total_bytes = (N * B + 7u) / 8u;
        if constexpr (total_bytes == 4u)
        {
            storeU32Fast(out, w);
        }
        else if constexpr (total_bytes == 3u)
        {
            storeU16Fast(out, static_cast<uint16_t>(w));
            out[2] = static_cast<unsigned char>(w >> 16);
        }
        else if constexpr (total_bytes == 2u)
        {
            storeU16Fast(out, static_cast<uint16_t>(w));
        }
        else
        {
            out[0] = static_cast<unsigned char>(w);
        }
        return out + total_bytes;
    }
    // Special case: b=32 - simple 32-bit stores
    else if constexpr (B == 32)
    {
        unsigned char * op = out;
        for (unsigned i = 0; i < N; ++i)
        {
            storeU32Fast(op, static_cast<uint32_t>(in[i]));
            op += 4u;
        }
        return op;
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
        // Use streaming write-through for all other bit widths.
        // No need for choose_block_size / recursive splitting —
        // the streaming approach handles any N directly without temporary buffers.
        return pack_block_stream64<B, N, 0u>(in, out);
    }
}

// Dispatch on runtime n (1-31) to compile-time N for 64-bit packing
template <unsigned B>
static TURBOPFOR_ALWAYS_INLINE unsigned char * pack64_dispatch_n(const uint64_t * in, unsigned n, unsigned char * out)
{
    switch (n)
    {
        case 1u:
            return pack64_n_b<B, 1>(in, out);
        case 2u:
            return pack64_n_b<B, 2>(in, out);
        case 3u:
            return pack64_n_b<B, 3>(in, out);
        case 4u:
            return pack64_n_b<B, 4>(in, out);
        case 5u:
            return pack64_n_b<B, 5>(in, out);
        case 6u:
            return pack64_n_b<B, 6>(in, out);
        case 7u:
            return pack64_n_b<B, 7>(in, out);
        case 8u:
            return pack64_n_b<B, 8>(in, out);
        case 9u:
            return pack64_n_b<B, 9>(in, out);
        case 10u:
            return pack64_n_b<B, 10>(in, out);
        case 11u:
            return pack64_n_b<B, 11>(in, out);
        case 12u:
            return pack64_n_b<B, 12>(in, out);
        case 13u:
            return pack64_n_b<B, 13>(in, out);
        case 14u:
            return pack64_n_b<B, 14>(in, out);
        case 15u:
            return pack64_n_b<B, 15>(in, out);
        case 16u:
            return pack64_n_b<B, 16>(in, out);
        case 17u:
            return pack64_n_b<B, 17>(in, out);
        case 18u:
            return pack64_n_b<B, 18>(in, out);
        case 19u:
            return pack64_n_b<B, 19>(in, out);
        case 20u:
            return pack64_n_b<B, 20>(in, out);
        case 21u:
            return pack64_n_b<B, 21>(in, out);
        case 22u:
            return pack64_n_b<B, 22>(in, out);
        case 23u:
            return pack64_n_b<B, 23>(in, out);
        case 24u:
            return pack64_n_b<B, 24>(in, out);
        case 25u:
            return pack64_n_b<B, 25>(in, out);
        case 26u:
            return pack64_n_b<B, 26>(in, out);
        case 27u:
            return pack64_n_b<B, 27>(in, out);
        case 28u:
            return pack64_n_b<B, 28>(in, out);
        case 29u:
            return pack64_n_b<B, 29>(in, out);
        case 30u:
            return pack64_n_b<B, 30>(in, out);
        case 31u:
            return pack64_n_b<B, 31>(in, out);
        default:
            __builtin_unreachable();
    }
}

// Main bitpack64 implementation struct with function table
struct Bitpack64ScalarImpl
{
    using Fn = unsigned char * (*)(const uint64_t *, unsigned, unsigned char *);

    template <unsigned B>
    static TURBOPFOR_ALWAYS_INLINE unsigned char * bitpack_b(const uint64_t * in, unsigned n, unsigned char * out)
    {
        // Process 32-element blocks
        const uint64_t * end = in + (n & ~31u);
        while (in < end)
        {
            out = pack64_n_b<B, 32>(in, out);
            in += 32;
        }

        n &= 31u;
        if (n == 0u)
            return out;
        return pack64_dispatch_n<B>(in, n, out);
    }

    static unsigned char * dispatch(const uint64_t * in, unsigned n, unsigned char * out, unsigned b)
    {
        if (b == 0u) [[unlikely]]
            return out;

        if (b == 64u) [[unlikely]]
        {
            copyU64ArrayToLe(out, in, n);
            return out + n * 8u;
        }

        return table[b](in, n, out);
    }

    static inline const Fn table[65] = {
        nullptr,        &bitpack_b<1>,  &bitpack_b<2>,  &bitpack_b<3>,  &bitpack_b<4>,  &bitpack_b<5>,  &bitpack_b<6>,  &bitpack_b<7>,
        &bitpack_b<8>,  &bitpack_b<9>,  &bitpack_b<10>, &bitpack_b<11>, &bitpack_b<12>, &bitpack_b<13>, &bitpack_b<14>, &bitpack_b<15>,
        &bitpack_b<16>, &bitpack_b<17>, &bitpack_b<18>, &bitpack_b<19>, &bitpack_b<20>, &bitpack_b<21>, &bitpack_b<22>, &bitpack_b<23>,
        &bitpack_b<24>, &bitpack_b<25>, &bitpack_b<26>, &bitpack_b<27>, &bitpack_b<28>, &bitpack_b<29>, &bitpack_b<30>, &bitpack_b<31>,
        &bitpack_b<32>, &bitpack_b<33>, &bitpack_b<34>, &bitpack_b<35>, &bitpack_b<36>, &bitpack_b<37>, &bitpack_b<38>, &bitpack_b<39>,
        &bitpack_b<40>, &bitpack_b<41>, &bitpack_b<42>, &bitpack_b<43>, &bitpack_b<44>, &bitpack_b<45>, &bitpack_b<46>, &bitpack_b<47>,
        &bitpack_b<48>, &bitpack_b<49>, &bitpack_b<50>, &bitpack_b<51>, &bitpack_b<52>, &bitpack_b<53>, &bitpack_b<54>, &bitpack_b<55>,
        &bitpack_b<56>, &bitpack_b<57>, &bitpack_b<58>, &bitpack_b<59>, &bitpack_b<60>, &bitpack_b<61>, &bitpack_b<62>, &bitpack_b<63>,
        &bitpack_b<64>,
    };
};

} // namespace turbopfor::scalar::detail
