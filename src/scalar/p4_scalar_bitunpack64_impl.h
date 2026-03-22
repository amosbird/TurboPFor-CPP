#pragma once

#include "p4_scalar_internal.h"

#include <algorithm>
#include <utility>

namespace turbopfor::scalar::detail
{

// Helper to prevent SLP vectorization for 64-bit unpacking
template <bool PreventSlp>
static TURBOPFOR_ALWAYS_INLINE void store_u64(uint64_t * out, unsigned idx, uint64_t val)
{
    if constexpr (PreventSlp)
    {
        volatile uint64_t * vout = out;
        vout[idx] = val;
    }
    else
    {
        out[idx] = val;
    }
}

// SLP prevention policy for 64-bit unpacking
// Start conservatively — no SLP prevention. Can be tuned via benchmarking later.
template <bool Delta1, unsigned B>
constexpr bool needs_slp_prevention64()
{
    (void)sizeof(Delta1); // suppress unused
    (void)sizeof(B);
    return false;
}

// Unpack one 64-bit element from word array at compile-time computed position
template <bool Delta1, unsigned B, unsigned Base, size_t I, bool PreventSlp = needs_slp_prevention64<Delta1, B>()>
static TURBOPFOR_ALWAYS_INLINE void
unpack_emit_one64(const uint64_t * __restrict w, uint64_t * __restrict out, [[maybe_unused]] uint64_t & acc)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned wi = bitpos / 64u;
    constexpr unsigned sh = bitpos % 64u;

    uint64_t v;
    if constexpr (B == 64u && sh == 0u)
    {
        // Full 64-bit word, no masking needed
        v = w[wi];
    }
    else if constexpr (sh + B <= 64u)
    {
        // Fits entirely within one word
        constexpr uint64_t mask = (B == 64u) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << B) - 1ull);
        v = (w[wi] >> sh) & mask;
    }
    else
    {
        // Spans two words — combine and mask
        constexpr uint64_t mask = (B == 64u) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << B) - 1ull);
        v = ((w[wi] >> sh) | (w[wi + 1u] << (64u - sh))) & mask;
    }

    if constexpr (Delta1)
    {
        acc += v;
        store_u64<PreventSlp>(out, idx, acc + (idx + 1u));
    }
    else
    {
        store_u64<PreventSlp>(out, idx, v);
    }
}

// Unpack multiple 64-bit elements using index sequence expansion
template <bool Delta1, unsigned B, unsigned Base, size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_emit64(const uint64_t * __restrict w, uint64_t * __restrict out, uint64_t & acc, std::index_sequence<I...>)
{
    (unpack_emit_one64<Delta1, B, Base, I>(w, out, acc), ...);
}

// Unpack a block of K 64-bit elements with bit width B
template <bool Delta1, unsigned B, unsigned K, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE unsigned char *
unpack_block64(unsigned char * __restrict in, uint64_t * __restrict out, [[maybe_unused]] uint64_t & acc)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;
    constexpr unsigned word_count = (total_bits + 63u) / 64u;
    constexpr unsigned last_bytes = total_bytes - (word_count - 1u) * 8u;

    uint64_t w[word_count];
    unsigned char * ip = in;
    for (unsigned i = 0; i + 1u < word_count; ++i)
    {
        w[i] = loadU64Fast(ip);
        ip += 8u;
    }
    if constexpr (last_bytes == 8u)
    {
        w[word_count - 1u] = loadU64Fast(ip);
        ip += 8u;
    }
    else
    {
        w[word_count - 1u] = load_partial<last_bytes>(ip);
    }

    unpack_emit64<Delta1, B, Base>(w, out, acc, std::make_index_sequence<K>{});
    return ip;
}

// Recursively unpack blocks of optimal size for 64-bit values
template <bool Delta1, unsigned B, unsigned N, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack_blocks64(unsigned char * __restrict in, uint64_t * __restrict out, uint64_t acc)
{
    if constexpr (N == 0u)
    {
        return in;
    }
    else
    {
        constexpr unsigned block = choose_block_size(B, N);

        unsigned char * ip = unpack_block64<Delta1, B, block, Base>(in, out, acc);
        if constexpr (N == block)
            return ip;
        else
            return unpack_blocks64<Delta1, B, N - block, Base + block>(ip, out, acc);
    }
}

// Special case: b=64 with delta1, 32 elements
template <std::size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_b64_d1_32_emit(const unsigned char * in, uint64_t * out, uint64_t & acc, std::index_sequence<I...>)
{
    ((acc += loadU64Fast(in + 8u * static_cast<unsigned>(I)), out[I] = acc + (static_cast<unsigned>(I) + 1u)), ...);
}

static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack_b64_d1_32(unsigned char * in, uint64_t * out, uint64_t start)
{
    uint64_t acc = start;
    unpack_b64_d1_32_emit(in, out, acc, std::make_index_sequence<32>{});
    return in + 256u;
}

// Special case: b=32 with delta1, 32 elements
template <std::size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_b32_d1_32_emit64(const unsigned char * in, uint64_t * out, uint64_t & acc, std::index_sequence<I...>)
{
    ((acc += static_cast<uint64_t>(loadU32Fast(in + 4u * static_cast<unsigned>(I))), out[I] = acc + (static_cast<unsigned>(I) + 1u)), ...);
}

static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack_b32_d1_32_u64(unsigned char * in, uint64_t * out, uint64_t start)
{
    uint64_t acc = start;
    unpack_b32_d1_32_emit64(in, out, acc, std::make_index_sequence<32>{});
    return in + 128u;
}

// Special case: b=16 with delta1, 32 elements
template <std::size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_b16_d1_32_emit64(const unsigned char * in, uint64_t * out, uint64_t & acc, std::index_sequence<I...>)
{
    ((acc += static_cast<uint64_t>(loadU16Fast(in + 2u * static_cast<unsigned>(I))), out[I] = acc + (static_cast<unsigned>(I) + 1u)), ...);
}

static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack_b16_d1_32_u64(unsigned char * in, uint64_t * out, uint64_t start)
{
    uint64_t acc = start;
    unpack_b16_d1_32_emit64(in, out, acc, std::make_index_sequence<32>{});
    return in + 64u;
}

// Special case: b=8 with delta1, 32 elements
template <std::size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_b8_d1_32_emit64(const unsigned char * in, uint64_t * out, uint64_t & acc, std::index_sequence<I...>)
{
    ((acc += static_cast<uint64_t>(in[static_cast<unsigned>(I)]), out[I] = acc + (static_cast<unsigned>(I) + 1u)), ...);
}

static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack_b8_d1_32_u64(unsigned char * in, uint64_t * out, uint64_t start)
{
    uint64_t acc = start;
    unpack_b8_d1_32_emit64(in, out, acc, std::make_index_sequence<32>{});
    return in + 32u;
}

// Unpack N 64-bit elements with bit width B
template <bool Delta1, unsigned B, unsigned N>
static TURBOPFOR_ALWAYS_INLINE unsigned char *
unpack64_n_b(unsigned char * __restrict in, uint64_t * __restrict out, [[maybe_unused]] uint64_t start)
{
    static_assert(B >= 1 && B <= 64);
    static_assert(N >= 1 && N <= 32);

    // Specialized fast paths for delta1 + aligned bit widths + full block
    if constexpr (Delta1 && B == 64 && N == 32)
        return unpack_b64_d1_32(in, out, start);
    if constexpr (Delta1 && B == 32 && N == 32)
        return unpack_b32_d1_32_u64(in, out, start);
    if constexpr (Delta1 && B == 16 && N == 32)
        return unpack_b16_d1_32_u64(in, out, start);
    if constexpr (Delta1 && B == 8 && N == 32)
        return unpack_b8_d1_32_u64(in, out, start);

    // General special cases for aligned bit widths
    if constexpr (B == 64)
    {
        unsigned char * ip = in;
        for (unsigned i = 0; i < N; ++i)
        {
            uint64_t v = loadU64Fast(ip);
            ip += 8u;
            if constexpr (Delta1)
                out[i] = (start += v) + (i + 1u);
            else
                out[i] = v;
        }
        return ip;
    }
    else if constexpr (B == 32)
    {
        unsigned char * ip = in;
        for (unsigned i = 0; i < N; ++i)
        {
            uint64_t v = loadU32Fast(ip);
            ip += 4u;
            if constexpr (Delta1)
                out[i] = (start += v) + (i + 1u);
            else
                out[i] = v;
        }
        return ip;
    }
    else if constexpr (B == 16)
    {
        unsigned char * ip = in;
        for (unsigned i = 0; i < N; ++i)
        {
            uint64_t v = loadU16Fast(ip);
            ip += 2u;
            if constexpr (Delta1)
                out[i] = (start += v) + (i + 1u);
            else
                out[i] = v;
        }
        return ip;
    }
    else if constexpr (B == 8)
    {
        unsigned char * ip = in;
        for (unsigned i = 0; i < N; ++i)
        {
            uint64_t v = *ip++;
            if constexpr (Delta1)
                out[i] = (start += v) + (i + 1u);
            else
                out[i] = v;
        }
        return ip;
    }
    else
    {
        return unpack_blocks64<Delta1, B, N, 0u>(in, out, start);
    }
}

// Dispatch on runtime n (1-31) to compile-time N for 64-bit unpacking
template <bool Delta1, unsigned B>
static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    switch (n)
    {
        case 1u:
            return unpack64_n_b<Delta1, B, 1>(in, out, start);
        case 2u:
            return unpack64_n_b<Delta1, B, 2>(in, out, start);
        case 3u:
            return unpack64_n_b<Delta1, B, 3>(in, out, start);
        case 4u:
            return unpack64_n_b<Delta1, B, 4>(in, out, start);
        case 5u:
            return unpack64_n_b<Delta1, B, 5>(in, out, start);
        case 6u:
            return unpack64_n_b<Delta1, B, 6>(in, out, start);
        case 7u:
            return unpack64_n_b<Delta1, B, 7>(in, out, start);
        case 8u:
            return unpack64_n_b<Delta1, B, 8>(in, out, start);
        case 9u:
            return unpack64_n_b<Delta1, B, 9>(in, out, start);
        case 10u:
            return unpack64_n_b<Delta1, B, 10>(in, out, start);
        case 11u:
            return unpack64_n_b<Delta1, B, 11>(in, out, start);
        case 12u:
            return unpack64_n_b<Delta1, B, 12>(in, out, start);
        case 13u:
            return unpack64_n_b<Delta1, B, 13>(in, out, start);
        case 14u:
            return unpack64_n_b<Delta1, B, 14>(in, out, start);
        case 15u:
            return unpack64_n_b<Delta1, B, 15>(in, out, start);
        case 16u:
            return unpack64_n_b<Delta1, B, 16>(in, out, start);
        case 17u:
            return unpack64_n_b<Delta1, B, 17>(in, out, start);
        case 18u:
            return unpack64_n_b<Delta1, B, 18>(in, out, start);
        case 19u:
            return unpack64_n_b<Delta1, B, 19>(in, out, start);
        case 20u:
            return unpack64_n_b<Delta1, B, 20>(in, out, start);
        case 21u:
            return unpack64_n_b<Delta1, B, 21>(in, out, start);
        case 22u:
            return unpack64_n_b<Delta1, B, 22>(in, out, start);
        case 23u:
            return unpack64_n_b<Delta1, B, 23>(in, out, start);
        case 24u:
            return unpack64_n_b<Delta1, B, 24>(in, out, start);
        case 25u:
            return unpack64_n_b<Delta1, B, 25>(in, out, start);
        case 26u:
            return unpack64_n_b<Delta1, B, 26>(in, out, start);
        case 27u:
            return unpack64_n_b<Delta1, B, 27>(in, out, start);
        case 28u:
            return unpack64_n_b<Delta1, B, 28>(in, out, start);
        case 29u:
            return unpack64_n_b<Delta1, B, 29>(in, out, start);
        case 30u:
            return unpack64_n_b<Delta1, B, 30>(in, out, start);
        case 31u:
            return unpack64_n_b<Delta1, B, 31>(in, out, start);
        default:
            __builtin_unreachable();
    }
}

// Main bitunpack64 implementation struct with function table
template <bool Delta1>
struct Bitunpack64ScalarImpl
{
    using Fn = unsigned char * (*)(unsigned char *, unsigned, uint64_t *, uint64_t);

    template <unsigned B>
    static TURBOPFOR_ALWAYS_INLINE unsigned char *
    bitunpack_b(unsigned char * in, unsigned n, uint64_t * out, [[maybe_unused]] uint64_t start)
    {
        // Process 32-element blocks
        uint64_t * end = out + (n & ~31u);
        while (out < end)
        {
            if constexpr (Delta1)
            {
                in = unpack64_n_b<true, B, 32>(in, out, start);
                start = out[31];
            }
            else
            {
                in = unpack64_n_b<false, B, 32>(in, out, 0ull);
            }
            out += 32;
        }

        n &= 31u;
        if (n == 0u)
            return in;
        if constexpr (Delta1)
            return unpack64<true, B>(in, n, out, start);
        else
            return unpack64<false, B>(in, n, out, 0ull);
    }

    static unsigned char * dispatch(unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b)
    {
        if (b == 0u) [[unlikely]]
        {
            if constexpr (Delta1)
            {
                for (unsigned i = 0; i < n; ++i)
                    out[i] = start + i + 1u;
            }
            else
            {
                std::fill(out, out + n, 0ull);
            }
            return in;
        }

        return table[b](in, n, out, start);
    }

    static inline const Fn table[65] = {
        nullptr,          &bitunpack_b<1>,  &bitunpack_b<2>,  &bitunpack_b<3>,  &bitunpack_b<4>,  &bitunpack_b<5>,  &bitunpack_b<6>,
        &bitunpack_b<7>,  &bitunpack_b<8>,  &bitunpack_b<9>,  &bitunpack_b<10>, &bitunpack_b<11>, &bitunpack_b<12>, &bitunpack_b<13>,
        &bitunpack_b<14>, &bitunpack_b<15>, &bitunpack_b<16>, &bitunpack_b<17>, &bitunpack_b<18>, &bitunpack_b<19>, &bitunpack_b<20>,
        &bitunpack_b<21>, &bitunpack_b<22>, &bitunpack_b<23>, &bitunpack_b<24>, &bitunpack_b<25>, &bitunpack_b<26>, &bitunpack_b<27>,
        &bitunpack_b<28>, &bitunpack_b<29>, &bitunpack_b<30>, &bitunpack_b<31>, &bitunpack_b<32>, &bitunpack_b<33>, &bitunpack_b<34>,
        &bitunpack_b<35>, &bitunpack_b<36>, &bitunpack_b<37>, &bitunpack_b<38>, &bitunpack_b<39>, &bitunpack_b<40>, &bitunpack_b<41>,
        &bitunpack_b<42>, &bitunpack_b<43>, &bitunpack_b<44>, &bitunpack_b<45>, &bitunpack_b<46>, &bitunpack_b<47>, &bitunpack_b<48>,
        &bitunpack_b<49>, &bitunpack_b<50>, &bitunpack_b<51>, &bitunpack_b<52>, &bitunpack_b<53>, &bitunpack_b<54>, &bitunpack_b<55>,
        &bitunpack_b<56>, &bitunpack_b<57>, &bitunpack_b<58>, &bitunpack_b<59>, &bitunpack_b<60>, &bitunpack_b<61>, &bitunpack_b<62>,
        &bitunpack_b<63>, &bitunpack_b<64>,
    };
};

} // namespace turbopfor::scalar::detail
