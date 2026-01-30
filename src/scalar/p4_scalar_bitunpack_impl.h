#pragma once

#include "p4_scalar_internal.h"

#include <algorithm>
#include <utility>

namespace turbopfor::scalar::detail
{

// Helper to prevent SLP vectorization for specific bit widths.
// SLP vectorizer merges consecutive 4-byte stores (out[0], out[1], out[2], out[3])
// into vector stores using pinsrd/movdqu, which is slower for some bit widths.
// Using volatile prevents the compiler from merging these stores.
template <bool PreventSlp>
static TURBOPFOR_ALWAYS_INLINE void store_u32(uint32_t * out, unsigned idx, uint32_t val)
{
    if constexpr (PreventSlp)
    {
        // Use volatile to prevent SLP vectorizer from merging stores
        volatile uint32_t * vout = out;
        vout[idx] = val;
    }
    else
    {
        out[idx] = val;
    }
}

// Bit widths that suffer from SLP vectorization overhead
// SLP vectorizer merges consecutive stores into vector ops (pinsrd/movdqu),
// which is slower for certain bit widths.
// Delta1=false and Delta1=true have different optimal sets.
// Determined by benchmarking across n=32,63,64,96,127 with --bitunpack and --bitunpackd1
template <bool Delta1, unsigned B>
constexpr bool needs_slp_prevention()
{
    if constexpr (Delta1)
    {
        // Delta decoding has data dependencies that mostly prevent harmful SLP,
        // and enabling SLP prevention hurts performance across almost all bit widths.
        return false;
    }
    else
    {
        // Non-delta bitunpack: B=17 shows consistent large improvement (+17.7% avg)
        // across most n values when SLP is prevented.
        return B == 17;
    }
}

template <unsigned R>
static TURBOPFOR_ALWAYS_INLINE uint64_t load_partial(unsigned char *& ip)
{
    static_assert(R >= 1 && R <= 7);
    uint64_t v = 0;
    if constexpr (R >= 4)
    {
        v |= static_cast<uint64_t>(loadU32Fast(ip));
        ip += 4u;
        if constexpr (R >= 6)
        {
            v |= static_cast<uint64_t>(loadU16Fast(ip)) << 32;
            ip += 2u;
            if constexpr (R == 7)
            {
                v |= static_cast<uint64_t>(ip[0]) << 48;
                ip += 1u;
            }
        }
        else if constexpr (R == 5)
        {
            v |= static_cast<uint64_t>(ip[0]) << 32;
            ip += 1u;
        }
    }
    else if constexpr (R >= 2)
    {
        v |= static_cast<uint64_t>(loadU16Fast(ip));
        ip += 2u;
        if constexpr (R == 3)
        {
            v |= static_cast<uint64_t>(ip[0]) << 16;
            ip += 1u;
        }
    }
    else
    {
        v |= static_cast<uint64_t>(ip[0]);
        ip += 1u;
    }
    return v;
}

constexpr unsigned gcd_u32(unsigned a, unsigned b)
{
    return b == 0u ? a : gcd_u32(b, a % b);
}

constexpr unsigned word_count_for(unsigned b, unsigned n)
{
    return (n * b + 63u) / 64u;
}

constexpr unsigned max_words_for_gcd(unsigned g)
{
    return (g <= 2u) ? 12u : 16u;
}

constexpr unsigned choose_block_size(unsigned b, unsigned n)
{
    if (n == 0u)
        return 0u;
    const unsigned g = gcd_u32(64u, b);
    unsigned period = 64u / g;
    unsigned max_words = max_words_for_gcd(g);

    unsigned k = period;
    while (k > 1u && k > n)
        k >>= 1u;
    if (k > n)
        k = 1u;

    for (;;)
    {
        if (word_count_for(b, k) <= max_words && ((k * b) % 8u == 0u))
            return k;
        if (k == 1u)
            break;
        k >>= 1u;
    }
    return n;
}

template <bool Delta1, unsigned B, unsigned Base, size_t I, bool PreventSlp = needs_slp_prevention<Delta1, B>()>
static TURBOPFOR_ALWAYS_INLINE void
unpack_emit_one(const uint64_t * __restrict w, uint32_t * __restrict out, [[maybe_unused]] uint32_t & acc)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned wi = bitpos / 64u;
    constexpr unsigned sh = bitpos % 64u;
    constexpr uint32_t mask = static_cast<uint32_t>((uint64_t{1} << B) - 1u);

    uint64_t v = w[wi] >> sh;
    if constexpr (sh + B > 64u)
        v |= w[wi + 1u] << (64u - sh);

    if constexpr (Delta1)
    {
        acc += static_cast<uint32_t>(v) & mask;
        store_u32<PreventSlp>(out, idx, acc + (idx + 1u));
    }
    else
    {
        store_u32<PreventSlp>(out, idx, static_cast<uint32_t>(v) & mask);
    }
}

template <bool Delta1, unsigned B, unsigned Base, size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_emit(const uint64_t * __restrict w, uint32_t * __restrict out, uint32_t & acc, std::index_sequence<I...>)
{
    (unpack_emit_one<Delta1, B, Base, I>(w, out, acc), ...);
}

template <bool Delta1, unsigned B, unsigned K, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE unsigned char *
unpack_block(unsigned char * __restrict in, uint32_t * __restrict out, [[maybe_unused]] uint32_t & acc)
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

    unpack_emit<Delta1, B, Base>(w, out, acc, std::make_index_sequence<K>{});
    return ip;
}

template <bool Delta1, unsigned B, unsigned N, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack_blocks(unsigned char * __restrict in, uint32_t * __restrict out, uint32_t acc)
{
    if constexpr (N == 0u)
    {
        return in;
    }
    else
    {
        constexpr unsigned block = choose_block_size(B, N);

        unsigned char * ip = unpack_block<Delta1, B, block, Base>(in, out, acc);
        if constexpr (N == block)
            return ip;
        else
            return unpack_blocks<Delta1, B, N - block, Base + block>(ip, out, acc);
    }
}

template <std::size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_b32_d1_32_emit(const unsigned char * in, uint32_t * out, uint32_t & acc, std::index_sequence<I...>)
{
    ((acc += loadU32Fast(in + 4u * static_cast<unsigned>(I)), out[I] = acc + (static_cast<unsigned>(I) + 1u)), ...);
}

static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack_b32_d1_32(unsigned char * in, uint32_t * out, uint32_t start)
{
    uint32_t acc = start;
    unpack_b32_d1_32_emit(in, out, acc, std::make_index_sequence<32>{});
    return in + 128u;
}

template <std::size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_b16_d1_32_emit(const unsigned char * in, uint32_t * out, uint32_t & acc, std::index_sequence<I...>)
{
    ((acc += loadU16Fast(in + 2u * static_cast<unsigned>(I)), out[I] = acc + (static_cast<unsigned>(I) + 1u)), ...);
}

static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack_b16_d1_32(unsigned char * in, uint32_t * out, uint32_t start)
{
    uint32_t acc = start;
    unpack_b16_d1_32_emit(in, out, acc, std::make_index_sequence<32>{});
    return in + 64u;
}

template <std::size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_b8_d1_32_emit(const unsigned char * in, uint32_t * out, uint32_t & acc, std::index_sequence<I...>)
{
    ((acc += static_cast<uint32_t>(in[static_cast<unsigned>(I)]), out[I] = acc + (static_cast<unsigned>(I) + 1u)), ...);
}

static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack_b8_d1_32(unsigned char * in, uint32_t * out, uint32_t start)
{
    uint32_t acc = start;
    unpack_b8_d1_32_emit(in, out, acc, std::make_index_sequence<32>{});
    return in + 32u;
}

template <bool Delta1, unsigned B, unsigned N>
static TURBOPFOR_ALWAYS_INLINE unsigned char *
unpack_n_b(unsigned char * __restrict in, uint32_t * __restrict out, [[maybe_unused]] uint32_t start)
{
    static_assert(B >= 1 && B <= 32);
    static_assert(N >= 1 && N <= 32);

    if constexpr (Delta1 && B == 32 && N == 32)
        return unpack_b32_d1_32(in, out, start);
    if constexpr (Delta1 && B == 16 && N == 32)
        return unpack_b16_d1_32(in, out, start);
    if constexpr (Delta1 && B == 8 && N == 32)
        return unpack_b8_d1_32(in, out, start);

    if constexpr (B == 32)
    {
        unsigned char * ip = in;
        for (unsigned i = 0; i < N; ++i)
        {
            uint32_t v = loadU32Fast(ip);
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
            uint32_t v = loadU16Fast(ip);
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
            uint32_t v = *ip++;
            if constexpr (Delta1)
                out[i] = (start += v) + (i + 1u);
            else
                out[i] = v;
        }
        return ip;
    }
    else
    {
        return unpack_blocks<Delta1, B, N, 0u>(in, out, start);
    }
}

template <bool Delta1, unsigned B>
static TURBOPFOR_ALWAYS_INLINE unsigned char * unpack(unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
{
    switch (n)
    {
        case 1u:
            return unpack_n_b<Delta1, B, 1>(in, out, start);
        case 2u:
            return unpack_n_b<Delta1, B, 2>(in, out, start);
        case 3u:
            return unpack_n_b<Delta1, B, 3>(in, out, start);
        case 4u:
            return unpack_n_b<Delta1, B, 4>(in, out, start);
        case 5u:
            return unpack_n_b<Delta1, B, 5>(in, out, start);
        case 6u:
            return unpack_n_b<Delta1, B, 6>(in, out, start);
        case 7u:
            return unpack_n_b<Delta1, B, 7>(in, out, start);
        case 8u:
            return unpack_n_b<Delta1, B, 8>(in, out, start);
        case 9u:
            return unpack_n_b<Delta1, B, 9>(in, out, start);
        case 10u:
            return unpack_n_b<Delta1, B, 10>(in, out, start);
        case 11u:
            return unpack_n_b<Delta1, B, 11>(in, out, start);
        case 12u:
            return unpack_n_b<Delta1, B, 12>(in, out, start);
        case 13u:
            return unpack_n_b<Delta1, B, 13>(in, out, start);
        case 14u:
            return unpack_n_b<Delta1, B, 14>(in, out, start);
        case 15u:
            return unpack_n_b<Delta1, B, 15>(in, out, start);
        case 16u:
            return unpack_n_b<Delta1, B, 16>(in, out, start);
        case 17u:
            return unpack_n_b<Delta1, B, 17>(in, out, start);
        case 18u:
            return unpack_n_b<Delta1, B, 18>(in, out, start);
        case 19u:
            return unpack_n_b<Delta1, B, 19>(in, out, start);
        case 20u:
            return unpack_n_b<Delta1, B, 20>(in, out, start);
        case 21u:
            return unpack_n_b<Delta1, B, 21>(in, out, start);
        case 22u:
            return unpack_n_b<Delta1, B, 22>(in, out, start);
        case 23u:
            return unpack_n_b<Delta1, B, 23>(in, out, start);
        case 24u:
            return unpack_n_b<Delta1, B, 24>(in, out, start);
        case 25u:
            return unpack_n_b<Delta1, B, 25>(in, out, start);
        case 26u:
            return unpack_n_b<Delta1, B, 26>(in, out, start);
        case 27u:
            return unpack_n_b<Delta1, B, 27>(in, out, start);
        case 28u:
            return unpack_n_b<Delta1, B, 28>(in, out, start);
        case 29u:
            return unpack_n_b<Delta1, B, 29>(in, out, start);
        case 30u:
            return unpack_n_b<Delta1, B, 30>(in, out, start);
        case 31u:
            return unpack_n_b<Delta1, B, 31>(in, out, start);
        default:
            __builtin_unreachable();
    }
}

template <bool Delta1>
struct Bitunpack32ScalarImpl
{
    using Fn = unsigned char * (*)(unsigned char *, unsigned, uint32_t *, uint32_t);

    template <unsigned B>
    static TURBOPFOR_ALWAYS_INLINE unsigned char *
    bitunpack_b(unsigned char * in, unsigned n, uint32_t * out, [[maybe_unused]] uint32_t start)
    {
        // Process 32-element blocks
        uint32_t * end = out + (n & ~31);
        while (out < end)
        {
            if constexpr (Delta1)
            {
                in = unpack_n_b<true, B, 32>(in, out, start);
                start = out[31];
            }
            else
            {
                in = unpack_n_b<false, B, 32>(in, out, 0u);
            }
            out += 32;
        }

        n &= 31;
        if (n == 0u)
            return in;
        if constexpr (Delta1)
            return unpack<true, B>(in, n, out, start);
        else
            return unpack<false, B>(in, n, out, 0u);
    }

    static unsigned char * dispatch(unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b)
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
                std::fill(out, out + n, 0u);
            }
            return in;
        }

        return table[b](in, n, out, start);
    }

    static inline const Fn table[33] = {
        nullptr,          &bitunpack_b<1>,  &bitunpack_b<2>,  &bitunpack_b<3>,  &bitunpack_b<4>,  &bitunpack_b<5>,  &bitunpack_b<6>,
        &bitunpack_b<7>,  &bitunpack_b<8>,  &bitunpack_b<9>,  &bitunpack_b<10>, &bitunpack_b<11>, &bitunpack_b<12>, &bitunpack_b<13>,
        &bitunpack_b<14>, &bitunpack_b<15>, &bitunpack_b<16>, &bitunpack_b<17>, &bitunpack_b<18>, &bitunpack_b<19>, &bitunpack_b<20>,
        &bitunpack_b<21>, &bitunpack_b<22>, &bitunpack_b<23>, &bitunpack_b<24>, &bitunpack_b<25>, &bitunpack_b<26>, &bitunpack_b<27>,
        &bitunpack_b<28>, &bitunpack_b<29>, &bitunpack_b<30>, &bitunpack_b<31>, &bitunpack_b<32>,
    };
};

} // namespace turbopfor::scalar::detail
