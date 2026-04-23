#pragma once

#include "p4_scalar_internal.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace turbopfor::scalar::detail
{

// =============================================================================
// Strategy overview for 64-bit bitunpacking
//
// Two code-generation strategies are used depending on bit-width:
//
// 1. B <= 32: Pre-loaded word array + fold-expression emit.
//    All input words are loaded into a local array w[], then elements are
//    extracted via compile-time index_sequence expansion. The word array is
//    small enough (<=16 words for 32 elements) for the compiler to keep
//    entirely in registers.
//
// 2. 33 <= B <= 63: Interleaved load/emit (lazy-load pattern).
//    Words are loaded one at a time into local variables, interleaved with
//    element extraction — each word is loaded exactly once, right before the
//    first element that needs it. This mirrors the C reference's macro pattern
//    and avoids: (a) large word arrays that spill to stack, and (b) the shrd
//    instruction which has higher latency than separate shr+shl+or on modern
//    x86. Non-delta stores use volatile to prevent SLP auto-vectorization
//    which hurts this access pattern.
//
// Both strategies share the same dispatch and tail-handling infrastructure.
// B == 64 and byte-aligned widths (8, 16, 32) have dedicated fast paths.
// =============================================================================

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Prevent SLP auto-vectorization for non-delta stores in the interleaved path.
/// The compiler's SLP pass merges adjacent scalar stores into SIMD stores,
/// which hurts throughput for the interleaved load pattern.
static TURBOPFOR_ALWAYS_INLINE void store_u64_noslp(uint64_t * out, unsigned idx, uint64_t val)
{
    volatile uint64_t * vout = out;
    vout[idx] = val;
}

// -----------------------------------------------------------------------------
// Strategy 1: Pre-loaded word array (B <= 32)
// -----------------------------------------------------------------------------

template <bool Delta1, unsigned B, unsigned Base, size_t I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_emit_one64(const uint64_t * __restrict w, uint64_t * __restrict out, [[maybe_unused]] uint64_t & acc)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned wi = bitpos / 64u;
    constexpr unsigned sh = bitpos % 64u;
    constexpr uint64_t mask = (B == 64u) ? ~0ull : ((1ull << B) - 1ull);

    uint64_t v;
    if constexpr (sh + B <= 64u)
        v = (w[wi] >> sh) & mask;
    else
        v = ((w[wi] >> sh) | (w[wi + 1u] << (64u - sh))) & mask;

    if constexpr (Delta1)
    {
        acc += v;
        out[idx] = acc + (idx + 1u);
    }
    else
    {
        out[idx] = v;
    }
}

template <bool Delta1, unsigned B, unsigned Base, size_t... I>
static TURBOPFOR_ALWAYS_INLINE void
unpack_emit64(const uint64_t * __restrict w, uint64_t * __restrict out, uint64_t & acc, std::index_sequence<I...>)
{
    (unpack_emit_one64<Delta1, B, Base, I>(w, out, acc), ...);
}

// -----------------------------------------------------------------------------
// Strategy 2: Interleaved load/emit (33 <= B <= 63)
//
// Each element is unpacked in sequence. The current word is kept in w_prev;
// when an element spans two words, w_cur is loaded and becomes the new w_prev.
// The cross-word combine uses (w_prev >> sh) | ((w_cur << (64-sh)) & mask)
// instead of the equivalent ((w_prev >> sh) | (w_cur << (64-sh))) & mask
// to prevent the compiler from emitting the high-latency shrd instruction.
// -----------------------------------------------------------------------------

#define TURBOPFOR_UNPACK64_ELEM(IDX)                                           \
    do                                                                         \
    {                                                                          \
        if constexpr ((IDX) < K)                                               \
        {                                                                      \
            constexpr unsigned bitpos_ = (IDX) *B;                             \
            constexpr unsigned wi_ = bitpos_ / 64u;                            \
            constexpr unsigned sh_ = bitpos_ % 64u;                            \
            constexpr unsigned out_idx_ = Base + (IDX);                        \
            uint64_t v_;                                                       \
            if constexpr (sh_ == 0u)                                           \
            {                                                                  \
                if constexpr (wi_ > 0u && (((IDX) -1u) * B) / 64u != wi_)     \
                    w_prev = loadU64Fast(in + wi_ * 8u);                       \
                v_ = w_prev & mask;                                            \
            }                                                                  \
            else if constexpr (sh_ + B <= 64u)                                 \
            {                                                                  \
                v_ = (w_prev >> sh_) & mask;                                   \
            }                                                                  \
            else                                                               \
            {                                                                  \
                w_cur = loadU64Fast(in + (wi_ + 1u) * 8u);                     \
                v_ = (w_prev >> sh_) | ((w_cur << (64u - sh_)) & mask);        \
                w_prev = w_cur;                                                \
            }                                                                  \
            if constexpr (Delta1)                                              \
            {                                                                  \
                acc += v_;                                                     \
                out[out_idx_] = acc + (out_idx_ + 1u);                         \
            }                                                                  \
            else                                                               \
            {                                                                  \
                store_u64_noslp(out, out_idx_, v_);                            \
            }                                                                  \
        }                                                                      \
    } while (0)

template <bool Delta1, unsigned B, unsigned K, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE void
unpack_interleaved64(const unsigned char * __restrict in, uint64_t * __restrict out, [[maybe_unused]] uint64_t & acc)
{
    static_assert(B > 32u && B < 64u);
    constexpr uint64_t mask = (1ull << B) - 1ull;

    uint64_t w_prev = loadU64Fast(in);
    uint64_t w_cur;

    TURBOPFOR_UNPACK64_ELEM(0);  TURBOPFOR_UNPACK64_ELEM(1);
    TURBOPFOR_UNPACK64_ELEM(2);  TURBOPFOR_UNPACK64_ELEM(3);
    TURBOPFOR_UNPACK64_ELEM(4);  TURBOPFOR_UNPACK64_ELEM(5);
    TURBOPFOR_UNPACK64_ELEM(6);  TURBOPFOR_UNPACK64_ELEM(7);
    TURBOPFOR_UNPACK64_ELEM(8);  TURBOPFOR_UNPACK64_ELEM(9);
    TURBOPFOR_UNPACK64_ELEM(10); TURBOPFOR_UNPACK64_ELEM(11);
    TURBOPFOR_UNPACK64_ELEM(12); TURBOPFOR_UNPACK64_ELEM(13);
    TURBOPFOR_UNPACK64_ELEM(14); TURBOPFOR_UNPACK64_ELEM(15);
    TURBOPFOR_UNPACK64_ELEM(16); TURBOPFOR_UNPACK64_ELEM(17);
    TURBOPFOR_UNPACK64_ELEM(18); TURBOPFOR_UNPACK64_ELEM(19);
    TURBOPFOR_UNPACK64_ELEM(20); TURBOPFOR_UNPACK64_ELEM(21);
    TURBOPFOR_UNPACK64_ELEM(22); TURBOPFOR_UNPACK64_ELEM(23);
    TURBOPFOR_UNPACK64_ELEM(24); TURBOPFOR_UNPACK64_ELEM(25);
    TURBOPFOR_UNPACK64_ELEM(26); TURBOPFOR_UNPACK64_ELEM(27);
    TURBOPFOR_UNPACK64_ELEM(28); TURBOPFOR_UNPACK64_ELEM(29);
    TURBOPFOR_UNPACK64_ELEM(30); TURBOPFOR_UNPACK64_ELEM(31);
}

#undef TURBOPFOR_UNPACK64_ELEM

// -----------------------------------------------------------------------------
// Block unpacking: selects strategy based on B
// -----------------------------------------------------------------------------

template <bool Delta1, unsigned B, unsigned K, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE const unsigned char *
unpack_block64(const unsigned char * __restrict in, uint64_t * __restrict out, [[maybe_unused]] uint64_t & acc)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;

    if constexpr (B > 32u && B < 64u)
    {
        unpack_interleaved64<Delta1, B, K, Base>(in, out, acc);
    }
    else
    {
        constexpr unsigned word_count = (total_bits + 63u) / 64u;
        constexpr unsigned last_bytes = total_bytes - (word_count - 1u) * 8u;

        uint64_t w[word_count];
        const unsigned char * ip = in;
        for (unsigned i = 0; i + 1u < word_count; ++i)
        {
            w[i] = loadU64Fast(ip);
            ip += 8u;
        }
        if constexpr (last_bytes == 8u)
            w[word_count - 1u] = loadU64Fast(ip);
        else
            w[word_count - 1u] = load_partial<last_bytes>(ip);

        unpack_emit64<Delta1, B, Base>(w, out, acc, std::make_index_sequence<K>{});
    }
    return in + total_bytes;
}

// Recursively unpack blocks of optimal size
template <bool Delta1, unsigned B, unsigned N, unsigned Base>
static TURBOPFOR_ALWAYS_INLINE const unsigned char *
unpack_blocks64(const unsigned char * __restrict in, uint64_t * __restrict out, uint64_t acc)
{
    if constexpr (N == 0u)
    {
        return in;
    }
    else
    {
        constexpr unsigned block = choose_block_size(B, N);
        const unsigned char * ip = unpack_block64<Delta1, B, block, Base>(in, out, acc);
        if constexpr (N == block)
            return ip;
        else
            return unpack_blocks64<Delta1, B, N - block, Base + block>(ip, out, acc);
    }
}

// -----------------------------------------------------------------------------
// Byte-aligned fast paths (delta1 + full 32-element block)
// -----------------------------------------------------------------------------

template <unsigned BytesPerElem, typename LoadFn>
static TURBOPFOR_ALWAYS_INLINE const unsigned char *
unpack_aligned_d1_32_u64(const unsigned char * in, uint64_t * out, uint64_t start, LoadFn load)
{
    for (unsigned i = 0; i < 32; ++i)
    {
        start += static_cast<uint64_t>(load(in + BytesPerElem * i));
        out[i] = start + (i + 1u);
    }
    return in + BytesPerElem * 32u;
}

// -----------------------------------------------------------------------------
// Top-level unpack: N elements at compile-time bit-width B
// -----------------------------------------------------------------------------

template <bool Delta1, unsigned B, unsigned N>
static TURBOPFOR_ALWAYS_INLINE const unsigned char *
unpack64_n_b(const unsigned char * __restrict in, uint64_t * __restrict out, [[maybe_unused]] uint64_t start)
{
    static_assert(B >= 1 && B <= 64);
    static_assert(N >= 1 && N <= 32);

    // Byte-aligned fast paths (B == 8, 16, 32, 64)
    if constexpr (B == 64)
    {
        if constexpr (Delta1 && N == 32)
            return unpack_aligned_d1_32_u64<8>(in, out, start, [](const unsigned char * p) { return loadU64Fast(p); });

        const unsigned char * ip = in;
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
        if constexpr (Delta1 && N == 32)
            return unpack_aligned_d1_32_u64<4>(in, out, start, [](const unsigned char * p) { return static_cast<uint64_t>(loadU32Fast(p)); });

        const unsigned char * ip = in;
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
        if constexpr (Delta1 && N == 32)
            return unpack_aligned_d1_32_u64<2>(in, out, start, [](const unsigned char * p) { return static_cast<uint64_t>(loadU16Fast(p)); });

        const unsigned char * ip = in;
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
        if constexpr (Delta1 && N == 32)
            return unpack_aligned_d1_32_u64<1>(in, out, start, [](const unsigned char * p) { return static_cast<uint64_t>(*p); });

        const unsigned char * ip = in;
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
        // General bitpacked path (non-byte-aligned)
        return unpack_blocks64<Delta1, B, N, 0u>(in, out, start);
    }
}

// -----------------------------------------------------------------------------
// Tail dispatch: runtime n -> compile-time N
// -----------------------------------------------------------------------------

template <bool Delta1, unsigned B>
static TURBOPFOR_ALWAYS_INLINE const unsigned char *
unpack64_tail(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    switch (n)
    {
        // clang-format off
        case  1: return unpack64_n_b<Delta1, B,  1>(in, out, start);
        case  2: return unpack64_n_b<Delta1, B,  2>(in, out, start);
        case  3: return unpack64_n_b<Delta1, B,  3>(in, out, start);
        case  4: return unpack64_n_b<Delta1, B,  4>(in, out, start);
        case  5: return unpack64_n_b<Delta1, B,  5>(in, out, start);
        case  6: return unpack64_n_b<Delta1, B,  6>(in, out, start);
        case  7: return unpack64_n_b<Delta1, B,  7>(in, out, start);
        case  8: return unpack64_n_b<Delta1, B,  8>(in, out, start);
        case  9: return unpack64_n_b<Delta1, B,  9>(in, out, start);
        case 10: return unpack64_n_b<Delta1, B, 10>(in, out, start);
        case 11: return unpack64_n_b<Delta1, B, 11>(in, out, start);
        case 12: return unpack64_n_b<Delta1, B, 12>(in, out, start);
        case 13: return unpack64_n_b<Delta1, B, 13>(in, out, start);
        case 14: return unpack64_n_b<Delta1, B, 14>(in, out, start);
        case 15: return unpack64_n_b<Delta1, B, 15>(in, out, start);
        case 16: return unpack64_n_b<Delta1, B, 16>(in, out, start);
        case 17: return unpack64_n_b<Delta1, B, 17>(in, out, start);
        case 18: return unpack64_n_b<Delta1, B, 18>(in, out, start);
        case 19: return unpack64_n_b<Delta1, B, 19>(in, out, start);
        case 20: return unpack64_n_b<Delta1, B, 20>(in, out, start);
        case 21: return unpack64_n_b<Delta1, B, 21>(in, out, start);
        case 22: return unpack64_n_b<Delta1, B, 22>(in, out, start);
        case 23: return unpack64_n_b<Delta1, B, 23>(in, out, start);
        case 24: return unpack64_n_b<Delta1, B, 24>(in, out, start);
        case 25: return unpack64_n_b<Delta1, B, 25>(in, out, start);
        case 26: return unpack64_n_b<Delta1, B, 26>(in, out, start);
        case 27: return unpack64_n_b<Delta1, B, 27>(in, out, start);
        case 28: return unpack64_n_b<Delta1, B, 28>(in, out, start);
        case 29: return unpack64_n_b<Delta1, B, 29>(in, out, start);
        case 30: return unpack64_n_b<Delta1, B, 30>(in, out, start);
        case 31: return unpack64_n_b<Delta1, B, 31>(in, out, start);
        // clang-format on
        default: __builtin_unreachable();
    }
}

// -----------------------------------------------------------------------------
// Entry point: function table indexed by runtime bit-width
// -----------------------------------------------------------------------------

/// Whether a given bit-width uses the interleaved (non-byte-aligned large) path.
/// These widths use the memcpy-tail strategy to avoid 31 template instantiations
/// per bitwidth, which would cause excessive icache pressure.
template <unsigned B>
constexpr bool uses_interleaved_path()
{
    return B > 32u && B < 64u && B != 48u && B != 40u && B != 56u;
}

template <bool Delta1>
struct Bitunpack64ScalarImpl
{
    using Fn = const unsigned char * (*)(const unsigned char *, unsigned, uint64_t *, uint64_t);

    template <unsigned B>
    static TURBOPFOR_ALWAYS_INLINE const unsigned char *
    bitunpack_b(const unsigned char * in, unsigned n, uint64_t * out, [[maybe_unused]] uint64_t start)
    {
        const unsigned char * ret = in + ((static_cast<uint64_t>(n) * B + 7u) / 8u);

        // Main loop: 32-element blocks
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

        // Tail: remaining 1-31 elements
        n &= 31u;
        if (n == 0u)
            return ret;

        if constexpr (uses_interleaved_path<B>())
        {
            // Decode a full 32-element block into a temp buffer and copy
            // only the needed elements. This avoids instantiating 31 tail
            // templates per bitwidth (mirrors the C reference's BU macro).
            alignas(64) uint64_t tmp[32];
            unpack64_n_b<Delta1, B, 32>(in, tmp, start);
            std::memcpy(out, tmp, n * sizeof(uint64_t));
            return ret;
        }
        else
        {
            if constexpr (Delta1)
                return unpack64_tail<true, B>(in, n, out, start);
            else
                return unpack64_tail<false, B>(in, n, out, 0ull);
        }
    }

    static const unsigned char * dispatch(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b)
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

    // clang-format off
    static inline const Fn table[65] = {
        nullptr,          &bitunpack_b<1>,  &bitunpack_b<2>,  &bitunpack_b<3>,
        &bitunpack_b<4>,  &bitunpack_b<5>,  &bitunpack_b<6>,  &bitunpack_b<7>,
        &bitunpack_b<8>,  &bitunpack_b<9>,  &bitunpack_b<10>, &bitunpack_b<11>,
        &bitunpack_b<12>, &bitunpack_b<13>, &bitunpack_b<14>, &bitunpack_b<15>,
        &bitunpack_b<16>, &bitunpack_b<17>, &bitunpack_b<18>, &bitunpack_b<19>,
        &bitunpack_b<20>, &bitunpack_b<21>, &bitunpack_b<22>, &bitunpack_b<23>,
        &bitunpack_b<24>, &bitunpack_b<25>, &bitunpack_b<26>, &bitunpack_b<27>,
        &bitunpack_b<28>, &bitunpack_b<29>, &bitunpack_b<30>, &bitunpack_b<31>,
        &bitunpack_b<32>, &bitunpack_b<33>, &bitunpack_b<34>, &bitunpack_b<35>,
        &bitunpack_b<36>, &bitunpack_b<37>, &bitunpack_b<38>, &bitunpack_b<39>,
        &bitunpack_b<40>, &bitunpack_b<41>, &bitunpack_b<42>, &bitunpack_b<43>,
        &bitunpack_b<44>, &bitunpack_b<45>, &bitunpack_b<46>, &bitunpack_b<47>,
        &bitunpack_b<48>, &bitunpack_b<49>, &bitunpack_b<50>, &bitunpack_b<51>,
        &bitunpack_b<52>, &bitunpack_b<53>, &bitunpack_b<54>, &bitunpack_b<55>,
        &bitunpack_b<56>, &bitunpack_b<57>, &bitunpack_b<58>, &bitunpack_b<59>,
        &bitunpack_b<60>, &bitunpack_b<61>, &bitunpack_b<62>, &bitunpack_b<63>,
        &bitunpack_b<64>,
    };
    // clang-format on
};

} // namespace turbopfor::scalar::detail
