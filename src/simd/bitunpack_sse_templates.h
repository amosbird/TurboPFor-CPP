#pragma once

#include <cstdint>
#include <immintrin.h>

#if defined(_MSC_VER)
#    define ALWAYS_INLINE __forceinline
#else
#    define ALWAYS_INLINE __attribute__((always_inline)) inline
#endif

namespace turbopfor::simd::detail
{

// Shuffle table for 4-bit population count / distribution
// Maps a 4-bit mask to a shuffle control that moves packed exceptions to correct lanes
// e.g. mask 0101 (binary) -> shuffle takes element 0 to lane 0, element 1 to lane 2.
// Wait, shuffle needs to move exceptions FROM packed TO sparse.
// If mask is 0101 (lanes 0 and 2 have exceptions).
// Packed exceptions: [E0, E1, x, x]
// Result: [E0, 0, E1, 0]
// Shuffle: 0, 0xFF, 1, 0xFF... (bytes)
// This table is standard TurboPFor _shuffle_128.
alignas(16) static const char _shuffle_128_table[16][16] = {
    {(char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff}, // 0
    {0,
     1,
     2,
     3,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff}, // 1
    {(char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     0,
     1,
     2,
     3,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff}, // 2
    {0, 1, 2, 3, 4, 5, 6, 7, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff}, // 3
    {(char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     0,
     1,
     2,
     3,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff}, // 4
    {0, 1, 2, 3, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 4, 5, 6, 7, (char)0xff, (char)0xff, (char)0xff, (char)0xff}, // 5
    {(char)0xff, (char)0xff, (char)0xff, (char)0xff, 0, 1, 2, 3, 4, 5, 6, 7, (char)0xff, (char)0xff, (char)0xff, (char)0xff}, // 6
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, (char)0xff, (char)0xff, (char)0xff, (char)0xff}, // 7
    {(char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     (char)0xff,
     0,
     1,
     2,
     3}, // 8
    {0, 1, 2, 3, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 4, 5, 6, 7}, // 9
    {(char)0xff, (char)0xff, (char)0xff, (char)0xff, 0, 1, 2, 3, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 4, 5, 6, 7}, // 10
    {0, 1, 2, 3, 4, 5, 6, 7, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 8, 9, 10, 11}, // 11
    {(char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 0, 1, 2, 3, 4, 5, 6, 7}, // 12
    {0, 1, 2, 3, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 4, 5, 6, 7, 8, 9, 10, 11}, // 13
    {(char)0xff, (char)0xff, (char)0xff, (char)0xff, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, // 14
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, // 15
};


template <unsigned B>
struct MaskGenSSE
{
    static constexpr uint32_t value = (1u << B) - 1u;
};

template <>
struct MaskGenSSE<32>
{
    static constexpr uint32_t value = 0xFFFFFFFFu;
};

template <unsigned B, unsigned G, unsigned MaxG, int CurrentLoadedIdx, bool Delta, bool Patching>
struct UnpackStepSSE
{
    static ALWAYS_INLINE void
    run(const __m128i *& ip,
        __m128i & iv,
        uint32_t * out,
        const __m128i & mask,
        const __m128i & cv,
        __m128i & sv,
        const uint64_t * bitmap,
        const uint32_t *& pex)
    {
        constexpr int TargetIdx = (G * B) / 32;
        constexpr int Offset = (G * B) % 32;
        constexpr bool Spans = (Offset + B > 32);

        // Load data if moved to new word (stripe)
        if (TargetIdx > CurrentLoadedIdx)
        {
            iv = _mm_loadu_si128(ip++);
        }

        __m128i ov;
        if (Offset == 0)
        {
            ov = iv;
        }
        else
        {
            ov = _mm_srli_epi32(iv, Offset);
        }

        if (Spans)
        {
            iv = _mm_loadu_si128(ip++);
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, BitsInFirst), mask));
        }
        else
        {
            if (B != 32)
            {
                ov = _mm_and_si128(ov, mask);
            }
        }

        if (Patching)
        {
            // Bitmap access for 128v (2 x 64-bit words)
            // G goes 0..31.
            uint64_t w = (G < 16) ? bitmap[0] : bitmap[1];
            unsigned shift = (G % 16) * 4;
            unsigned m = (w >> shift) & 0xF;

            // Always execute patching logic to avoid branch misprediction
            // For sparse exceptions, this adds overhead. For dense, it helps.
            // Since we only call this when exceptions exist, and misprediction is costly...
            // Let's test unconditional execution.
            // if (m) {
            __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
            __m128i exc_s = _mm_slli_epi32(exc, B);

            __m128i p_mask = _mm_load_si128(reinterpret_cast<const __m128i *>(_shuffle_128_table[m]));

            __m128i p = _mm_shuffle_epi8(exc_s, p_mask);
            ov = _mm_add_epi32(ov, p);

#if defined(__GNUC__) || defined(__clang__)
            pex += __builtin_popcount(m);
#else
            unsigned c = 0;
            unsigned tm = m;
            while (tm)
            {
                tm &= tm - 1;
                c++;
            }
            pex += c;
#endif
            // }
        }

        if (Delta)
        {
            // Prefix sum (scan)
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));

            // Add carry + offset
            ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(out + G * 4), ov);

            // Update sv (carry)
            int last = _mm_cvtsi128_si32(_mm_shuffle_epi32(ov, 0xFF));
            sv = _mm_set1_epi32(last);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(out + G * 4), ov);
        }

        constexpr int NextLoadedIdx = Spans ? TargetIdx + 1 : TargetIdx;
        UnpackStepSSE<B, G + 1, MaxG, NextLoadedIdx, Delta, Patching>::run(ip, iv, out, mask, cv, sv, bitmap, pex);
    }
};

// Base case
template <unsigned B, unsigned MaxG, int CurrentLoadedIdx, bool Delta, bool Patching>
struct UnpackStepSSE<B, MaxG, MaxG, CurrentLoadedIdx, Delta, Patching>
{
    static ALWAYS_INLINE void
    run(const __m128i *&, __m128i &, uint32_t *, const __m128i &, const __m128i &, __m128i &, const uint64_t *, const uint32_t *&)
    {
    }
};

// Entry point wrapper
template <unsigned B, unsigned Count, bool Delta, bool Patching>
ALWAYS_INLINE const unsigned char *
bitunpack_sse_entry(const unsigned char * in, uint32_t * out, __m128i & sv, const uint64_t * bitmap, const uint32_t * pex)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    if (B == 0)
    {
        if (Delta)
        {
            const __m128i cv = _mm_setr_epi32(1, 2, 3, 4);
            if (!Patching)
            {
                sv = _mm_add_epi32(sv, cv);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(out), sv);
                const __m128i four = _mm_set1_epi32(4);
                for (unsigned i = 1; i < MaxG; ++i)
                {
                    sv = _mm_add_epi32(sv, four);
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i * 4), sv);
                }
                return in;
            }
        }
        else
        {
            if (!Patching)
            {
                const __m128i zero = _mm_setzero_si128();
                for (unsigned i = 0; i < MaxG; ++i)
                {
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i * 4), zero);
                }
                return in;
            }
        }
    }

    const __m128i * ip = reinterpret_cast<const __m128i *>(in);
    __m128i iv = _mm_setzero_si128();

    const uint32_t mask_val = MaskGenSSE<B>::value;
    const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));

    // cv for Delta (1,2,3,4)
    const __m128i cv = Delta ? _mm_setr_epi32(1, 2, 3, 4) : _mm_setzero_si128();

    UnpackStepSSE<B, 0, MaxG, -1, Delta, Patching>::run(ip, iv, out, mask, cv, sv, bitmap, pex);

    return reinterpret_cast<const unsigned char *>(ip);
}

// ============================================================================
// STO64-fused unpack: writes 4×32-bit groups directly as 4×64-bit (zero-extended)
// to uint64_t* output, eliminating the temp buffer in bitunpack128v64.
//
// This replicates TurboPFor C's approach of redefining VO32 to STO64 before
// including the unpack template. Each group's store writes 2×__m128i (32 bytes)
// instead of 1×__m128i (16 bytes).
//
// Only non-delta, non-patching variant is needed — those paths use the temp buffer
// approach (the delta/patching paths are cold enough that the extra pass doesn't matter).
// ============================================================================

template <unsigned B, unsigned G, unsigned MaxG, int CurrentLoadedIdx>
struct UnpackStepSSE_STO64
{
    static ALWAYS_INLINE void run(const __m128i *& ip, __m128i & iv, __m128i *& op, const __m128i & mask, const __m128i & zv)
    {
        constexpr int TargetIdx = (G * B) / 32;
        constexpr int Offset = (G * B) % 32;
        constexpr bool Spans = (Offset + B > 32);

        // Load data if moved to new word (stripe)
        if (TargetIdx > CurrentLoadedIdx)
        {
            iv = _mm_loadu_si128(ip++);
        }

        __m128i ov;
        if (Offset == 0)
        {
            ov = iv;
        }
        else
        {
            ov = _mm_srli_epi32(iv, Offset);
        }

        if (Spans)
        {
            iv = _mm_loadu_si128(ip++);
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, BitsInFirst), mask));
        }
        else
        {
            if (B != 32)
            {
                ov = _mm_and_si128(ov, mask);
            }
        }

        // Reverse IP32 pair-swap: the encoded order is [v2,v3,v0,v1] due to the
        // IP32 shuffle in encode. Swap pairs back to get [v0,v1,v2,v3].
        ov = _mm_shuffle_epi32(ov, _MM_SHUFFLE(1, 0, 3, 2));

        // STO64: zero-extend 4×uint32 → 4×uint64 via two stores
        // ov = [v0, v1, v2, v3] (4×32-bit, now in correct order)
        // → [v0, 0, v1, 0] then [v2, 0, v3, 0] (2×__m128i, each 2×64-bit)
        _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
        _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));

        constexpr int NextLoadedIdx = Spans ? TargetIdx + 1 : TargetIdx;
        UnpackStepSSE_STO64<B, G + 1, MaxG, NextLoadedIdx>::run(ip, iv, op, mask, zv);
    }
};

// Base case
template <unsigned B, unsigned MaxG, int CurrentLoadedIdx>
struct UnpackStepSSE_STO64<B, MaxG, MaxG, CurrentLoadedIdx>
{
    static ALWAYS_INLINE void run(const __m128i *&, __m128i &, __m128i *&, const __m128i &, const __m128i &) { }
};

// Entry point for STO64-fused unpack
// Unpacks Count values at bit-width B, writing directly as zero-extended 64-bit values
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_entry(const unsigned char * in, uint64_t * out)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    __m128i * op = reinterpret_cast<__m128i *>(out);

    if (B == 0)
    {
        const __m128i zero = _mm_setzero_si128();
        for (unsigned i = 0; i < Count / 2; ++i)
        {
            _mm_storeu_si128(op++, zero);
        }
        return in;
    }

    const __m128i * ip = reinterpret_cast<const __m128i *>(in);
    __m128i iv = _mm_setzero_si128();

    const uint32_t mask_val = MaskGenSSE<B>::value;
    const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));
    const __m128i zv = _mm_setzero_si128();

    UnpackStepSSE_STO64<B, 0, MaxG, -1>::run(ip, iv, op, mask, zv);

    return reinterpret_cast<const unsigned char *>(ip);
}

// ============================================================================
// STO64-fused unpack + delta1: writes 4×32-bit groups as 4×64-bit (zero-extended)
// with integrated prefix-sum delta1 decode. This matches TurboPFor C's approach
// of fusing VO32 = mm_scani_epi32 + STO64 in the unpack template.
//
// The delta1 prefix sum operates in 32-bit domain (since b <= 32 means all
// deltas fit in 32 bits), then zero-extends to 64-bit for output. The carry
// is tracked in 32-bit, matching TurboPFor C's `sv = _mm_set1_epi32((uint32_t)start)`.
// ============================================================================

template <unsigned B, unsigned G, unsigned MaxG, int CurrentLoadedIdx>
struct UnpackStepSSE_STO64_D1
{
    static ALWAYS_INLINE void
    run(const __m128i *& ip, __m128i & iv, __m128i *& op, const __m128i & mask, const __m128i & cv, __m128i & sv, const __m128i & zv)
    {
        constexpr int TargetIdx = (G * B) / 32;
        constexpr int Offset = (G * B) % 32;
        constexpr bool Spans = (Offset + B > 32);

        // Load data if moved to new word (stripe)
        if (TargetIdx > CurrentLoadedIdx)
        {
            iv = _mm_loadu_si128(ip++);
        }

        __m128i ov;
        if (Offset == 0)
        {
            ov = iv;
        }
        else
        {
            ov = _mm_srli_epi32(iv, Offset);
        }

        if (Spans)
        {
            iv = _mm_loadu_si128(ip++);
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, BitsInFirst), mask));
        }
        else
        {
            if (B != 32)
            {
                ov = _mm_and_si128(ov, mask);
            }
        }

        // Reverse IP32 pair-swap before prefix sum: encoded order is [d2,d3,d0,d1],
        // need [d0,d1,d2,d3] for correct delta accumulation.
        ov = _mm_shuffle_epi32(ov, _MM_SHUFFLE(1, 0, 3, 2));

        // Delta1 prefix sum (mm_scani_epi32 equivalent):
        // 1. In-lane prefix sum: [a, a+b, a+b+c, a+b+c+d]
        ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
        ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
        // 2. Add carry from previous group + delta1 offset cv
        ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

        // STO64: zero-extend and store (values now in correct order)
        _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
        // 3. Update carry: broadcast last element (between the two stores)
        sv = _mm_shuffle_epi32(ov, 0xFF);
        _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));

        constexpr int NextLoadedIdx = Spans ? TargetIdx + 1 : TargetIdx;
        UnpackStepSSE_STO64_D1<B, G + 1, MaxG, NextLoadedIdx>::run(ip, iv, op, mask, cv, sv, zv);
    }
};

// Base case
template <unsigned B, unsigned MaxG, int CurrentLoadedIdx>
struct UnpackStepSSE_STO64_D1<B, MaxG, MaxG, CurrentLoadedIdx>
{
    static ALWAYS_INLINE void
    run(const __m128i *&, __m128i &, __m128i *&, const __m128i &, const __m128i &, __m128i &, const __m128i &)
    {
    }
};

// Entry point for STO64-fused unpack + delta1 (fully unrolled — ~86KB total)
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_d1_entry(const unsigned char * in, uint64_t * out, __m128i sv)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i cv = _mm_setr_epi32(1, 2, 3, 4);
    const __m128i zv = _mm_setzero_si128();

    if (B == 0)
    {
        // All deltas are 0, so delta1 gives consecutive values: start+1, start+2, ...
        sv = _mm_add_epi32(sv, cv);
        __m128i lo = _mm_unpacklo_epi32(sv, zv);
        __m128i hi = _mm_unpackhi_epi32(sv, zv);
        _mm_storeu_si128(op++, lo);
        _mm_storeu_si128(op++, hi);
        const __m128i four = _mm_set1_epi32(4);
        for (unsigned i = 1; i < MaxG; ++i)
        {
            sv = _mm_add_epi32(sv, four);
            lo = _mm_unpacklo_epi32(sv, zv);
            hi = _mm_unpackhi_epi32(sv, zv);
            _mm_storeu_si128(op++, lo);
            _mm_storeu_si128(op++, hi);
        }
        return in;
    }

    const __m128i * ip = reinterpret_cast<const __m128i *>(in);
    __m128i iv = _mm_setzero_si128();

    const uint32_t mask_val = MaskGenSSE<B>::value;
    const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));

    UnpackStepSSE_STO64_D1<B, 0, MaxG, -1>::run(ip, iv, op, mask, cv, sv, zv);

    return reinterpret_cast<const unsigned char *>(ip);
}

// ============================================================================
// PERIODIC-UNROLL STO64+D1 unpack: unrolls exactly one "period" of groups
// (P = 32/gcd(B,32)), then loops over periods. Within each period body, all
// shift amounts and load/span decisions are compile-time constants (because
// the group's relative index R is a template parameter).
//
// This produces code that:
// - Has IDENTICAL instruction quality to the fully-unrolled version (immediate
//   shifts, no variable-shift instructions)
// - Is much smaller: each case is ~P×30 bytes ≈ 100-400 bytes instead of ~2.7KB
// - Total switch function is ~5-15KB (well within 32KB L1i)
// - Has a small, perfectly-predictable loop back-edge (same trip count every call)
//
// The math: for bit width B processing 32 groups of 4 elements each (128 total),
// the bit offset pattern (G*B)%32 repeats with period P = 32/gcd(B,32).
// After P groups, exactly P*B/32 stripes have been consumed, and the pattern
// restarts. So we unroll P groups, then loop 32/P times.
//
// Examples:
//   B=1:  P=32, 1 iteration  (same as full unroll)
//   B=4:  P=8,  4 iterations
//   B=8:  P=4,  8 iterations
//   B=16: P=2,  16 iterations
//   B=28: P=8,  4 iterations (the former outlier!)
//   B=32: P=1,  32 iterations
// ============================================================================

// Compile-time GCD
template <unsigned A, unsigned B_>
struct GCD { static constexpr unsigned value = GCD<B_, A % B_>::value; };
template <unsigned A>
struct GCD<A, 0> { static constexpr unsigned value = A; };

// Period length for bit width B: P = 32 / gcd(B, 32)
// Special case: B=0 → P=32 (handled separately in entry point)
template <unsigned B>
struct PeriodLen { static constexpr unsigned value = (B == 0) ? 32 : (32 / GCD<B, 32>::value); };

// One step within a period: R is the relative group index (0 ≤ R < P).
// All bit offset calculations use R*B, giving compile-time constants.
// LoadedOffset tracks which relative stripe index was last loaded (relative
// to the period's starting stripe), starting at -1.
template <unsigned B, unsigned R, unsigned P, int RelLoadedIdx>
struct UnpackPeriodStepSSE_STO64_D1
{
    static ALWAYS_INLINE void
    run(const __m128i *& ip, __m128i & iv, __m128i *& op, const __m128i & mask, const __m128i & cv, __m128i & sv, const __m128i & zv)
    {
        // Bit position relative to period start
        constexpr unsigned RelBitPos = R * B;
        constexpr int RelTargetIdx = static_cast<int>(RelBitPos / 32);
        constexpr int Offset = RelBitPos % 32;
        constexpr bool Spans = (Offset + B > 32) && (B < 32);

        // Load new stripe if we've moved past the last loaded one
        if (RelTargetIdx > RelLoadedIdx)
        {
            iv = _mm_loadu_si128(ip++);
        }

        __m128i ov;
        if (Offset == 0)
        {
            ov = iv;
        }
        else
        {
            ov = _mm_srli_epi32(iv, Offset);
        }

        if (Spans)
        {
            iv = _mm_loadu_si128(ip++);
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, BitsInFirst), mask));
        }
        else
        {
            if (B != 32)
            {
                ov = _mm_and_si128(ov, mask);
            }
        }

        // Reverse IP32 pair-swap before prefix sum
        ov = _mm_shuffle_epi32(ov, _MM_SHUFFLE(1, 0, 3, 2));

        // Delta1 prefix sum
        ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
        ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
        ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

        // STO64: store + carry extract
        _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
        sv = _mm_shuffle_epi32(ov, 0xFF);
        _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));

        constexpr int NextRelLoadedIdx = Spans ? RelTargetIdx + 1 : RelTargetIdx;
        UnpackPeriodStepSSE_STO64_D1<B, R + 1, P, NextRelLoadedIdx>::run(ip, iv, op, mask, cv, sv, zv);
    }
};

// Base case: end of period
template <unsigned B, unsigned P, int RelLoadedIdx>
struct UnpackPeriodStepSSE_STO64_D1<B, P, P, RelLoadedIdx>
{
    static ALWAYS_INLINE void
    run(const __m128i *&, __m128i &, __m128i *&, const __m128i &, const __m128i &, __m128i &, const __m128i &)
    {
    }
};

// Body: unroll one full period of P groups
template <unsigned B>
ALWAYS_INLINE void bitunpack_sse_sto64_d1_period_body(
    const __m128i *& ip, __m128i & iv, __m128i *& op,
    const __m128i & mask, const __m128i & cv, __m128i & sv, const __m128i & zv)
{
    constexpr unsigned P = PeriodLen<B>::value;
    UnpackPeriodStepSSE_STO64_D1<B, 0, P, -1>::run(ip, iv, op, mask, cv, sv, zv);
}

// Entry point for periodic-unroll STO64+D1
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_d1_periodic_entry(const unsigned char * in, uint64_t * out, __m128i sv)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i cv = _mm_setr_epi32(1, 2, 3, 4);
    const __m128i zv = _mm_setzero_si128();

    if constexpr (B == 0)
    {
        // All deltas are 0, so delta1 gives consecutive values: start+1, start+2, ...
        sv = _mm_add_epi32(sv, cv);
        __m128i lo = _mm_unpacklo_epi32(sv, zv);
        __m128i hi = _mm_unpackhi_epi32(sv, zv);
        _mm_storeu_si128(op++, lo);
        _mm_storeu_si128(op++, hi);
        const __m128i four = _mm_set1_epi32(4);
        for (unsigned i = 1; i < MaxG; ++i)
        {
            sv = _mm_add_epi32(sv, four);
            lo = _mm_unpacklo_epi32(sv, zv);
            hi = _mm_unpackhi_epi32(sv, zv);
            _mm_storeu_si128(op++, lo);
            _mm_storeu_si128(op++, hi);
        }
        return in;
    }
    else
    {
        const __m128i * ip = reinterpret_cast<const __m128i *>(in);
        __m128i iv = _mm_setzero_si128();

        constexpr uint32_t mask_val = MaskGenSSE<B>::value;
        const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));

        constexpr unsigned P = PeriodLen<B>::value;
        constexpr unsigned NumPeriods = MaxG / P;
        static_assert(MaxG % P == 0, "MaxG must be divisible by period length P");

        for (unsigned period = 0; period < NumPeriods; ++period)
        {
            bitunpack_sse_sto64_d1_period_body<B>(ip, iv, op, mask, cv, sv, zv);
        }

        return reinterpret_cast<const unsigned char *>(ip);
    }
}

// ============================================================================
// PERIODIC-UNROLL STO64 unpack (non-delta): analogous to the periodic D1
// variant above, but without prefix-sum / carry tracking.
//
// The fully-unrolled bitunpack_sse_sto64_entry produces ~86KB total for the
// 33-case switch, exceeding L1i.  This periodic variant reduces code size to
// ~5-15KB while keeping identical instruction quality (immediate shifts, no
// variable-shift instructions), exactly like the D1 periodic variant.
// ============================================================================

template <unsigned B, unsigned R, unsigned P, int RelLoadedIdx>
struct UnpackPeriodStepSSE_STO64
{
    static ALWAYS_INLINE void
    run(const __m128i *& ip, __m128i & iv, __m128i *& op, const __m128i & mask, const __m128i & zv)
    {
        constexpr unsigned RelBitPos = R * B;
        constexpr int RelTargetIdx = static_cast<int>(RelBitPos / 32);
        constexpr int Offset = RelBitPos % 32;
        constexpr bool Spans = (Offset + B > 32) && (B < 32);

        if (RelTargetIdx > RelLoadedIdx)
        {
            iv = _mm_loadu_si128(ip++);
        }

        __m128i ov;
        if (Offset == 0)
        {
            ov = iv;
        }
        else
        {
            ov = _mm_srli_epi32(iv, Offset);
        }

        if (Spans)
        {
            iv = _mm_loadu_si128(ip++);
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, BitsInFirst), mask));
        }
        else
        {
            if (B != 32)
            {
                ov = _mm_and_si128(ov, mask);
            }
        }

        ov = _mm_shuffle_epi32(ov, _MM_SHUFFLE(1, 0, 3, 2));

        _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
        _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));

        constexpr int NextRelLoadedIdx = Spans ? RelTargetIdx + 1 : RelTargetIdx;
        UnpackPeriodStepSSE_STO64<B, R + 1, P, NextRelLoadedIdx>::run(ip, iv, op, mask, zv);
    }
};

template <unsigned B, unsigned P, int RelLoadedIdx>
struct UnpackPeriodStepSSE_STO64<B, P, P, RelLoadedIdx>
{
    static ALWAYS_INLINE void
    run(const __m128i *&, __m128i &, __m128i *&, const __m128i &, const __m128i &)
    {
    }
};

template <unsigned B>
ALWAYS_INLINE void bitunpack_sse_sto64_period_body(
    const __m128i *& ip, __m128i & iv, __m128i *& op,
    const __m128i & mask, const __m128i & zv)
{
    constexpr unsigned P = PeriodLen<B>::value;
    UnpackPeriodStepSSE_STO64<B, 0, P, -1>::run(ip, iv, op, mask, zv);
}

template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_periodic_entry(const unsigned char * in, uint64_t * out)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    __m128i * op = reinterpret_cast<__m128i *>(out);

    if constexpr (B == 0)
    {
        const __m128i zero = _mm_setzero_si128();
        for (unsigned i = 0; i < Count / 2; ++i)
        {
            _mm_storeu_si128(op++, zero);
        }
        return in;
    }
    else
    {
        const __m128i * ip = reinterpret_cast<const __m128i *>(in);
        __m128i iv = _mm_setzero_si128();

        constexpr uint32_t mask_val = MaskGenSSE<B>::value;
        const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));
        const __m128i zv = _mm_setzero_si128();

        constexpr unsigned P = PeriodLen<B>::value;
        constexpr unsigned NumPeriods = MaxG / P;
        static_assert(MaxG % P == 0, "MaxG must be divisible by period length P");

        for (unsigned period = 0; period < NumPeriods; ++period)
        {
            bitunpack_sse_sto64_period_body<B>(ip, iv, op, mask, zv);
        }

        return reinterpret_cast<const unsigned char *>(ip);
    }
}

// ============================================================================
// LOOP-BASED STO64+D1 unpack: same semantics as UnpackStepSSE_STO64_D1 but
// uses a runtime loop over the 32 groups instead of recursive template unrolling.
//
// MOTIVATION: The fully-unrolled version produces ~86KB total for the switch
// function (33 cases × ~2.7KB each), which is ~2.7× the 32KB L1 instruction
// cache. This causes unpredictable iTLB/L1i cache pressure that manifests as
// random per-bit-width performance outliers (the "alignment lottery").
//
// This version keeps B as a compile-time template parameter (so shifts, masks,
// and bit-offset computations use compile-time constants where possible) but
// iterates over groups at runtime. Each case should be ~200-300 bytes instead
// of ~2700 bytes, shrinking the total function to ~8-10KB — well within L1i.
//
// The loop branches (back-edge, load check, spans check) are all perfectly
// predictable for a given B since the bit pattern repeats identically every
// iteration.
// ============================================================================

template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_d1_loop_entry(const unsigned char * in, uint64_t * out, __m128i sv)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i cv = _mm_setr_epi32(1, 2, 3, 4);
    const __m128i zv = _mm_setzero_si128();

    if constexpr (B == 0)
    {
        // All deltas are 0, so delta1 gives consecutive values: start+1, start+2, ...
        sv = _mm_add_epi32(sv, cv);
        __m128i lo = _mm_unpacklo_epi32(sv, zv);
        __m128i hi = _mm_unpackhi_epi32(sv, zv);
        _mm_storeu_si128(op++, lo);
        _mm_storeu_si128(op++, hi);
        const __m128i four = _mm_set1_epi32(4);
        for (unsigned i = 1; i < MaxG; ++i)
        {
            sv = _mm_add_epi32(sv, four);
            lo = _mm_unpacklo_epi32(sv, zv);
            hi = _mm_unpackhi_epi32(sv, zv);
            _mm_storeu_si128(op++, lo);
            _mm_storeu_si128(op++, hi);
        }
        return in;
    }
    else
    {
        const __m128i * ip = reinterpret_cast<const __m128i *>(in);
        __m128i iv = _mm_setzero_si128();

        constexpr uint32_t mask_val = MaskGenSSE<B>::value;
        const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));

        // Runtime loop over 32 groups. B is compile-time, so the compiler knows
        // all shift amounts and can optimize the bit-offset arithmetic.
        // lastLoadedIdx tracks which __m128i stripe we last loaded (like
        // the template parameter CurrentLoadedIdx in the unrolled version).
        int lastLoadedIdx = -1;

        for (unsigned g = 0; g < MaxG; ++g)
        {
            // Compute bit position for group g with compile-time B
            const unsigned bitPos = g * B;
            const int targetIdx = static_cast<int>(bitPos / 32u);
            const unsigned offset = bitPos % 32u;

            // Load new stripe if needed
            if (targetIdx > lastLoadedIdx)
            {
                iv = _mm_loadu_si128(ip++);
                lastLoadedIdx = targetIdx;
            }

            // Extract and shift
            __m128i ov;
            if (offset == 0)
            {
                ov = iv;
            }
            else
            {
                ov = _mm_srli_epi32(iv, offset);
            }

            // Check if value spans two stripes
            if constexpr (B < 32)
            {
                if (offset + B > 32u)
                {
                    // Spans: need bits from next stripe
                    iv = _mm_loadu_si128(ip++);
                    lastLoadedIdx = targetIdx + 1;
                    const unsigned bitsInFirst = 32u - offset;
                    ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, bitsInFirst), mask));
                }
                else
                {
                    ov = _mm_and_si128(ov, mask);
                }
            }
            // B == 32: no masking needed (full 32-bit word)

            // Reverse IP32 pair-swap before prefix sum
            ov = _mm_shuffle_epi32(ov, _MM_SHUFFLE(1, 0, 3, 2));

            // Delta1 prefix sum (mm_scani_epi32 equivalent)
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
            ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

            // STO64: zero-extend and store, with carry extraction between stores
            _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
            sv = _mm_shuffle_epi32(ov, 0xFF);
            _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));
        }

        return reinterpret_cast<const unsigned char *>(ip);
    }
}

// ============================================================================
// STO64-fused unpack + delta1 + exception patching: fuses all three operations
// (bitunpack + SSSE3 exception merge + delta1 prefix scan + STO64 zero-extend)
// into a single pass. Matches TurboPFor C's _bitd1unpack128v64 approach.
// ============================================================================

template <unsigned B, unsigned G, unsigned MaxG, int CurrentLoadedIdx>
struct UnpackStepSSE_STO64_D1_EX
{
    static ALWAYS_INLINE void run(
        const __m128i *& ip,
        __m128i & iv,
        __m128i *& op,
        const __m128i & mask,
        const __m128i & cv,
        __m128i & sv,
        const __m128i & zv,
        const uint64_t * bitmap,
        const uint32_t *& pex)
    {
        constexpr int TargetIdx = (G * B) / 32;
        constexpr int Offset = (G * B) % 32;
        constexpr bool Spans = (Offset + B > 32);

        // Load data if moved to new word (stripe)
        if (TargetIdx > CurrentLoadedIdx)
        {
            iv = _mm_loadu_si128(ip++);
        }

        __m128i ov;
        if (Offset == 0)
        {
            ov = iv;
        }
        else
        {
            ov = _mm_srli_epi32(iv, Offset);
        }

        if (Spans)
        {
            iv = _mm_loadu_si128(ip++);
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, BitsInFirst), mask));
        }
        else
        {
            if (B != 32)
            {
                ov = _mm_and_si128(ov, mask);
            }
        }

        // Reverse IP32 pair-swap: must happen before exception patching and delta1,
        // since the bitmap positions correspond to logical output order.
        ov = _mm_shuffle_epi32(ov, _MM_SHUFFLE(1, 0, 3, 2));

        // Exception patching via SSSE3 shuffle (same as 32-bit patching path)
        // Bitmap layout: 2 x 64-bit words for 128 values, 4 bits per group of 4
        {
            uint64_t w = (G < 16) ? bitmap[0] : bitmap[1];
            unsigned shift = (G % 16) * 4;
            unsigned m = (w >> shift) & 0xF;

            __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
            __m128i exc_s = _mm_slli_epi32(exc, B);
            __m128i p_mask = _mm_load_si128(reinterpret_cast<const __m128i *>(_shuffle_128_table[m]));
            __m128i p = _mm_shuffle_epi8(exc_s, p_mask);
            ov = _mm_add_epi32(ov, p);

#if defined(__GNUC__) || defined(__clang__)
            pex += __builtin_popcount(m);
#else
            unsigned c = 0;
            unsigned tm = m;
            while (tm)
            {
                tm &= tm - 1;
                c++;
            }
            pex += c;
#endif
        }

        // Delta1 prefix sum (mm_scani_epi32 equivalent)
        ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
        ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
        ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

        // STO64: store low half first, then carry extract, then high half
        _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
        sv = _mm_shuffle_epi32(ov, 0xFF);
        _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));

        constexpr int NextLoadedIdx = Spans ? TargetIdx + 1 : TargetIdx;
        UnpackStepSSE_STO64_D1_EX<B, G + 1, MaxG, NextLoadedIdx>::run(ip, iv, op, mask, cv, sv, zv, bitmap, pex);
    }
};

// Base case
template <unsigned B, unsigned MaxG, int CurrentLoadedIdx>
struct UnpackStepSSE_STO64_D1_EX<B, MaxG, MaxG, CurrentLoadedIdx>
{
    static ALWAYS_INLINE void run(
        const __m128i *&,
        __m128i &,
        __m128i *&,
        const __m128i &,
        const __m128i &,
        __m128i &,
        const __m128i &,
        const uint64_t *,
        const uint32_t *&)
    {
    }
};

// Entry point for STO64-fused unpack + delta1 + exception patching (fully unrolled)
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char *
bitunpack_sse_sto64_d1_ex_entry(const unsigned char * in, uint64_t * out, __m128i sv, const uint64_t * bitmap, const uint32_t *& pex)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i cv = _mm_setr_epi32(1, 2, 3, 4);
    const __m128i zv = _mm_setzero_si128();

    if (B == 0)
    {
        // B=0 with exceptions: zero base + patched exceptions + delta1
        const __m128i four = _mm_set1_epi32(4);
        for (unsigned g = 0; g < MaxG; ++g)
        {
            __m128i ov = _mm_setzero_si128();

            // Exception patching
            uint64_t w = (g < 16) ? bitmap[0] : bitmap[1];
            unsigned shift = (g % 16) * 4;
            unsigned m = (w >> shift) & 0xF;

            __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
            // B=0 so no shift needed, but exc_s = exc << 0 = exc
            __m128i p_mask = _mm_load_si128(reinterpret_cast<const __m128i *>(_shuffle_128_table[m]));
            ov = _mm_shuffle_epi8(exc, p_mask);
#if defined(__GNUC__) || defined(__clang__)
            pex += __builtin_popcount(m);
#else
            unsigned c = 0;
            unsigned tm = m;
            while (tm)
            {
                tm &= tm - 1;
                c++;
            }
            pex += c;
#endif

            // Delta1 prefix sum
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
            ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

            // STO64
            _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
            sv = _mm_shuffle_epi32(ov, 0xFF);
            _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));
        }
        return in;
    }

    const __m128i * ip = reinterpret_cast<const __m128i *>(in);
    __m128i iv = _mm_setzero_si128();

    const uint32_t mask_val = MaskGenSSE<B>::value;
    const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));

    UnpackStepSSE_STO64_D1_EX<B, 0, MaxG, -1>::run(ip, iv, op, mask, cv, sv, zv, bitmap, pex);

    return reinterpret_cast<const unsigned char *>(ip);
}

// ============================================================================
// PERIODIC-UNROLL STO64+D1+EX unpack: same periodic approach as D1, but with
// exception patching. The bitmap index uses a runtime groupBase parameter
// since the absolute group index is needed for bitmap word selection.
// ============================================================================

template <unsigned B, unsigned R, unsigned P, int RelLoadedIdx>
struct UnpackPeriodStepSSE_STO64_D1_EX
{
    static ALWAYS_INLINE void run(
        const __m128i *& ip,
        __m128i & iv,
        __m128i *& op,
        const __m128i & mask,
        const __m128i & cv,
        __m128i & sv,
        const __m128i & zv,
        const uint64_t * bitmap,
        const uint32_t *& pex,
        unsigned absG)
    {
        constexpr unsigned RelBitPos = R * B;
        constexpr int RelTargetIdx = static_cast<int>(RelBitPos / 32);
        constexpr int Offset = RelBitPos % 32;
        constexpr bool Spans = (Offset + B > 32) && (B < 32);

        if (RelTargetIdx > RelLoadedIdx)
        {
            iv = _mm_loadu_si128(ip++);
        }

        __m128i ov;
        if (Offset == 0)
        {
            ov = iv;
        }
        else
        {
            ov = _mm_srli_epi32(iv, Offset);
        }

        if (Spans)
        {
            iv = _mm_loadu_si128(ip++);
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, BitsInFirst), mask));
        }
        else
        {
            if (B != 32)
            {
                ov = _mm_and_si128(ov, mask);
            }
        }

        // Reverse IP32 pair-swap before exception patching and delta1
        ov = _mm_shuffle_epi32(ov, _MM_SHUFFLE(1, 0, 3, 2));

        // Exception patching: use absolute group index for bitmap
        {
            uint64_t w = (absG < 16) ? bitmap[0] : bitmap[1];
            unsigned shift = (absG % 16) * 4;
            unsigned m = (w >> shift) & 0xF;

            __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
            __m128i exc_s = _mm_slli_epi32(exc, B);
            __m128i p_mask = _mm_load_si128(reinterpret_cast<const __m128i *>(_shuffle_128_table[m]));
            __m128i p = _mm_shuffle_epi8(exc_s, p_mask);
            ov = _mm_add_epi32(ov, p);

#if defined(__GNUC__) || defined(__clang__)
            pex += __builtin_popcount(m);
#else
            unsigned c = 0;
            unsigned tm = m;
            while (tm) { tm &= tm - 1; c++; }
            pex += c;
#endif
        }

        // Delta1 prefix sum
        ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
        ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
        ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

        // STO64
        _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
        sv = _mm_shuffle_epi32(ov, 0xFF);
        _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));

        constexpr int NextRelLoadedIdx = Spans ? RelTargetIdx + 1 : RelTargetIdx;
        UnpackPeriodStepSSE_STO64_D1_EX<B, R + 1, P, NextRelLoadedIdx>::run(
            ip, iv, op, mask, cv, sv, zv, bitmap, pex, absG + 1);
    }
};

template <unsigned B, unsigned P, int RelLoadedIdx>
struct UnpackPeriodStepSSE_STO64_D1_EX<B, P, P, RelLoadedIdx>
{
    static ALWAYS_INLINE void run(
        const __m128i *&, __m128i &, __m128i *&, const __m128i &, const __m128i &,
        __m128i &, const __m128i &, const uint64_t *, const uint32_t *&, unsigned)
    {
    }
};

template <unsigned B>
ALWAYS_INLINE void bitunpack_sse_sto64_d1_ex_period_body(
    const __m128i *& ip, __m128i & iv, __m128i *& op,
    const __m128i & mask, const __m128i & cv, __m128i & sv, const __m128i & zv,
    const uint64_t * bitmap, const uint32_t *& pex, unsigned absGroupBase)
{
    constexpr unsigned P = PeriodLen<B>::value;
    UnpackPeriodStepSSE_STO64_D1_EX<B, 0, P, -1>::run(
        ip, iv, op, mask, cv, sv, zv, bitmap, pex, absGroupBase);
}

// Periodic entry point for D1+EX
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char *
bitunpack_sse_sto64_d1_ex_periodic_entry(const unsigned char * in, uint64_t * out, __m128i sv, const uint64_t * bitmap, const uint32_t *& pex)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i cv = _mm_setr_epi32(1, 2, 3, 4);
    const __m128i zv = _mm_setzero_si128();

    if constexpr (B == 0)
    {
        for (unsigned g = 0; g < MaxG; ++g)
        {
            __m128i ov = _mm_setzero_si128();

            uint64_t w = (g < 16) ? bitmap[0] : bitmap[1];
            unsigned shift = (g % 16) * 4;
            unsigned m = (w >> shift) & 0xF;

            __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
            __m128i p_mask = _mm_load_si128(reinterpret_cast<const __m128i *>(_shuffle_128_table[m]));
            ov = _mm_shuffle_epi8(exc, p_mask);
#if defined(__GNUC__) || defined(__clang__)
            pex += __builtin_popcount(m);
#else
            unsigned c = 0;
            unsigned tm = m;
            while (tm) { tm &= tm - 1; c++; }
            pex += c;
#endif

            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
            ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

            _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
            sv = _mm_shuffle_epi32(ov, 0xFF);
            _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));
        }
        return in;
    }
    else
    {
        const __m128i * ip = reinterpret_cast<const __m128i *>(in);
        __m128i iv = _mm_setzero_si128();

        constexpr uint32_t mask_val = MaskGenSSE<B>::value;
        const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));

        constexpr unsigned P = PeriodLen<B>::value;
        constexpr unsigned NumPeriods = MaxG / P;
        static_assert(MaxG % P == 0, "MaxG must be divisible by period length P");

        for (unsigned period = 0; period < NumPeriods; ++period)
        {
            bitunpack_sse_sto64_d1_ex_period_body<B>(
                ip, iv, op, mask, cv, sv, zv, bitmap, pex, period * P);
        }

        return reinterpret_cast<const unsigned char *>(ip);
    }
}

// ============================================================================
// LOOP-BASED STO64+D1+EX unpack: same semantics as the fully-unrolled D1_EX
// but uses a runtime loop over groups. Same rationale as the D1 loop variant.
// ============================================================================

template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char *
bitunpack_sse_sto64_d1_ex_loop_entry(const unsigned char * in, uint64_t * out, __m128i sv, const uint64_t * bitmap, const uint32_t *& pex)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    __m128i * op = reinterpret_cast<__m128i *>(out);
    const __m128i cv = _mm_setr_epi32(1, 2, 3, 4);
    const __m128i zv = _mm_setzero_si128();

    if constexpr (B == 0)
    {
        // B=0 with exceptions: zero base + patched exceptions + delta1
        for (unsigned g = 0; g < MaxG; ++g)
        {
            __m128i ov = _mm_setzero_si128();

            uint64_t w = (g < 16) ? bitmap[0] : bitmap[1];
            unsigned shift = (g % 16) * 4;
            unsigned m = (w >> shift) & 0xF;

            __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
            __m128i p_mask = _mm_load_si128(reinterpret_cast<const __m128i *>(_shuffle_128_table[m]));
            ov = _mm_shuffle_epi8(exc, p_mask);
#if defined(__GNUC__) || defined(__clang__)
            pex += __builtin_popcount(m);
#else
            unsigned c = 0;
            unsigned tm = m;
            while (tm) { tm &= tm - 1; c++; }
            pex += c;
#endif

            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
            ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

            _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
            sv = _mm_shuffle_epi32(ov, 0xFF);
            _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));
        }
        return in;
    }
    else
    {
        const __m128i * ip = reinterpret_cast<const __m128i *>(in);
        __m128i iv = _mm_setzero_si128();

        constexpr uint32_t mask_val = MaskGenSSE<B>::value;
        const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));

        int lastLoadedIdx = -1;

        for (unsigned g = 0; g < MaxG; ++g)
        {
            const unsigned bitPos = g * B;
            const int targetIdx = static_cast<int>(bitPos / 32u);
            const unsigned offset = bitPos % 32u;

            if (targetIdx > lastLoadedIdx)
            {
                iv = _mm_loadu_si128(ip++);
                lastLoadedIdx = targetIdx;
            }

            __m128i ov;
            if (offset == 0)
                ov = iv;
            else
                ov = _mm_srli_epi32(iv, offset);

            if constexpr (B < 32)
            {
                if (offset + B > 32u)
                {
                    iv = _mm_loadu_si128(ip++);
                    lastLoadedIdx = targetIdx + 1;
                    const unsigned bitsInFirst = 32u - offset;
                    ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, bitsInFirst), mask));
                }
                else
                {
                    ov = _mm_and_si128(ov, mask);
                }
            }

            // Reverse IP32 pair-swap before exception patching and delta1
            ov = _mm_shuffle_epi32(ov, _MM_SHUFFLE(1, 0, 3, 2));

            // Exception patching
            {
                uint64_t w = (g < 16) ? bitmap[0] : bitmap[1];
                unsigned shift = (g % 16) * 4;
                unsigned m = (w >> shift) & 0xF;

                __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
                __m128i exc_s = _mm_slli_epi32(exc, B);
                __m128i p_mask = _mm_load_si128(reinterpret_cast<const __m128i *>(_shuffle_128_table[m]));
                __m128i p = _mm_shuffle_epi8(exc_s, p_mask);
                ov = _mm_add_epi32(ov, p);

#if defined(__GNUC__) || defined(__clang__)
                pex += __builtin_popcount(m);
#else
                unsigned c = 0;
                unsigned tm = m;
                while (tm) { tm &= tm - 1; c++; }
                pex += c;
#endif
            }

            // Delta1 prefix sum
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
            ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

            // STO64
            _mm_storeu_si128(op++, _mm_unpacklo_epi32(ov, zv));
            sv = _mm_shuffle_epi32(ov, 0xFF);
            _mm_storeu_si128(op++, _mm_unpackhi_epi32(ov, zv));
        }

        return reinterpret_cast<const unsigned char *>(ip);
    }
}

// ============================================================================
// STO64-fused unpack + delta1 with 64-BIT ACCUMULATION:
//
// Same SIMD bitunpack as the 32-bit D1 variant, but uses scalar 64-bit prefix
// sum instead of 32-bit SIMD prefix sum. This avoids overflow when
// start + sum_of_deltas + count > UINT32_MAX (which happens for b>=25 with
// random data, or any b with large start values).
//
// Uses periodic-unroll: for each group of 4 elements, the template does:
//   1. SIMD unpack 4 × B-bit values (compile-time shifts, same as 32-bit path)
//   2. IP32 pair-swap reversal
//   3. Extract 4 scalars via _mm_cvtsi128_si32 / _mm_extract_epi32
//   4. 64-bit scalar prefix sum accumulation
//   5. SIMD store via _mm_set_epi64x + _mm_storeu_si128
//
// Performance is close to the 32-bit fused path since the bottleneck is
// memory stores (2×128-bit per group = 32 bytes), not the prefix sum.
// ============================================================================

template <unsigned B, unsigned R, unsigned P, int RelLoadedIdx>
struct UnpackPeriodStepSSE_STO64_D1_64ACC
{
    static ALWAYS_INLINE void
    run(const __m128i *& ip, __m128i & iv, uint64_t *& op, const __m128i & mask, uint64_t & acc, unsigned & groupIdx)
    {
        constexpr unsigned RelBitPos = R * B;
        constexpr int RelTargetIdx = static_cast<int>(RelBitPos / 32);
        constexpr int Offset = RelBitPos % 32;
        constexpr bool Spans = (Offset + B > 32) && (B < 32);

        if (RelTargetIdx > RelLoadedIdx)
        {
            iv = _mm_loadu_si128(ip++);
        }

        __m128i ov;
        if (Offset == 0)
        {
            ov = iv;
        }
        else
        {
            ov = _mm_srli_epi32(iv, Offset);
        }

        if (Spans)
        {
            iv = _mm_loadu_si128(ip++);
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm_or_si128(ov, _mm_and_si128(_mm_slli_epi32(iv, BitsInFirst), mask));
        }
        else
        {
            if (B != 32)
            {
                ov = _mm_and_si128(ov, mask);
            }
        }

        // Reverse IP32 pair-swap
        ov = _mm_shuffle_epi32(ov, _MM_SHUFFLE(1, 0, 3, 2));

        // Extract 4 deltas as scalars, do 64-bit prefix sum
        uint32_t d0 = static_cast<uint32_t>(_mm_cvtsi128_si32(ov));
        uint32_t d1 = static_cast<uint32_t>(_mm_extract_epi32(ov, 1));
        uint32_t d2 = static_cast<uint32_t>(_mm_extract_epi32(ov, 2));
        uint32_t d3 = static_cast<uint32_t>(_mm_extract_epi32(ov, 3));

        const unsigned base1 = groupIdx * 4u + 1u;
        uint64_t v0 = (acc += d0) + base1;
        uint64_t v1 = (acc += d1) + base1 + 1u;
        uint64_t v2 = (acc += d2) + base1 + 2u;
        uint64_t v3 = (acc += d3) + base1 + 3u;

        // Store 4×64-bit output using SIMD (2 stores)
        _mm_storeu_si128(reinterpret_cast<__m128i *>(op),
                         _mm_set_epi64x(static_cast<int64_t>(v1), static_cast<int64_t>(v0)));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(op + 2),
                         _mm_set_epi64x(static_cast<int64_t>(v3), static_cast<int64_t>(v2)));
        op += 4;
        ++groupIdx;

        constexpr int NextRelLoadedIdx = Spans ? RelTargetIdx + 1 : RelTargetIdx;
        UnpackPeriodStepSSE_STO64_D1_64ACC<B, R + 1, P, NextRelLoadedIdx>::run(ip, iv, op, mask, acc, groupIdx);
    }
};

// Base case: end of period
template <unsigned B, unsigned P, int RelLoadedIdx>
struct UnpackPeriodStepSSE_STO64_D1_64ACC<B, P, P, RelLoadedIdx>
{
    static ALWAYS_INLINE void
    run(const __m128i *&, __m128i &, uint64_t *&, const __m128i &, uint64_t &, unsigned &)
    {
    }
};

// Body: unroll one full period
template <unsigned B>
ALWAYS_INLINE void bitunpack_sse_sto64_d1_64acc_period_body(
    const __m128i *& ip, __m128i & iv, uint64_t *& op, const __m128i & mask, uint64_t & acc, unsigned & groupIdx)
{
    constexpr unsigned P = PeriodLen<B>::value;
    UnpackPeriodStepSSE_STO64_D1_64ACC<B, 0, P, -1>::run(ip, iv, op, mask, acc, groupIdx);
}

// Entry point for periodic-unroll STO64+D1 with 64-bit accumulation
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_d1_64acc_entry(const unsigned char * in, uint64_t * out, uint64_t start)
{
    constexpr unsigned MaxG = Count / 4;
    static_assert(Count % 4 == 0, "Count must be multiple of 4");

    if constexpr (B == 0)
    {
        uint64_t acc = start;
        for (unsigned i = 0; i < Count; ++i)
            out[i] = ++acc + i;
        return in;
    }
    else
    {
        const __m128i * ip = reinterpret_cast<const __m128i *>(in);
        __m128i iv = _mm_setzero_si128();
        uint64_t * op = out;
        uint64_t acc = start;
        unsigned groupIdx = 0;

        constexpr uint32_t mask_val = MaskGenSSE<B>::value;
        const __m128i mask = _mm_set1_epi32(static_cast<int>(mask_val));

        constexpr unsigned P = PeriodLen<B>::value;
        constexpr unsigned NumPeriods = MaxG / P;
        static_assert(MaxG % P == 0, "MaxG must be divisible by period length P");

        for (unsigned period = 0; period < NumPeriods; ++period)
        {
            bitunpack_sse_sto64_d1_64acc_period_body<B>(ip, iv, op, mask, acc, groupIdx);
        }

        return reinterpret_cast<const unsigned char *>(ip);
    }
}

// Compile-time selector: use 64-bit accumulation for P > 2, fully-unrolled for P <= 2
template <unsigned B, unsigned Count>
ALWAYS_INLINE const unsigned char * bitunpack_sse_sto64_d1_64acc_hybrid_entry(const unsigned char * in, uint64_t * out, uint64_t start)
{
    constexpr unsigned P = PeriodLen<B>::value;
    if constexpr (B == 0 || P <= 2)
        return bitunpack_sse_sto64_d1_64acc_entry<B, Count>(in, out, start);
    else
        return bitunpack_sse_sto64_d1_64acc_entry<B, Count>(in, out, start);
}

// ============================================================================
// Fused IP32 bitpack templates: single-pass encode for 128v64
//
// IP32 shuffle extracts the low 32 bits from 4 uint64_t values by loading
// two __m128i, shuffling each, and OR-ing. This is fused directly into the
// bitpack loop, eliminating the temp buffer and its cache traffic.
//
// Uses the same periodic-unroll approach as decode: for bit width B, the
// shift-offset pattern repeats with period P = 32/gcd(B,32). Each step
// within the period has compile-time shift constants.
// ============================================================================

// One step within a period for fused IP32 bitpacking.
// R = relative group index within period (0 ≤ R < P).
// All offsets are compile-time constants.
template <unsigned B, unsigned R, unsigned P>
struct PackPeriodStepSSE_IP32
{
    static ALWAYS_INLINE void
    run(const __m128i *& ip, __m128i *& op, __m128i & ov)
    {
        // Inline IP32 shuffle: load 4 uint64_t (2 × __m128i), extract low 32 bits
        __m128i lo = _mm_loadu_si128(ip++);
        __m128i hi = _mm_loadu_si128(ip++);
        __m128i iv = _mm_or_si128(
            _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 0, 3, 1)),
            _mm_shuffle_epi32(hi, _MM_SHUFFLE(3, 1, 2, 0)));

        // No masking needed: the p4 encoder guarantees input values fit in B bits.
        // TurboPFor C's BITPACK128V32_N macros also omit the mask.

        // Bit position relative to period start — all compile-time
        constexpr unsigned Offset = (R * B) % 32;
        constexpr unsigned End = Offset + B;
        constexpr bool Spans = (End > 32) && (B < 32);

        // Accumulate: shift left by compile-time constant and OR into output
        if constexpr (Offset == 0)
            ov = iv;
        else
            ov = _mm_or_si128(ov, _mm_slli_epi32(iv, Offset));

        // Store when we've filled (or overfilled) 32 bits
        if constexpr (Spans || End == 32)
        {
            _mm_storeu_si128(op++, ov);
            if constexpr (Spans)
                ov = _mm_srli_epi32(iv, static_cast<int>(B - (End - 32)));
            // else: ov will be overwritten by next step's Offset==0 case
        }

        // Recurse to next step in period
        PackPeriodStepSSE_IP32<B, R + 1, P>::run(ip, op, ov);
    }
};

// Base case: end of period
template <unsigned B, unsigned P>
struct PackPeriodStepSSE_IP32<B, P, P>
{
    static ALWAYS_INLINE void
    run(const __m128i *&, __m128i *&, __m128i &)
    {
    }
};

// One full period body for bitpacking
template <unsigned B>
ALWAYS_INLINE void bitpack_ip32_period_body(
    const __m128i *& ip, __m128i *& op, __m128i & ov)
{
    constexpr unsigned P = PeriodLen<B>::value;
    PackPeriodStepSSE_IP32<B, 0, P>::run(ip, op, ov);
}

// Entry point for fused IP32 bitpack with periodic-unroll
template <unsigned B>
ALWAYS_INLINE void bitpack128v64_fused(const uint64_t * in, unsigned char * out)
{
    if constexpr (B == 0)
        return; // No output for 0-bit packing

    const __m128i * ip = reinterpret_cast<const __m128i *>(in);
    __m128i * op = reinterpret_cast<__m128i *>(out);

    __m128i ov = _mm_setzero_si128();

    if constexpr (B == 32)
    {
        // Special case: no shifting needed, just IP32 shuffle + store
        for (unsigned g = 0; g < 32; ++g)
        {
            __m128i lo = _mm_loadu_si128(ip++);
            __m128i hi = _mm_loadu_si128(ip++);
            __m128i iv = _mm_or_si128(
                _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 0, 3, 1)),
                _mm_shuffle_epi32(hi, _MM_SHUFFLE(3, 1, 2, 0)));
            _mm_storeu_si128(op++, iv);
        }
    }
    else
    {
        // Periodic-unroll: loop over 32/P periods, each unrolling P groups
        constexpr unsigned P = PeriodLen<B>::value;
        constexpr unsigned NumPeriods = 32 / P;

        for (unsigned p = 0; p < NumPeriods; ++p)
        {
            bitpack_ip32_period_body<B>(ip, op, ov);
        }

        // 32 groups × B bits = 32B bits = B stripes of 32 bits exactly.
        // The periodic unroll handles exactly 32 groups, so no tail flush needed.
    }
}

} // namespace turbopfor::simd::detail
