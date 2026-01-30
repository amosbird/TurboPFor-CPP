#pragma once

#include <cstdint>
#include <immintrin.h>

#if defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE __attribute__((always_inline)) inline
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

} // namespace turbopfor::simd::detail
