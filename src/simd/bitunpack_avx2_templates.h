#pragma once

#include <cstdint>
#include <immintrin.h>
#include "p4_simd_internal_256v.h"

namespace turbopfor::simd::detail
{

// Extern the shuffle table (defined in bitunpack128v32_simd.cpp or similar)
// We need it here. If not available, we define it.
// To avoid linker errors, we can use a static const in header or weak symbol.
// Or just replicate it as a static member.
alignas(16) static const char _shuffle_avx2[16][16] = {
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
     (char)0xff},
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
     (char)0xff},
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
     (char)0xff},
    {0, 1, 2, 3, 4, 5, 6, 7, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff},
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
     (char)0xff},
    {0, 1, 2, 3, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 4, 5, 6, 7, (char)0xff, (char)0xff, (char)0xff, (char)0xff},
    {(char)0xff, (char)0xff, (char)0xff, (char)0xff, 0, 1, 2, 3, 4, 5, 6, 7, (char)0xff, (char)0xff, (char)0xff, (char)0xff},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, (char)0xff, (char)0xff, (char)0xff, (char)0xff},
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
     3},
    {0, 1, 2, 3, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 4, 5, 6, 7},
    {(char)0xff, (char)0xff, (char)0xff, (char)0xff, 0, 1, 2, 3, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 8, 9, 10, 11},
    {(char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, (char)0xff, (char)0xff, (char)0xff, (char)0xff, 4, 5, 6, 7, 8, 9, 10, 11},
    {(char)0xff, (char)0xff, (char)0xff, (char)0xff, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
};

template <unsigned B>
struct MaskGen
{
    static constexpr uint32_t value = (1u << B) - 1u;
};

template <>
struct MaskGen<32>
{
    static constexpr uint32_t value = 0xFFFFFFFFu;
};

template <unsigned B, unsigned G, unsigned MaxG, int CurrentLoadedIdx, bool Delta, bool Patching>
struct UnpackStep
{
    static ALWAYS_INLINE void
    run(const __m256i *& ip,
        __m256i & iv,
        uint32_t * out,
        const __m256i & mask,
        const __m256i & cv,
        __m256i & sv,
        const uint64_t * bitmap,
        const uint32_t *& pex)
    {
        constexpr int TargetIdx = (G * B) / 32;
        constexpr int Offset = (G * B) % 32;
        constexpr bool Spans = (Offset + B > 32);

        // Load data if moved to new word
        if (TargetIdx > CurrentLoadedIdx)
        {
            iv = _mm256_loadu_si256(ip++);
        }

        __m256i ov;
        if (Offset == 0)
        {
            ov = iv;
        }
        else
        {
            ov = _mm256_srli_epi32(iv, Offset);
        }

        if (Spans)
        {
            iv = _mm256_loadu_si256(ip++);
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm256_or_si256(ov, _mm256_and_si256(_mm256_slli_epi32(iv, BitsInFirst), mask));
        }
        else
        {
            // If B=32 and Offset=0, mask is -1, so this is just ov = iv
            if (B != 32)
            {
                ov = _mm256_and_si256(ov, mask);
            }
        }

        if (Patching)
        {
            // Bitmap access for 256v (4 x 64-bit words)
            // G goes 0..31. Each G produces 8 values (256 bits).
            // But 'bitmap' is array of uint64_t.
            // Values 0..63 are in bitmap[0].
            // G=0 -> values 0..7. G=7 -> values 56..63.
            // G=8 -> values 64..71 (bitmap[1]).

            constexpr int WordIdx = G / 8;
            constexpr int Shift = (G % 8) * 8;

            // We need 8 bits for the current 8 values.
            uint64_t w = bitmap[WordIdx];
            unsigned m = (w >> Shift) & 0xFF; // 8 bits

            // Unconditional patching
            // Split into low 4 (for low 128 lane) and high 4 (for high 128 lane)
            unsigned m_low = m & 0xF;
            unsigned m_high = m >> 4;

            // Low Lane
            {
                __m128i v_low = _mm256_extracti128_si256(ov, 0);
                __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
                __m128i exc_s = _mm_slli_epi32(exc, B);
                __m128i p_mask = _mm_load_si128(reinterpret_cast<const __m128i *>(_shuffle_avx2[m_low]));
                __m128i p = _mm_shuffle_epi8(exc_s, p_mask);
                v_low = _mm_add_epi32(v_low, p);
                ov = _mm256_inserti128_si256(ov, v_low, 0);

#if defined(__GNUC__) || defined(__clang__)
                pex += __builtin_popcount(m_low);
#else
                unsigned c = 0, tm = m_low;
                while (tm)
                {
                    tm &= tm - 1;
                    c++;
                }
                pex += c;
#endif
            }

            // High Lane
            {
                __m128i v_high = _mm256_extracti128_si256(ov, 1);
                __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pex));
                __m128i exc_s = _mm_slli_epi32(exc, B);
                __m128i p_mask = _mm_load_si128(reinterpret_cast<const __m128i *>(_shuffle_avx2[m_high]));
                __m128i p = _mm_shuffle_epi8(exc_s, p_mask);
                v_high = _mm_add_epi32(v_high, p);
                ov = _mm256_inserti128_si256(ov, v_high, 1);

#if defined(__GNUC__) || defined(__clang__)
                pex += __builtin_popcount(m_high);
#else
                unsigned c = 0, tm = m_high;
                while (tm)
                {
                    tm &= tm - 1;
                    c++;
                }
                pex += c;
#endif
            }
        }

        if (Delta)
        {
            sv = mm256_scani_epi32(ov, sv, cv);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + G * 8), sv);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + G * 8), ov);
        }

        constexpr int NextLoadedIdx = Spans ? TargetIdx + 1 : TargetIdx;
        UnpackStep<B, G + 1, MaxG, NextLoadedIdx, Delta, Patching>::run(ip, iv, out, mask, cv, sv, bitmap, pex);
    }
};

// Base case
template <unsigned B, unsigned MaxG, int CurrentLoadedIdx, bool Delta, bool Patching>
struct UnpackStep<B, MaxG, MaxG, CurrentLoadedIdx, Delta, Patching>
{
    static ALWAYS_INLINE void
    run(const __m256i *&, __m256i &, uint32_t *, const __m256i &, const __m256i &, __m256i &, const uint64_t *, const uint32_t *&)
    {
    }
};

// Entry point wrappers
template <unsigned B, unsigned Count, bool Delta, bool Patching>
ALWAYS_INLINE const unsigned char *
bitunpack_avx2_entry(const unsigned char * in, uint32_t * out, __m256i & sv, const uint64_t * bitmap, const uint32_t * pex)
{
    constexpr unsigned MaxG = Count / 8;
    static_assert(Count % 8 == 0, "Count must be multiple of 8");

    if (B == 0)
    {
        if (Delta)
        {
            const __m256i eight = _mm256_set1_epi32(8);
            const __m256i cv = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);

            if (!Patching)
            {
                sv = _mm256_add_epi32(sv, cv);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), sv);
                for (unsigned i = 1; i < MaxG; ++i)
                {
                    sv = _mm256_add_epi32(sv, eight);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * 8), sv);
                }
                return in;
            }
            // Fallthrough if Patching B=0
        }
        else
        {
            if (!Patching)
            {
                const __m256i zero = _mm256_setzero_si256();
                for (unsigned i = 0; i < MaxG; ++i)
                {
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * 8), zero);
                }
                return in;
            }
        }
    }

    const __m256i * ip = reinterpret_cast<const __m256i *>(in);
    __m256i iv = _mm256_setzero_si256();

    const uint32_t mask_val = MaskGen<B>::value;
    const __m256i mask = _mm256_set1_epi32(static_cast<int>(mask_val));

    const __m256i cv = Delta ? _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8) : _mm256_setzero_si256();

    UnpackStep<B, 0, MaxG, -1, Delta, Patching>::run(ip, iv, out, mask, cv, sv, bitmap, pex);
    return reinterpret_cast<const unsigned char *>(ip);
}

} // namespace turbopfor::simd::detail
