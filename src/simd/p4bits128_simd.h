#pragma once

#include "scalar/p4_scalar_internal.h"

#include <smmintrin.h> // SSE4.1

namespace turbopfor::simd::detail
{

// Import constants from scalar namespace
using scalar::detail::MAX_BITS;
using scalar::detail::bitWidth32;
using scalar::detail::pad8;

/// Optimized bit width selection for n=128 using SSE4.1
/// Returns base bit width and sets *pbx to exception strategy
///
/// Performance critical: this function is called for every 128-element block
inline unsigned p4Bits128(const uint32_t * in, unsigned * pbx)
{
    constexpr unsigned n = 128;

    // Phase 1: SIMD reduction to find OR of all values and count equal to first
    const __m128i first_vec = _mm_set1_epi32(static_cast<int>(in[0]));
    __m128i or_acc = _mm_setzero_si128();
    __m128i eq_count = _mm_setzero_si128();
    const __m128i ones = _mm_set1_epi32(1);

    // Unroll 4x for better throughput (16 values per iteration)
    for (unsigned i = 0; i < n; i += 16)
    {
        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i + 4));
        __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i + 8));
        __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i + 12));

        // OR accumulation
        or_acc = _mm_or_si128(or_acc, v0);
        or_acc = _mm_or_si128(or_acc, v1);
        or_acc = _mm_or_si128(or_acc, v2);
        or_acc = _mm_or_si128(or_acc, v3);

        // Count equals
        __m128i eq0 = _mm_and_si128(_mm_cmpeq_epi32(v0, first_vec), ones);
        __m128i eq1 = _mm_and_si128(_mm_cmpeq_epi32(v1, first_vec), ones);
        __m128i eq2 = _mm_and_si128(_mm_cmpeq_epi32(v2, first_vec), ones);
        __m128i eq3 = _mm_and_si128(_mm_cmpeq_epi32(v3, first_vec), ones);

        eq_count = _mm_add_epi32(eq_count, eq0);
        eq_count = _mm_add_epi32(eq_count, eq1);
        eq_count = _mm_add_epi32(eq_count, eq2);
        eq_count = _mm_add_epi32(eq_count, eq3);
    }

    // Horizontal reduction for OR
    __m128i or_tmp = _mm_or_si128(or_acc, _mm_srli_si128(or_acc, 8));
    or_tmp = _mm_or_si128(or_tmp, _mm_srli_si128(or_tmp, 4));
    uint32_t u = static_cast<uint32_t>(_mm_cvtsi128_si32(or_tmp));

    // Horizontal reduction for eq_count
    __m128i eq_tmp = _mm_add_epi32(eq_count, _mm_srli_si128(eq_count, 8));
    eq_tmp = _mm_add_epi32(eq_tmp, _mm_srli_si128(eq_tmp, 4));
    unsigned eq = static_cast<unsigned>(_mm_cvtsi128_si32(eq_tmp));

    // Compute bit width from OR result using lzcnt (branchless)
    unsigned b = bitWidth32(u);

    // Check for constant block (all values equal and non-zero)
    if (eq == n && in[0] != 0u)
    {
        *pbx = MAX_BITS + 2u;
        return b;
    }

    // Early exit for all-zeros
    if (b == 0)
    {
        *pbx = 0;
        return 0;
    }

    // Phase 2: Build histogram of bit widths using lzcnt
    alignas(64) unsigned cnt[MAX_BITS + 8] = {0};

    // Process 8 elements at a time - lzcnt is branchless
    for (unsigned i = 0; i < n; i += 8)
    {
        ++cnt[bitWidth32(in[i])];
        ++cnt[bitWidth32(in[i + 1])];
        ++cnt[bitWidth32(in[i + 2])];
        ++cnt[bitWidth32(in[i + 3])];
        ++cnt[bitWidth32(in[i + 4])];
        ++cnt[bitWidth32(in[i + 5])];
        ++cnt[bitWidth32(in[i + 6])];
        ++cnt[bitWidth32(in[i + 7])];
    }

    // Phase 3: Find optimal bit width using cost model
    unsigned bx = b;
    unsigned x = cnt[bx];
    int ml = static_cast<int>(pad8(n * bx) + 1);

    const unsigned bmp8 = pad8(n); // 16 bytes for 128 elements

    // Quick check: if no exceptions at max bit width, simple bitpacking wins
    if (x == 0)
    {
        *pbx = 0;
        return b;
    }

    // vb array for variable-byte cost estimation
    int vb_storage[MAX_BITS * 2 + 64 + 16] = {0};
    int * vb = vb_storage + MAX_BITS + 16;

    auto vbb = [&vb](unsigned xval, int bval) {
        vb[bval - 7] += static_cast<int>(xval);
        vb[bval - 15] += static_cast<int>(xval * 2u);
        vb[bval - 19] += static_cast<int>(xval * 3u);
        vb[bval - 25] += static_cast<int>(xval * 4u);
    };

    int vv = static_cast<int>(x);
    vbb(x, static_cast<int>(bx));

    int fx = 0;

    for (int i = static_cast<int>(b) - 1; i >= 0; --i)
    {
        int fi;
        const unsigned ui = static_cast<unsigned>(i);
        const int v = static_cast<int>(pad8(n * ui) + 2 + x + static_cast<unsigned>(vv));
        const int l = static_cast<int>(pad8(n * ui) + 2 + bmp8 + pad8(x * (bx - ui)));

        x += cnt[ui];
        vv += static_cast<int>(cnt[ui]) + vb[i];
        vbb(cnt[ui], i);

        fi = l < ml;
        ml = fi ? l : ml;
        if (fi)
        {
            b = ui;
            fx = 0;
        }

        fi = v < ml;
        ml = fi ? v : ml;
        if (fi)
        {
            b = ui;
            fx = 1;
        }
    }

    *pbx = fx ? (MAX_BITS + 1u) : (bx - b);
    return b;
}

} // namespace turbopfor::simd::detail
