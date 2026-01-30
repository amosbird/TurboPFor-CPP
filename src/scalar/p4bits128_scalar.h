#pragma once

#include "p4_scalar_internal.h"

namespace turbopfor::scalar::detail
{

/// Optimized bit width selection for n=128 (scalar version)
/// Returns base bit width and sets *pbx to exception strategy
///
/// Performance critical: this function is called for every 128-element block
/// Optimizations vs generic p4Bits32:
/// - Early exit for all-zeros (very common)
/// - Early exit for constant blocks
/// - Unrolled loops for 128 elements
inline unsigned p4Bits128(const uint32_t * in, unsigned * pbx)
{
    constexpr unsigned n = 128;

    // Phase 1: Fast scan to find OR of all values and count equal to first
    uint32_t or_acc = 0;
    const uint32_t first = in[0];
    unsigned eq = 0;

    // Unroll 8x for better throughput
    for (unsigned i = 0; i < n; i += 8)
    {
        or_acc |= in[i] | in[i + 1] | in[i + 2] | in[i + 3] | in[i + 4] | in[i + 5] | in[i + 6] | in[i + 7];

        eq += (in[i] == first) + (in[i + 1] == first) + (in[i + 2] == first) + (in[i + 3] == first) + (in[i + 4] == first)
            + (in[i + 5] == first) + (in[i + 6] == first) + (in[i + 7] == first);
    }

    // Compute bit width from OR result
    unsigned b = bitWidth32(or_acc);

    // Check for constant block (all values equal and non-zero)
    if (eq == n && first != 0u)
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

    // Phase 2: Build histogram of bit widths
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

} // namespace turbopfor::scalar::detail
