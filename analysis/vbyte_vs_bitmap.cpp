// Analysis: when does vbyte exception encoding beat bitmap patching?
//
// Extracts the exact cost formulas from p4Bits32/p4Bits128 and sweeps
// across all realistic (exc_count, patch_bits, vbyte_size_per_exc) combos
// for n=128 (the SIMD block size that matters).
//
// Also runs the real p4Bits cost model on synthetic data to count how
// often vbyte wins and by how many bytes.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr unsigned pad8(unsigned x) { return (x + 7u) / 8u; }

static unsigned bitWidth32(uint32_t x)
{
    return x ? (32u - __builtin_clz(x)) : 0u;
}

// ============================================================
// Part 1: Pure math — sweep (exc_count, patch_bits) space
// ============================================================

// For a given exc_count and patch_bits, compute worst-case and best-case
// vbyte sizes and compare against bitmap cost.
//
// vbyte_size_per_exc depends on the actual bit-width of exception values:
//   remainder bits <= 7  → 1 byte
//   remainder bits <= 15 → 2 bytes
//   remainder bits <= 19 → 3 bytes
//   remainder bits <= 25 → 4 bytes
//   remainder bits > 25  → 5 bytes (raw 32-bit)
//
// patch_bits = max_bits - base_bits = the "remainder" width.
// So for patch_bits p, vbyte encodes (v >> base_bits), which has at most p bits.
// vbyte cost per exception = vbyte_bytes(p) + 1 (position byte is in exc_count)
//
// Wait — looking at the code more carefully:
//   vbyte_cost = pad8(n*b) + 2 + exc_count + vbyte_size_accumulator
//   where vbyte_size_accumulator starts at exc_count for the top bit-width
//   and accumulates based on {7,15,19,25} breakpoints
//
// The simplest model: each exception with remainder value < 156 costs 1 byte,
// < 16540 costs 2 bytes, etc. But the cost model estimates based on bit-width
// of the remainder (not the actual value), using the breakpoints.
//
// For exceptions at exactly patch_bits p:
//   p <= 7  → 1 byte estimated
//   p <= 15 → 2 bytes estimated  (actually 1 + vb[bits-7] accumulation)
//   p <= 19 → 3 bytes estimated
//   p <= 25 → 4 bytes estimated
//   p > 25  → 5 bytes estimated

static unsigned vbyte_bytes_for_patch_bits(unsigned p)
{
    if (p <= 7) return 1;
    if (p <= 15) return 2;
    if (p <= 19) return 3;
    if (p <= 25) return 4;
    return 5;
}

static void sweep_analytical()
{
    constexpr unsigned n = 128;
    constexpr unsigned bitmap_base = pad8(n); // 16 bytes

    printf("=== Part 1: Analytical sweep (n=%u) ===\n", n);
    printf("  bitmap_overhead = %u (bitmap) + pad8(exc_count * patch_bits) (exc_pack)\n", bitmap_base);
    printf("  vbyte_overhead  = exc_count (positions) + exc_count * vbyte_bytes(patch_bits) (values)\n");
    printf("\n");
    printf("  %5s  %5s  %8s  %8s  %8s  %s\n",
           "exc", "patch", "bitmap", "vbyte", "savings", "winner");
    printf("  %5s  %5s  %8s  %8s  %8s  %s\n",
           "count", "bits", "bytes", "bytes", "(B-V)", "");
    printf("  %-60s\n", "------------------------------------------------------------");

    unsigned vbyte_win_count = 0;
    unsigned total_count = 0;
    int max_vbyte_savings = 0;

    for (unsigned patch_bits = 1; patch_bits <= 31; ++patch_bits)
    {
        for (unsigned exc_count = 1; exc_count <= 64; ++exc_count)
        {
            // bitmap overhead (beyond shared base)
            unsigned bitmap_oh = bitmap_base + pad8(exc_count * patch_bits);
            // vbyte overhead
            unsigned vb_per = vbyte_bytes_for_patch_bits(patch_bits);
            unsigned vbyte_oh = exc_count + exc_count * vb_per; // positions + values

            int savings = (int)bitmap_oh - (int)vbyte_oh;
            ++total_count;

            if (savings > 0)
            {
                ++vbyte_win_count;
                if (savings > max_vbyte_savings)
                    max_vbyte_savings = savings;

                // Only print vbyte wins
                printf("  %5u  %5u  %8u  %8u  %+8d  vbyte\n",
                       exc_count, patch_bits, bitmap_oh, vbyte_oh, savings);
            }
        }
    }

    printf("\n  Summary: vbyte wins in %u/%u cases (%.1f%%), max savings = %d bytes\n\n",
           vbyte_win_count, total_count, 100.0 * vbyte_win_count / total_count, max_vbyte_savings);
}

// ============================================================
// Part 2: Run the real cost model on synthetic data
// ============================================================

struct P4BitsResult
{
    unsigned b;    // base bit width
    unsigned bx;   // exception strategy: 0=none, 1-32=bitmap, 33=vbyte, 34=constant
    unsigned cost; // total encoded size
};

static P4BitsResult p4Bits_analyze(const uint32_t* in, unsigned n)
{
    uint32_t or_acc = 0;
    const uint32_t first = in[0];
    unsigned eq = 0;

    for (unsigned i = 0; i < n; ++i)
    {
        or_acc |= in[i];
        eq += (in[i] == first);
    }

    if (or_acc == 0)
        return {0, 0, 1};

    unsigned max_bits = bitWidth32(or_acc);

    if (eq == n)
        return {max_bits, 34, pad8(max_bits) + 1};

    unsigned cnt[40] = {};
    for (unsigned i = 0; i < n; ++i)
        ++cnt[bitWidth32(in[i])];

    // Cost model sweep (matching p4Bits32 exactly)
    unsigned optimal_b = max_bits;
    unsigned min_size = pad8(n * max_bits) + 1;
    unsigned use_vbyte = 0;
    unsigned exc_count = cnt[max_bits];
    const unsigned bmp8 = pad8(n);

    // vbyte accumulators
    int vb_storage[128] = {};
    int* vb = vb_storage + 48;

    auto vbb = [&](unsigned count, unsigned bits) {
        vb[(int)bits - 7] += (int)count;
        vb[(int)bits - 15] += (int)(count * 2);
        vb[(int)bits - 19] += (int)(count * 3);
        vb[(int)bits - 25] += (int)(count * 4);
    };

    unsigned vbyte_acc = exc_count;
    vbb(exc_count, max_bits);

    for (int i = (int)max_bits - 1; i >= 0; --i)
    {
        unsigned ui = (unsigned)i;
        unsigned patch_bits = max_bits - ui;

        unsigned vbyte_cost = pad8(n * ui) + 2 + exc_count + vbyte_acc;
        unsigned bitmap_cost = pad8(n * ui) + 2 + bmp8 + pad8(exc_count * patch_bits);

        if (bitmap_cost < min_size && bitmap_cost <= vbyte_cost)
        {
            min_size = bitmap_cost;
            optimal_b = ui;
            use_vbyte = 0;
        }
        else if (vbyte_cost < min_size)
        {
            min_size = vbyte_cost;
            optimal_b = ui;
            use_vbyte = 1;
        }

        exc_count += cnt[ui];
        vbyte_acc += cnt[ui] + (unsigned)vb[i];
        vbb(cnt[ui], ui);
    }

    unsigned bx;
    if (use_vbyte)
        bx = 33;
    else if (max_bits == optimal_b)
        bx = 0;
    else
        bx = max_bits - optimal_b;

    // Also compute what bitmap cost would have been at the chosen b
    unsigned forced_bitmap_cost;
    if (bx == 0 || bx == 34)
    {
        forced_bitmap_cost = min_size; // no difference
    }
    else
    {
        // Recount exceptions at optimal_b
        unsigned exc_at_b = 0;
        for (unsigned j = optimal_b + 1; j <= max_bits; ++j)
            exc_at_b += cnt[j];
        unsigned pb = max_bits - optimal_b;
        forced_bitmap_cost = pad8(n * optimal_b) + 2 + bmp8 + pad8(exc_at_b * pb);
    }

    return {optimal_b, bx, min_size};
}

// Generate a block with known characteristics:
//   base_bits: most values fit in this many bits
//   exc_bits: exceptions need this many bits
//   exc_pct: percentage of exceptions (0-100)
static void gen_block(uint32_t* out, unsigned n, unsigned base_bits, unsigned exc_bits,
                      unsigned exc_pct, std::mt19937& rng)
{
    assert(base_bits <= 32 && exc_bits <= 32 && exc_bits >= base_bits);
    uint32_t base_max = base_bits == 0 ? 0 : ((1u << base_bits) - 1u);
    uint32_t exc_max = exc_bits >= 32 ? 0xFFFFFFFFu : ((1u << exc_bits) - 1u);
    uint32_t exc_min = base_bits >= 32 ? 0xFFFFFFFFu : (1u << base_bits);

    std::uniform_int_distribution<uint32_t> base_dist(0, base_max);
    std::uniform_int_distribution<uint32_t> exc_dist(exc_min, exc_max);
    std::uniform_int_distribution<unsigned> pct_dist(0, 99);

    for (unsigned i = 0; i < n; ++i)
    {
        if (pct_dist(rng) < exc_pct)
            out[i] = exc_dist(rng);
        else
            out[i] = base_dist(rng);
    }
}

static void run_synthetic_sweep()
{
    constexpr unsigned n = 128;
    constexpr unsigned TRIALS = 100;

    printf("=== Part 2: Synthetic data sweep (n=%u, %u trials per config) ===\n\n", n, TRIALS);
    printf("  %5s  %5s  %5s  %8s  %8s  %12s  %12s  %12s\n",
           "base", "exc", "exc%", "vbyte", "bitmap", "vbyte_cost", "bitmap_cost", "savings");
    printf("  %5s  %5s  %5s  %8s  %8s  %12s  %12s  %12s\n",
           "bits", "bits", "", "wins", "wins", "(avg)", "(avg)", "(vbyte avg)");
    printf("  %-90s\n", "------------------------------------------------------------------------------------------");

    std::mt19937 rng(42);
    uint32_t block[128];

    unsigned total_blocks = 0;
    unsigned total_vbyte_wins = 0;
    long long total_vbyte_savings = 0;

    for (unsigned base_bits : {2, 4, 6, 8, 10, 12, 16, 20, 24})
    {
        for (unsigned exc_bits : {8, 12, 16, 20, 24, 28, 32})
        {
            if (exc_bits <= base_bits) continue;

            for (unsigned exc_pct : {1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50})
            {
                unsigned vbyte_wins = 0;
                unsigned bitmap_wins = 0;
                long long vbyte_cost_sum = 0;
                long long bitmap_cost_sum = 0;
                long long savings_when_vbyte = 0;

                for (unsigned t = 0; t < TRIALS; ++t)
                {
                    gen_block(block, n, base_bits, exc_bits, exc_pct, rng);
                    auto result = p4Bits_analyze(block, n);

                    // Also compute what bitmap would cost at same b
                    unsigned cnt[40] = {};
                    for (unsigned i = 0; i < n; ++i)
                        ++cnt[bitWidth32(block[i])];

                    unsigned max_b = bitWidth32(*std::max_element(block, block + n));
                    unsigned exc_at_b = 0;
                    for (unsigned j = result.b + 1; j <= max_b; ++j)
                        exc_at_b += cnt[j];

                    unsigned pb = max_b > result.b ? max_b - result.b : 0;
                    unsigned bm_cost = (result.bx == 0 || result.bx == 34)
                                          ? result.cost
                                          : pad8(n * result.b) + 2 + pad8(n) + pad8(exc_at_b * pb);

                    if (result.bx == 33) // vbyte chosen
                    {
                        ++vbyte_wins;
                        savings_when_vbyte += (long long)bm_cost - (long long)result.cost;
                    }
                    else if (result.bx > 0 && result.bx <= 32)
                    {
                        ++bitmap_wins;
                    }

                    vbyte_cost_sum += result.cost;
                    bitmap_cost_sum += bm_cost;
                    ++total_blocks;
                }

                if (vbyte_wins > 0)
                {
                    total_vbyte_wins += vbyte_wins;
                    total_vbyte_savings += savings_when_vbyte;

                    printf("  %5u  %5u  %4u%%  %8u  %8u  %12.1f  %12.1f  %+12.1f\n",
                           base_bits, exc_bits, exc_pct,
                           vbyte_wins, bitmap_wins,
                           (double)vbyte_cost_sum / TRIALS,
                           (double)bitmap_cost_sum / TRIALS,
                           (double)savings_when_vbyte / vbyte_wins);
                }
            }
        }
    }

    printf("\n  Total: %u/%u blocks chose vbyte (%.2f%%)\n",
           total_vbyte_wins, total_blocks, 100.0 * total_vbyte_wins / total_blocks);
    printf("  Average savings when vbyte wins: %.1f bytes per block\n",
           total_vbyte_wins > 0 ? (double)total_vbyte_savings / total_vbyte_wins : 0.0);
    printf("  Average savings per ALL blocks: %.2f bytes per block\n",
           total_blocks > 0 ? (double)total_vbyte_savings / total_blocks : 0.0);
}

// ============================================================
// Part 3: Realistic inverted-index-like data (sorted doc IDs)
// ============================================================

static void run_realistic_posting_lists()
{
    constexpr unsigned n = 128;

    printf("\n=== Part 3: Realistic posting list simulation ===\n\n");

    std::mt19937 rng(12345);
    uint32_t block[256];

    // Simulate sorted doc IDs in an inverted index.
    // After delta-1 encoding, gaps follow roughly geometric/zipf distribution:
    // most gaps small, occasional large gaps.
    unsigned total_blocks = 0;
    unsigned vbyte_blocks = 0;
    long long total_savings = 0;
    long long total_bitmap_cost = 0;
    long long total_actual_cost = 0;

    // Different "selectivity" levels (how dense the posting list is)
    for (double density : {0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5})
    {
        // Reset per-density stats
        unsigned d_total = 0, d_vbyte = 0;
        long long d_savings = 0;
        long long d_bitmap_sum = 0, d_actual_sum = 0;

        unsigned NUM_LISTS = 1000;

        for (unsigned list = 0; list < NUM_LISTS; ++list)
        {
            // Generate a posting list as sorted doc IDs
            uint32_t doc_id = rng() % 100;
            std::geometric_distribution<unsigned> gap_dist(density);

            // Generate enough doc IDs to fill blocks
            std::vector<uint32_t> docs;
            for (unsigned i = 0; i < n * 4; ++i)
            {
                doc_id += gap_dist(rng) + 1;
                docs.push_back(doc_id);
            }

            // Delta-1 encode and process in blocks of 128
            for (unsigned blk = 0; blk + n <= docs.size(); blk += n)
            {
                uint32_t start = (blk == 0) ? 0 : docs[blk - 1];
                for (unsigned i = 0; i < n; ++i)
                    block[i] = docs[blk + i] - start - 1;
                start = docs[blk + n - 1];

                auto result = p4Bits_analyze(block, n);

                // Compute bitmap cost at same b
                unsigned cnt[40] = {};
                for (unsigned i = 0; i < n; ++i)
                    ++cnt[bitWidth32(block[i])];

                unsigned max_b = 0;
                for (unsigned i = 0; i < n; ++i)
                {
                    unsigned bw = bitWidth32(block[i]);
                    if (bw > max_b) max_b = bw;
                }

                unsigned exc_at_b = 0;
                for (unsigned j = result.b + 1; j <= max_b; ++j)
                    exc_at_b += cnt[j];

                unsigned pb = max_b > result.b ? max_b - result.b : 0;
                unsigned bm_cost = (result.bx == 0 || result.bx == 34)
                                      ? result.cost
                                      : pad8(n * result.b) + 2 + pad8(n) + pad8(exc_at_b * pb);

                ++d_total;

                if (result.bx == 33)
                {
                    ++d_vbyte;
                    d_savings += (long long)bm_cost - (long long)result.cost;
                }

                d_bitmap_sum += bm_cost;
                d_actual_sum += result.cost;
            }
        }

        total_blocks += d_total;
        vbyte_blocks += d_vbyte;
        total_savings += d_savings;
        total_bitmap_cost += d_bitmap_sum;
        total_actual_cost += d_actual_sum;

        printf("  density=%.3f: %u blocks, %u vbyte (%.1f%%), "
               "avg savings when vbyte=%.1f B, "
               "bitmap_total=%lld, actual_total=%lld, overhead=%.2f%%\n",
               density, d_total, d_vbyte,
               d_total > 0 ? 100.0 * d_vbyte / d_total : 0.0,
               d_vbyte > 0 ? (double)d_savings / d_vbyte : 0.0,
               d_bitmap_sum, d_actual_sum,
               d_actual_sum > 0 ? 100.0 * (d_bitmap_sum - d_actual_sum) / (double)d_actual_sum : 0.0);
    }

    printf("\n  TOTAL: %u blocks, %u chose vbyte (%.2f%%)\n",
           total_blocks, vbyte_blocks, 100.0 * vbyte_blocks / total_blocks);
    printf("  Total with bitmap-only: %lld bytes\n", total_bitmap_cost);
    printf("  Total with vbyte allowed: %lld bytes\n", total_actual_cost);
    printf("  Overhead of bitmap-only: %.3f%%\n",
           100.0 * (total_bitmap_cost - total_actual_cost) / (double)total_actual_cost);
}

int main()
{
    sweep_analytical();
    run_synthetic_sweep();
    run_realistic_posting_lists();
    return 0;
}
