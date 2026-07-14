// Find worst-case per-block overhead of bitmap-only vs vbyte-allowed.
// For each block, compute:
//   best_with_vbyte:    min cost across all (b, strategy) including vbyte
//   best_without_vbyte: min cost across all (b, strategy) excluding vbyte
// Report the blocks with largest absolute and percentage overhead.

#include <algorithm>
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

static unsigned vbyte_bytes(unsigned patch_bits)
{
    if (patch_bits <= 7) return 1;
    if (patch_bits <= 15) return 2;
    if (patch_bits <= 19) return 3;
    if (patch_bits <= 25) return 4;
    return 5;
}

struct CostResult {
    unsigned best_b;
    unsigned best_bx;       // 0=none, 1-32=bitmap, 33=vbyte, 34=constant
    unsigned best_cost;
    unsigned best_b_no_vb;
    unsigned best_bx_no_vb;
    unsigned best_cost_no_vb;
    unsigned exc_count_at_b; // exceptions at chosen vbyte b
};

static CostResult analyze_block(const uint32_t* in, unsigned n)
{
    uint32_t or_acc = 0;
    const uint32_t first = in[0];
    unsigned eq = 0;
    for (unsigned i = 0; i < n; ++i) {
        or_acc |= in[i];
        eq += (in[i] == first);
    }

    CostResult r = {};

    if (or_acc == 0) {
        r.best_cost = r.best_cost_no_vb = 1;
        return r;
    }

    unsigned max_bits = bitWidth32(or_acc);

    if (eq == n) {
        unsigned c = pad8(max_bits) + 1;
        r.best_b = r.best_b_no_vb = max_bits;
        r.best_bx = r.best_bx_no_vb = 34;
        r.best_cost = r.best_cost_no_vb = c;
        return r;
    }

    unsigned cnt[40] = {};
    for (unsigned i = 0; i < n; ++i)
        ++cnt[bitWidth32(in[i])];

    const unsigned bmp8 = pad8(n);

    // Sweep all base widths, track best with and without vbyte
    unsigned best_b_vb = max_bits, best_cost_vb = pad8(n * max_bits) + 1;
    unsigned best_bx_vb = 0;
    unsigned best_b_no = max_bits, best_cost_no = pad8(n * max_bits) + 1;
    unsigned best_bx_no = 0;
    unsigned exc_at_vb_b = 0;

    unsigned exc_count = cnt[max_bits];

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

    for (int i = (int)max_bits - 1; i >= 0; --i) {
        unsigned ui = (unsigned)i;
        unsigned patch_bits = max_bits - ui;

        unsigned vbyte_cost  = pad8(n * ui) + 2 + exc_count + vbyte_acc;
        unsigned bitmap_cost = pad8(n * ui) + 2 + bmp8 + pad8(exc_count * patch_bits);

        // With vbyte allowed
        if (bitmap_cost < best_cost_vb && bitmap_cost <= vbyte_cost) {
            best_cost_vb = bitmap_cost;
            best_b_vb = ui;
            best_bx_vb = patch_bits;  // bitmap
        } else if (vbyte_cost < best_cost_vb) {
            best_cost_vb = vbyte_cost;
            best_b_vb = ui;
            best_bx_vb = 33;  // vbyte
            exc_at_vb_b = exc_count;
        }

        // Without vbyte
        if (bitmap_cost < best_cost_no) {
            best_cost_no = bitmap_cost;
            best_b_no = ui;
            best_bx_no = patch_bits;
        }

        exc_count += cnt[ui];
        vbyte_acc += cnt[ui] + (unsigned)vb[i];
        vbb(cnt[ui], ui);
    }

    r.best_b = best_b_vb;
    r.best_bx = best_bx_vb;
    r.best_cost = best_cost_vb;
    r.best_b_no_vb = best_b_no;
    r.best_bx_no_vb = best_bx_no;
    r.best_cost_no_vb = best_cost_no;
    r.exc_count_at_b = exc_at_vb_b;
    return r;
}

// ============================================================
// Part 1: Constructed worst cases
// ============================================================
static void constructed_worst_cases()
{
    printf("=== Constructed worst cases (n=128) ===\n\n");

    constexpr unsigned n = 128;
    uint32_t block[128];

    struct Case {
        const char* desc;
        unsigned n_zeros;
        uint32_t exc_val;
        unsigned exc_count;
    };

    Case cases[] = {
        {"127 zeros + 1 one",              127, 1, 1},
        {"126 zeros + 2 ones",             126, 1, 2},
        {"120 zeros + 8 ones",             120, 1, 8},
        {"127 zeros + 1 val=255 (8-bit)",  127, 255, 1},
        {"125 zeros + 3 val=255",          125, 255, 3},
        {"127 zeros + 1 val=65535 (16-bit)", 127, 65535, 1},
        {"125 zeros + 3 val=65535",        125, 65535, 3},
        {"127 zeros + 1 val=16M (24-bit)", 127, (1u<<24)-1, 1},
        {"126 zeros + 2 val=16M",          126, (1u<<24)-1, 2},
        {"124 twos + 4 val=1023 (10-bit)", 0, 0, 0}, // special
        {"120 8-bit + 8 val=20-bit",       0, 0, 0}, // special
    };

    printf("  %-40s  %6s  %6s  %6s  %6s  %8s\n",
           "description", "vbyte", "bitmap", "diff", "exc#", "overhead%");
    printf("  %-40s  %6s  %6s  %6s  %6s  %8s\n",
           "", "cost", "cost", "(B-V)", "", "");
    printf("  %-80s\n", "--------------------------------------------------------------------------------");

    // Run the simple cases
    for (int c = 0; c < 9; ++c) {
        memset(block, 0, sizeof(block));
        for (unsigned i = 0; i < cases[c].exc_count; ++i)
            block[n - 1 - i] = cases[c].exc_val;

        auto r = analyze_block(block, n);
        int diff = (int)r.best_cost_no_vb - (int)r.best_cost;
        double pct = r.best_cost > 0 ? 100.0 * diff / r.best_cost : 0;

        printf("  %-40s  %6u  %6u  %+6d  %6u  %+7.1f%%\n",
               cases[c].desc, r.best_cost, r.best_cost_no_vb, diff,
               r.exc_count_at_b, pct);
    }

    // Special case: 124 twos + 4 val=1023
    memset(block, 0, sizeof(block));
    for (unsigned i = 0; i < 124; ++i) block[i] = 2;
    for (unsigned i = 124; i < 128; ++i) block[i] = 1023;
    {
        auto r = analyze_block(block, n);
        int diff = (int)r.best_cost_no_vb - (int)r.best_cost;
        double pct = r.best_cost > 0 ? 100.0 * diff / r.best_cost : 0;
        printf("  %-40s  %6u  %6u  %+6d  %6u  %+7.1f%%\n",
               "124 twos + 4 val=1023 (10-bit)", r.best_cost, r.best_cost_no_vb, diff,
               r.exc_count_at_b, pct);
    }

    // Special case: 120 8-bit + 8 20-bit
    {
        std::mt19937 rng(99);
        for (unsigned i = 0; i < 120; ++i) block[i] = rng() % 256;
        for (unsigned i = 120; i < 128; ++i) block[i] = (1u << 19) + (rng() % (1u << 19));

        auto r = analyze_block(block, n);
        int diff = (int)r.best_cost_no_vb - (int)r.best_cost;
        double pct = r.best_cost > 0 ? 100.0 * diff / r.best_cost : 0;
        printf("  %-40s  %6u  %6u  %+6d  %6u  %+7.1f%%\n",
               "120 8-bit + 8 20-bit", r.best_cost, r.best_cost_no_vb, diff,
               r.exc_count_at_b, pct);
    }

    printf("\n");
}

// ============================================================
// Part 2: Monte Carlo search for worst-case blocks
// ============================================================
static void monte_carlo_worst()
{
    printf("=== Monte Carlo worst-case search (n=128, 10M random blocks) ===\n\n");

    constexpr unsigned n = 128;
    constexpr unsigned TRIALS = 10'000'000;

    std::mt19937 rng(42);
    uint32_t block[128];

    int worst_diff = 0;
    double worst_pct = 0;
    unsigned worst_vb_cost = 0, worst_no_cost = 0;
    unsigned worst_b = 0, worst_max = 0, worst_exc = 0;
    unsigned vbyte_chosen = 0;

    // Distribution of overhead percentages
    unsigned pct_hist[20] = {}; // 0-5%, 5-10%, ..., 95-100%+

    for (unsigned t = 0; t < TRIALS; ++t) {
        // Generate block: random base_bits, exc_bits, exc_count
        unsigned base_bits = (rng() % 28) + 1;   // 1-28
        unsigned spread = (rng() % 24) + 1;       // 1-24 extra bits
        unsigned exc_bits = std::min(base_bits + spread, 32u);
        unsigned exc_count = 1 + (rng() % 20);    // 1-20 exceptions

        uint32_t base_max = (1u << base_bits) - 1u;
        uint32_t exc_min = 1u << base_bits;
        uint32_t exc_max = exc_bits >= 32 ? 0xFFFFFFFFu : (1u << exc_bits) - 1u;
        if (exc_min > exc_max) continue;

        std::uniform_int_distribution<uint32_t> base_dist(0, base_max);
        std::uniform_int_distribution<uint32_t> exc_dist(exc_min, exc_max);

        for (unsigned i = 0; i < n; ++i) {
            if (i < exc_count)
                block[i] = exc_dist(rng);
            else
                block[i] = base_dist(rng);
        }
        // Shuffle
        for (unsigned i = n - 1; i > 0; --i) {
            unsigned j = rng() % (i + 1);
            std::swap(block[i], block[j]);
        }

        auto r = analyze_block(block, n);

        if (r.best_bx != 33) continue; // only care when vbyte was chosen
        ++vbyte_chosen;

        int diff = (int)r.best_cost_no_vb - (int)r.best_cost;
        double pct = r.best_cost > 0 ? 100.0 * diff / r.best_cost : 0;

        unsigned bucket = std::min((unsigned)(pct / 5.0), 19u);
        ++pct_hist[bucket];

        if (diff > worst_diff) {
            worst_diff = diff;
            worst_pct = pct;
            worst_vb_cost = r.best_cost;
            worst_no_cost = r.best_cost_no_vb;
            worst_b = r.best_b;
            worst_exc = r.exc_count_at_b;
            worst_max = 0;
            for (unsigned i = 0; i < n; ++i)
                worst_max = std::max(worst_max, bitWidth32(block[i]));
        }
        if (pct > worst_pct) {
            worst_pct = pct;
        }
    }

    printf("  Blocks where vbyte was chosen: %u / %u (%.1f%%)\n\n",
           vbyte_chosen, TRIALS, 100.0 * vbyte_chosen / TRIALS);

    printf("  Worst absolute overhead: %d bytes (vbyte=%u, bitmap-only=%u, +%.1f%%)\n",
           worst_diff, worst_vb_cost, worst_no_cost, worst_pct);
    printf("    (b=%u, max_bits=%u, exc_count=%u)\n\n", worst_b, worst_max, worst_exc);

    printf("  Overhead distribution (when vbyte chosen):\n");
    printf("  %12s  %8s  %8s\n", "range", "count", "pct");
    for (unsigned i = 0; i < 20; ++i) {
        if (pct_hist[i] == 0) continue;
        printf("  %5u-%4u%%  %8u  %7.2f%%\n",
               i * 5, (i + 1) * 5, pct_hist[i],
               100.0 * pct_hist[i] / vbyte_chosen);
    }
    printf("\n");
}

// ============================================================
// Part 3: Posting list worst case — find the single worst block
// ============================================================
static void posting_list_worst()
{
    printf("=== Posting list simulation — per-block worst case ===\n\n");

    constexpr unsigned n = 128;
    std::mt19937 rng(12345);
    uint32_t block[256];

    for (double density : {0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5}) {
        int worst_diff = 0;
        double worst_pct = 0;
        unsigned worst_vb = 0, worst_no = 0, worst_exc = 0;
        unsigned d_total = 0, d_vbyte = 0;
        // histogram
        unsigned pct_hist[10] = {}; // 0-10%, 10-20%, ..., 90-100%+

        for (unsigned list = 0; list < 1000; ++list) {
            uint32_t doc_id = rng() % 100;
            std::geometric_distribution<unsigned> gap_dist(density);

            std::vector<uint32_t> docs;
            for (unsigned i = 0; i < n * 4; ++i) {
                doc_id += gap_dist(rng) + 1;
                docs.push_back(doc_id);
            }

            for (unsigned blk = 0; blk + n <= docs.size(); blk += n) {
                uint32_t start = (blk == 0) ? 0 : docs[blk - 1];
                for (unsigned i = 0; i < n; ++i)
                    block[i] = docs[blk + i] - start - 1;

                auto r = analyze_block(block, n);
                ++d_total;

                if (r.best_bx != 33) continue;
                ++d_vbyte;

                int diff = (int)r.best_cost_no_vb - (int)r.best_cost;
                double pct = r.best_cost > 0 ? 100.0 * diff / r.best_cost : 0;

                unsigned bucket = std::min((unsigned)(pct / 10.0), 9u);
                ++pct_hist[bucket];

                if (diff > worst_diff) {
                    worst_diff = diff;
                    worst_pct = pct;
                    worst_vb = r.best_cost;
                    worst_no = r.best_cost_no_vb;
                    worst_exc = r.exc_count_at_b;
                }
            }
        }

        printf("  density=%.3f: %u blocks, %u vbyte (%.1f%%)\n",
               density, d_total, d_vbyte, 100.0 * d_vbyte / d_total);
        printf("    worst block: vbyte=%u bytes, bitmap-only=%u bytes, +%d bytes (+%.1f%%)\n",
               worst_vb, worst_no, worst_diff, worst_pct);
        if (d_vbyte > 0) {
            printf("    overhead distribution: ");
            for (unsigned i = 0; i < 10; ++i) {
                if (pct_hist[i] > 0)
                    printf("[%u-%u%%]=%u ", i*10, (i+1)*10, pct_hist[i]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main()
{
    constructed_worst_cases();
    monte_carlo_worst();
    posting_list_worst();
    return 0;
}
