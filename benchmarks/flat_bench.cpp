// Quick benchmark: our scalar flat-format D1 decode vs C reference
// This tests what happens if we encode in flat format and decode with our scalar path
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>

extern "C" {
unsigned char * p4enc64(uint64_t *in, unsigned n, unsigned char *out);
unsigned char * p4d1enc64(uint64_t *in, unsigned n, unsigned char *out, uint64_t start);
unsigned char * p4d1dec64(unsigned char *in, unsigned n, uint64_t *out, uint64_t start);
}

namespace turbopfor::scalar::detail {
unsigned char * bitunpackd1_64Scalar(unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b);
unsigned char * bitpack64Scalar(const uint64_t * in, unsigned n, unsigned char * out, unsigned b);
}

int main() {
    constexpr unsigned N = 128;
    constexpr unsigned ITERS = 200000;
    constexpr unsigned RUNS = 5;

    // Test bw=16 random (no exceptions)
    for (unsigned bw : {16u, 32u}) {
        printf("\n=== bw=%u random (flat format) ===\n", bw);

        // Generate data
        std::mt19937_64 rng(42);
        uint64_t orig[N], deltas[N];
        uint64_t prev = 1000;
        for (unsigned i = 0; i < N; ++i) {
            uint64_t delta = rng() & ((1ull << bw) - 1ull);
            prev += delta + 1;
            orig[i] = prev;
            deltas[i] = delta;
        }

        // Encode with C reference (flat format)
        unsigned char buf_c[N * 16];
        {
            uint64_t tmp[N];
            memcpy(tmp, orig, sizeof(orig));
            p4d1enc64(tmp, N, buf_c, 1000);
        }

        // Encode with our scalar flat format
        unsigned char buf_ours[N * 16];
        turbopfor::scalar::detail::bitpack64Scalar(deltas, N, buf_ours, bw);

        // Decode benchmark: C reference
        double best_c = 0;
        for (unsigned r = 0; r < RUNS; ++r) {
            uint64_t out[N];
            auto t0 = std::chrono::high_resolution_clock::now();
            for (unsigned i = 0; i < ITERS; ++i) {
                p4d1dec64(buf_c, N, out, 1000);
                asm volatile("" : : "r"(out[N-1]) : "memory");
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            double mbs = (double)(N * 8) * ITERS / ns * 1000.0;
            if (mbs > best_c) best_c = mbs;
        }

        // Decode benchmark: our scalar flat-format D1
        double best_ours = 0;
        for (unsigned r = 0; r < RUNS; ++r) {
            uint64_t out[N];
            auto t0 = std::chrono::high_resolution_clock::now();
            for (unsigned i = 0; i < ITERS; ++i) {
                turbopfor::scalar::detail::bitunpackd1_64Scalar(buf_ours, N, out, 1000, bw);
                asm volatile("" : : "r"(out[N-1]) : "memory");
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            double mbs = (double)(N * 8) * ITERS / ns * 1000.0;
            if (mbs > best_ours) best_ours = mbs;
        }

        printf("  C ref (flat decode):   %.1f MB/s\n", best_c);
        printf("  Ours (flat D1 scalar): %.1f MB/s\n", best_ours);
        printf("  Diff: %+.1f%%\n", (best_ours - best_c) / best_c * 100.0);
    }

    return 0;
}
