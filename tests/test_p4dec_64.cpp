#include "test_helpers.h"

unsigned runP4Dec64CompatibilityTest()
{
    std::mt19937_64 rng(42ull);

    std::vector<unsigned> sizes;
    for (unsigned n = 1; n <= 127; ++n)
        sizes.push_back(n);

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== P4Dec64 Binary Compatibility Test ===\n");
    std::printf("=== Verifying C p4dec64 <-> C++ turbopfor::scalar::p4Dec64 ===\n");
    std::printf("=== Testing n = 1 to 127 ===\n\n");

    for (unsigned n : sizes)
    {
        struct TestPattern
        {
            std::string name;
            std::function<void(std::vector<uint64_t> &, std::mt19937_64 &)> fill;
        };

        std::vector<TestPattern> patterns;

        patterns.push_back({"sequential", [](std::vector<uint64_t> & d, std::mt19937_64 &) { fillSequential64(d, 0ull, 1ull); }});
        patterns.push_back({"all_zeros", [](std::vector<uint64_t> & d, std::mt19937_64 &) { fillConstant64(d, 0ull); }});
        patterns.push_back({"all_same", [](std::vector<uint64_t> & d, std::mt19937_64 &) { fillConstant64(d, 42ull); }});

        for (unsigned bw = 1; bw <= 64; ++bw)
        {
            uint64_t max_val = (bw == 64) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << bw) - 1ull);
            patterns.push_back(
                {"random_bw" + std::to_string(bw), [max_val](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillRandom64(d, max_val, r); }});
        }

        patterns.push_back(
            {"exceptions_5pct", [](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillWithExceptions64(d, 255ull, 100000ull, 5u, r); }});
        patterns.push_back(
            {"exceptions_10pct", [](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillWithExceptions64(d, 255ull, 100000ull, 10u, r); }});
        patterns.push_back(
            {"exceptions_25pct", [](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillWithExceptions64(d, 255ull, 100000ull, 25u, r); }});

        for (const auto & pattern : patterns)
        {
            const unsigned input_extra = 32u;
            std::vector<uint64_t> input_copy(n + input_extra, 0ull);
            std::vector<uint64_t> input(n);
            std::vector<unsigned char> c_buf(n * 10 + 256);
            std::vector<uint64_t> out_c(n, 0ull);
            std::vector<uint64_t> out_cxx(n, 0ull);

            pattern.fill(input, rng);
            std::copy(input.begin(), input.end(), input_copy.begin());
            std::fill(input_copy.begin() + n, input_copy.end(), 0ull);
            std::fill(c_buf.begin(), c_buf.end(), 0u);

            unsigned char * c_end = ::p4enc64(input_copy.data(), n, c_buf.data());
            size_t c_len = c_end - c_buf.data();
            (void)c_len;

            bool ok = true;

            ::p4dec64(c_buf.data(), n, out_c.data());
            turbopfor::scalar::p4Dec64(c_buf.data(), n, out_cxx.data());

            if (out_c != out_cxx)
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec64 decode mismatch C vs C++\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }

            if (ok)
            {
                std::fill(out_cxx.begin(), out_cxx.end(), 0ull);
                std::vector<unsigned char> cxx_buf(n * 10 + 256, 0u);
                unsigned char * cxx_end = turbopfor::scalar::p4Enc64(input_copy.data(), n, cxx_buf.data());
                (void)cxx_end;

                turbopfor::scalar::p4Dec64(cxx_buf.data(), n, out_cxx.data());
                std::fill(out_c.begin(), out_c.end(), 0ull);
                ::p4dec64(cxx_buf.data(), n, out_c.data());

                if (out_c != out_cxx)
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec64 cross-decode mismatch\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
            }

            if (ok)
                ++passed;
        }
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}
