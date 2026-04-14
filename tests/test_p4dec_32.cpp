#include "test_helpers.h"

unsigned runP4Dec32CompatibilityTest()
{
    std::mt19937 rng(42u);

    std::vector<unsigned> sizes;
    for (unsigned n = 1; n <= 127; ++n)
        sizes.push_back(n);

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== P4Dec32 Binary Compatibility Test ===\n");
    std::printf("=== Verifying C p4dec32 <-> C++ turbopfor::scalar::p4Dec32 ===\n");
    std::printf("=== Testing n = 1 to 127 ===\n\n");

    for (unsigned n : sizes)
    {
        struct TestPattern
        {
            std::string name;
            std::function<void(std::vector<uint32_t> &, std::mt19937 &)> fill;
        };

        std::vector<TestPattern> patterns;

        patterns.push_back({"sequential", [](std::vector<uint32_t> & d, std::mt19937 &) { fillSequential(d, 0u, 1u); }});
        patterns.push_back({"all_zeros", [](std::vector<uint32_t> & d, std::mt19937 &) { fillConstant(d, 0u); }});
        patterns.push_back({"all_same", [](std::vector<uint32_t> & d, std::mt19937 &) { fillConstant(d, 42u); }});

        for (unsigned bw = 1; bw <= 32; ++bw)
        {
            uint32_t max_val = (bw == 32) ? 0xFFFFFFFFu : ((1u << bw) - 1u);
            patterns.push_back(
                {"random_bw" + std::to_string(bw), [max_val](std::vector<uint32_t> & d, std::mt19937 & r) { fillRandom(d, max_val, r); }});
        }

        patterns.push_back(
            {"exceptions_5pct", [](std::vector<uint32_t> & d, std::mt19937 & r) { fillWithExceptions(d, 255u, 100000u, 5u, r); }});
        patterns.push_back(
            {"exceptions_10pct", [](std::vector<uint32_t> & d, std::mt19937 & r) { fillWithExceptions(d, 255u, 100000u, 10u, r); }});
        patterns.push_back(
            {"exceptions_25pct", [](std::vector<uint32_t> & d, std::mt19937 & r) { fillWithExceptions(d, 255u, 100000u, 25u, r); }});

        for (const auto & pattern : patterns)
        {
            const unsigned input_extra = 32u;
            std::vector<uint32_t> input_copy(n + input_extra, 0u);
            std::vector<uint32_t> input(n);
            std::vector<unsigned char> c_buf(n * 5 + 256);
            std::vector<uint32_t> out_c(n, 0u);
            std::vector<uint32_t> out_cxx(n, 0u);

            pattern.fill(input, rng);
            std::copy(input.begin(), input.end(), input_copy.begin());
            std::fill(input_copy.begin() + n, input_copy.end(), 0u);
            std::fill(c_buf.begin(), c_buf.end(), 0u);

            unsigned char * c_end = ::p4enc32(input_copy.data(), n, c_buf.data());
            size_t c_len = c_end - c_buf.data();

            bool ok = true;

            ::p4dec32(c_buf.data(), n, out_c.data());
            turbopfor::scalar::p4Dec32(c_buf.data(), n, out_cxx.data());

            if (out_c != out_cxx)
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec32 decode mismatch C vs C++\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }

            if (ok)
            {
                std::fill(out_cxx.begin(), out_cxx.end(), 0u);
                std::vector<unsigned char> cxx_buf(n * 5 + 256, 0u);
                unsigned char * cxx_end = turbopfor::scalar::p4Enc32(input_copy.data(), n, cxx_buf.data());
                size_t cxx_len = cxx_end - cxx_buf.data();

                turbopfor::scalar::p4Dec32(cxx_buf.data(), n, out_cxx.data());
                std::fill(out_c.begin(), out_c.end(), 0u);
                ::p4dec32(cxx_buf.data(), n, out_c.data());

                if (out_c != out_cxx)
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec32 cross-decode mismatch\n", n, pattern.name.c_str());
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

unsigned runP4Dec128v32CompatibilityTest()
{
    std::mt19937 rng(42u);
    const unsigned n = 128u;

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== P4Dec128v32 Binary Compatibility Test ===\n");
    std::printf("=== Verifying C p4dec128v32 <-> C++ scalar/simd p4Dec128v32 ===\n");
    std::printf("=== Testing n = 128 ===\n\n");

    struct TestPattern
    {
        std::string name;
        std::function<void(std::vector<uint32_t> &, std::mt19937 &)> fill;
    };

    std::vector<TestPattern> patterns;

    patterns.push_back({"sequential", [](std::vector<uint32_t> & d, std::mt19937 &) { fillSequential(d, 0u, 1u); }});
    patterns.push_back({"all_zeros", [](std::vector<uint32_t> & d, std::mt19937 &) { fillConstant(d, 0u); }});
    patterns.push_back({"all_same", [](std::vector<uint32_t> & d, std::mt19937 &) { fillConstant(d, 42u); }});

    for (unsigned bw = 1; bw <= 32; ++bw)
    {
        uint32_t max_val = (bw == 32) ? 0xFFFFFFFFu : ((1u << bw) - 1u);
        patterns.push_back(
            {"random_bw" + std::to_string(bw), [max_val](std::vector<uint32_t> & d, std::mt19937 & r) { fillRandom(d, max_val, r); }});
    }

    patterns.push_back(
        {"exceptions_5pct", [](std::vector<uint32_t> & d, std::mt19937 & r) { fillWithExceptions(d, 255u, 100000u, 5u, r); }});
    patterns.push_back(
        {"exceptions_10pct", [](std::vector<uint32_t> & d, std::mt19937 & r) { fillWithExceptions(d, 255u, 100000u, 10u, r); }});
    patterns.push_back(
        {"exceptions_25pct", [](std::vector<uint32_t> & d, std::mt19937 & r) { fillWithExceptions(d, 255u, 100000u, 25u, r); }});

    for (const auto & pattern : patterns)
    {
        const unsigned alloc_n = 128u;
        std::vector<uint32_t> input(alloc_n, 0u);
        std::vector<unsigned char> c_buf(alloc_n * 5 + 256);
        std::vector<uint32_t> out_c(alloc_n, 0u);
        std::vector<uint32_t> out_scalar(alloc_n, 0u);
        std::vector<uint32_t> out_simd(alloc_n, 0u);

        pattern.fill(input, rng);
        std::fill(c_buf.begin(), c_buf.end(), 0u);

        ::p4enc128v32(input.data(), n, c_buf.data());

        ::p4dec128v32(c_buf.data(), n, out_c.data());
        turbopfor::scalar::p4Dec128v32(c_buf.data(), n, out_scalar.data());
        turbopfor::simd::p4Dec128v32(c_buf.data(), n, out_simd.data());

        bool ok = true;

        if (!std::equal(out_c.begin(), out_c.begin() + n, out_scalar.begin()))
        {
            std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec128v32 C vs scalar mismatch\n", n, pattern.name.c_str());
            ++failed;
            ok = false;
        }
        else if (!std::equal(out_c.begin(), out_c.begin() + n, out_simd.begin()))
        {
            std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec128v32 C vs simd mismatch\n", n, pattern.name.c_str());
            ++failed;
            ok = false;
        }

        if (ok)
        {
            if (!std::equal(out_c.begin(), out_c.begin() + n, input.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec128v32 roundtrip mismatch (decoded != original)\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
        }

        if (ok)
        {
            std::vector<unsigned char> scalar_buf(alloc_n * 5 + 256, 0u);
            std::vector<unsigned char> simd_buf(alloc_n * 5 + 256, 0u);

            turbopfor::scalar::p4Enc128v32(input.data(), n, scalar_buf.data());
            turbopfor::simd::p4Enc128v32(input.data(), n, simd_buf.data());

            std::fill(out_c.begin(), out_c.end(), 0u);
            std::fill(out_scalar.begin(), out_scalar.end(), 0u);
            std::fill(out_simd.begin(), out_simd.end(), 0u);

            ::p4dec128v32(scalar_buf.data(), n, out_c.data());
            turbopfor::scalar::p4Dec128v32(simd_buf.data(), n, out_scalar.data());
            turbopfor::simd::p4Dec128v32(scalar_buf.data(), n, out_simd.data());

            if (!std::equal(out_c.begin(), out_c.begin() + n, input.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec128v32 cross-decode C(scalar_enc) mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
            else if (!std::equal(out_scalar.begin(), out_scalar.begin() + n, input.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec128v32 cross-decode scalar(simd_enc) mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
            else if (!std::equal(out_simd.begin(), out_simd.begin() + n, input.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec128v32 cross-decode simd(scalar_enc) mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
        }

        if (ok)
            ++passed;
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}

unsigned runP4Dec256v32CompatibilityTest()
{
    std::mt19937 rng(42u);
    const unsigned n = 256u;

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== P4Dec256v32 Binary Compatibility Test ===\n");
    std::printf("=== Verifying C p4dec256v32 <-> C++ scalar p4Dec256v32 ===\n");
    std::printf("=== Testing n = 256 ===\n\n");

    struct TestPattern
    {
        std::string name;
        std::function<void(std::vector<uint32_t> &, std::mt19937 &)> fill;
    };

    std::vector<TestPattern> patterns;

    patterns.push_back({"sequential", [](std::vector<uint32_t> & d, std::mt19937 &) { fillSequential(d, 0u, 1u); }});
    patterns.push_back({"all_zeros", [](std::vector<uint32_t> & d, std::mt19937 &) { fillConstant(d, 0u); }});
    patterns.push_back({"all_same", [](std::vector<uint32_t> & d, std::mt19937 &) { fillConstant(d, 42u); }});

    for (unsigned bw = 1; bw <= 32; ++bw)
    {
        uint32_t max_val = (bw == 32) ? 0xFFFFFFFFu : ((1u << bw) - 1u);
        patterns.push_back(
            {"random_bw" + std::to_string(bw), [max_val](std::vector<uint32_t> & d, std::mt19937 & r) { fillRandom(d, max_val, r); }});
    }

    patterns.push_back(
        {"exceptions_5pct", [](std::vector<uint32_t> & d, std::mt19937 & r) { fillWithExceptions(d, 255u, 100000u, 5u, r); }});
    patterns.push_back(
        {"exceptions_10pct", [](std::vector<uint32_t> & d, std::mt19937 & r) { fillWithExceptions(d, 255u, 100000u, 10u, r); }});
    patterns.push_back(
        {"exceptions_25pct", [](std::vector<uint32_t> & d, std::mt19937 & r) { fillWithExceptions(d, 255u, 100000u, 25u, r); }});

    for (const auto & pattern : patterns)
    {
        const unsigned alloc_n = 256u;
        std::vector<uint32_t> input(alloc_n, 0u);
        std::vector<unsigned char> c_buf(alloc_n * 5 + 512);
        std::vector<uint32_t> out_c(alloc_n, 0u);
        std::vector<uint32_t> out_scalar(alloc_n, 0u);

        pattern.fill(input, rng);
        std::fill(c_buf.begin(), c_buf.end(), 0u);

        ::p4enc256v32(input.data(), n, c_buf.data());

        ::p4dec256v32(c_buf.data(), n, out_c.data());
        turbopfor::scalar::p4Dec256v32(c_buf.data(), n, out_scalar.data());

        bool ok = true;

        if (!std::equal(out_c.begin(), out_c.begin() + n, out_scalar.begin()))
        {
            std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec256v32 C vs scalar mismatch\n", n, pattern.name.c_str());
            ++failed;
            ok = false;
        }

        if (ok)
        {
            if (!std::equal(out_c.begin(), out_c.begin() + n, input.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec256v32 roundtrip mismatch (decoded != original)\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
        }

        if (ok)
        {
            std::vector<unsigned char> scalar_buf(alloc_n * 5 + 512, 0u);
            turbopfor::scalar::p4Enc256v32(input.data(), n, scalar_buf.data());

            std::fill(out_c.begin(), out_c.end(), 0u);
            std::fill(out_scalar.begin(), out_scalar.end(), 0u);

            ::p4dec256v32(scalar_buf.data(), n, out_c.data());
            turbopfor::scalar::p4Dec256v32(scalar_buf.data(), n, out_scalar.data());

            if (!std::equal(out_c.begin(), out_c.begin() + n, input.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec256v32 cross-decode C(scalar_enc) mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
            else if (!std::equal(out_scalar.begin(), out_scalar.begin() + n, input.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: p4Dec256v32 cross-decode scalar(scalar_enc) mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
        }

        if (ok)
            ++passed;
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}
