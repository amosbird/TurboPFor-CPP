#include "test_helpers.h"

unsigned runBitpack64CompatibilityTest()
{
    std::mt19937_64 rng(42ull);

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Bitpack64 Compatibility Test ===\n");
    std::printf("=== Verifying C bitpack64/bitunpack64 <-> C++ bitpack64Scalar/bitunpack64Scalar ===\n");
    std::printf("=== Testing n = 1 to 127, bit widths 1..64 ===\n\n");

    for (unsigned n = 1; n <= 127; ++n)
    {
        for (unsigned bw = 1; bw <= 64; ++bw)
        {
            const uint64_t max_val = (bw == 64) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << bw) - 1ull);
            std::vector<uint64_t> input(n);
            std::vector<unsigned char> c_buf(n * 8 + 128);
            std::vector<unsigned char> cxx_buf(n * 8 + 128);
            std::vector<uint64_t> out_c(n, 0ull);
            std::vector<uint64_t> out_cxx(n, 0ull);

            // Pattern: sequential (modulo max_val) to stay within bit width.
            const uint64_t range = (bw == 64) ? 0ull : (max_val + 1ull);
            for (unsigned i = 0; i < n; ++i)
                input[i] = (range == 0ull) ? static_cast<uint64_t>(i) : static_cast<uint64_t>(i % range);

            auto test_case = [&](const char * name)
            {
                std::fill(c_buf.begin(), c_buf.end(), 0u);
                std::fill(cxx_buf.begin(), cxx_buf.end(), 0u);
                std::fill(out_c.begin(), out_c.end(), 0ull);
                std::fill(out_cxx.begin(), out_cxx.end(), 0ull);

                unsigned char * c_end = ::bitpack64(input.data(), n, c_buf.data(), bw);
                unsigned char * cxx_end = turbopfor::scalar::detail::bitpack64Scalar(input.data(), n, cxx_buf.data(), bw);

                size_t c_len = static_cast<size_t>(c_end - c_buf.data());
                size_t cxx_len = static_cast<size_t>(cxx_end - cxx_buf.data());

                // Decode both with C decoder
                ::bitunpack64(c_buf.data(), n, out_c.data(), bw);
                turbopfor::scalar::detail::bitunpack64Scalar(c_buf.data(), n, out_cxx.data(), bw);

                if (out_c != out_cxx)
                {
                    std::fprintf(stderr, "FAIL [n=%u b=%u %s]: decode mismatch (C pack)\n", n, bw, name);
                    ++failed;
                    return;
                }

                // Decode C++ packed data
                std::fill(out_c.begin(), out_c.end(), 0ull);
                std::fill(out_cxx.begin(), out_cxx.end(), 0ull);
                ::bitunpack64(cxx_buf.data(), n, out_c.data(), bw);
                turbopfor::scalar::detail::bitunpack64Scalar(cxx_buf.data(), n, out_cxx.data(), bw);

                if (out_c != out_cxx)
                {
                    std::fprintf(stderr, "FAIL [n=%u b=%u %s]: decode mismatch (C++ pack)\n", n, bw, name);
                    ++failed;
                    return;
                }

                if (c_len != cxx_len)
                {
                    std::fprintf(stderr, "FAIL [n=%u b=%u %s]: pack size mismatch (C=%zu C++=%zu)\n", n, bw, name, c_len, cxx_len);
                    ++failed;
                    return;
                }

                // Verify the packed bytes match (after normalizing padding)
                maskPaddingBits(c_buf.data(), static_cast<unsigned>(static_cast<uint64_t>(n) * bw));
                maskPaddingBits(cxx_buf.data(), static_cast<unsigned>(static_cast<uint64_t>(n) * bw));
                if (!std::equal(c_buf.begin(), c_buf.begin() + c_len, cxx_buf.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u b=%u %s]: packed bytes mismatch\n", n, bw, name);
                    ++failed;
                    return;
                }

                ++passed;
            };

            test_case("sequential");

            // Pattern: all_zeros
            std::fill(input.begin(), input.end(), 0ull);
            test_case("all_zeros");

            // Pattern: all_same
            std::fill(input.begin(), input.end(), max_val ? (max_val / 2u) : 0ull);
            test_case("all_same");

            // Pattern: random
            std::uniform_int_distribution<uint64_t> dist(0ull, max_val);
            for (auto & v : input)
                v = dist(rng);
            test_case("random");
        }
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}

//
// Test 9: Binary Compatibility Test (64-bit)
// Verifies C p4enc64/p4d1dec64 is compatible with C++ turbopfor::p4Enc64/p4D1Dec64
// Tests n = 1 to 127

unsigned runBinaryCompatibility64Test()
{
    std::mt19937_64 rng(42ull);

    std::vector<unsigned> sizes;
    for (unsigned n = 1; n <= 127; ++n)
        sizes.push_back(n);

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Binary Compatibility Test (64-bit) ===\n");
    std::printf("=== Verifying C p4enc64/p4d1dec64 <-> C++ turbopfor::p4Enc64/p4D1Dec64 ===\n");
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

        // Test various bit widths including values that span the 32-bit boundary
        for (unsigned bw : {1u, 2u, 4u, 8u, 16u, 24u, 31u, 32u, 33u, 40u, 48u, 56u, 63u, 64u})
        {
            uint64_t max_val = (bw == 64) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << bw) - 1ull);
            patterns.push_back({"random_bw" + std::to_string(bw), [max_val](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                                    fillRandom64(d, max_val, r);
                                }});
        }

        // Exception patterns: base values fit in 8 bits, exceptions require more bits
        patterns.push_back({"exceptions_5pct_32b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                                fillWithExceptions64(d, 255ull, 100000ull, 5u, r);
                            }});
        patterns.push_back({"exceptions_10pct_32b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                                fillWithExceptions64(d, 255ull, 100000ull, 10u, r);
                            }});
        patterns.push_back({"exceptions_25pct_32b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                                fillWithExceptions64(d, 255ull, 100000ull, 25u, r);
                            }});

        // Exception patterns with 64-bit exception values
        patterns.push_back({"exceptions_5pct_64b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                                fillWithExceptions64(d, 255ull, 0x100000000ull, 5u, r);
                            }});
        patterns.push_back({"exceptions_10pct_64b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                                fillWithExceptions64(d, 255ull, 0x100000000ull, 10u, r);
                            }});

        // Edge case: values requiring exactly 63 bits (tests 63→64 quirk)
        patterns.push_back(
            {"random_bw63_large", [](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillRandom64(d, 0x7FFFFFFFFFFFFFFFull, r); }});

        for (const auto & pattern : patterns)
        {
            const unsigned input_extra = 32u;
            std::vector<uint64_t> input_copy(n + input_extra, 0ull);
            std::vector<uint64_t> input(n);
            std::vector<unsigned char> c_buf(n * 10 + 512);
            std::vector<unsigned char> cxx_scalar_buf(n * 10 + 512);
            std::vector<uint64_t> out_c(n, 0ull);
            std::vector<uint64_t> out_cxx_scalar(n, 0ull);

            pattern.fill(input, rng);
            std::copy(input.begin(), input.end(), input_copy.begin());
            std::fill(input_copy.begin() + n, input_copy.end(), 0ull);
            std::fill(c_buf.begin(), c_buf.end(), 0u);
            std::fill(cxx_scalar_buf.begin(), cxx_scalar_buf.end(), 0u);

            unsigned char * c_end = ::p4enc64(input_copy.data(), n, c_buf.data());
            unsigned char * cxx_scalar_end = turbopfor::scalar::p4Enc64(input_copy.data(), n, cxx_scalar_buf.data());

            size_t c_len = c_end - c_buf.data();
            size_t cxx_scalar_len = cxx_scalar_end - cxx_scalar_buf.data();

            bool ok = true;

            // Special check for all_zeros: should be constant block, 1 byte header
            if (pattern.name == "all_zeros")
            {
                if (c_len != 1u || c_buf[0] != 0u)
                {
                    std::fprintf(
                        stderr,
                        "FAIL [n=%u %s]: C header mismatch (len=%zu byte0=0x%02X)\n",
                        n,
                        pattern.name.c_str(),
                        c_len,
                        static_cast<unsigned>(c_buf[0]));
                    ++failed;
                    ok = false;
                }
                if (cxx_scalar_len != 1u || cxx_scalar_buf[0] != 0u)
                {
                    std::fprintf(
                        stderr,
                        "FAIL [n=%u %s]: C++(scalar) header mismatch (len=%zu byte0=0x%02X)\n",
                        n,
                        pattern.name.c_str(),
                        cxx_scalar_len,
                        static_cast<unsigned>(cxx_scalar_buf[0]));
                    ++failed;
                    ok = false;
                }
            }

            if (c_len != cxx_scalar_len)
            {
                std::fprintf(
                    stderr, "FAIL [n=%u %s]: size mismatch C=%zu C++(scalar)=%zu\n", n, pattern.name.c_str(), c_len, cxx_scalar_len);
                ++failed;
                ok = false;
            }
            else
            {
                normalizeP4Enc64(c_buf.data(), n);
                normalizeP4Enc64(cxx_scalar_buf.data(), n);
                if (!std::equal(c_buf.begin(), c_buf.begin() + c_len, cxx_scalar_buf.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: byte mismatch\n", n, pattern.name.c_str());
                    // Print first differing byte for debugging
                    for (size_t i = 0; i < c_len; ++i)
                    {
                        if (c_buf[i] != cxx_scalar_buf[i])
                        {
                            std::fprintf(
                                stderr,
                                "  first diff at byte %zu: C=0x%02X C++=0x%02X\n",
                                i,
                                static_cast<unsigned>(c_buf[i]),
                                static_cast<unsigned>(cxx_scalar_buf[i]));
                            break;
                        }
                    }
                    ++failed;
                    ok = false;
                }
                else
                {
                    // Decode with both implementations
                    ::p4d1dec64(c_buf.data(), n, out_c.data(), 0ull);
                    turbopfor::scalar::p4D1Dec64(cxx_scalar_buf.data(), n, out_cxx_scalar.data(), 0ull);
                    if (out_c != out_cxx_scalar)
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch\n", n, pattern.name.c_str());
                        ++failed;
                        ok = false;
                    }
                    else
                    {
                        // Cross-decode: C encode -> C++ scalar decode
                        std::fill(out_cxx_scalar.begin(), out_cxx_scalar.end(), 0ull);
                        turbopfor::scalar::p4D1Dec64(c_buf.data(), n, out_cxx_scalar.data(), 0ull);
                        if (out_c != out_cxx_scalar)
                        {
                            std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode C->C++(scalar) mismatch\n", n, pattern.name.c_str());
                            ++failed;
                            ok = false;
                        }
                        else
                        {
                            // Cross-decode: C++ scalar encode -> C decode
                            std::fill(out_c.begin(), out_c.end(), 0ull);
                            ::p4d1dec64(cxx_scalar_buf.data(), n, out_c.data(), 0ull);
                            if (out_cxx_scalar != out_c)
                            {
                                std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode C++(scalar)->C mismatch\n", n, pattern.name.c_str());
                                ++failed;
                                ok = false;
                            }
                        }
                    }
                }
            }

            if (ok)
                ++passed;
        }
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}

//
// Test 10: Binary Compatibility Test (128v64)
// Verifies C p4enc128v64/p4dec128v64 is compatible with C++ p4Enc128v64
// Tests n = 128 only
// Note: TurboPFor does NOT have p4d1dec128v64, so we test non-delta cross-decode
// and also test our delta1 decoder via roundtrip

unsigned runBinaryCompatibility128v64Test()
{
    std::mt19937_64 rng(42ull);
    const unsigned n = 128u;

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Binary Compatibility Test (128v64) ===\n");
    std::printf("=== Verifying C p4enc128v64 <-> C++ p4Enc128v64 ===\n");
    std::printf("=== Testing n = 128 ===\n\n");

    struct TestPattern
    {
        std::string name;
        std::function<void(std::vector<uint64_t> &, std::mt19937_64 &)> fill;
    };

    std::vector<TestPattern> patterns;

    patterns.push_back({"sequential", [](std::vector<uint64_t> & d, std::mt19937_64 &) { fillSequential64(d, 0ull, 1ull); }});
    patterns.push_back({"all_zeros", [](std::vector<uint64_t> & d, std::mt19937_64 &) { fillConstant64(d, 0ull); }});
    patterns.push_back({"all_same", [](std::vector<uint64_t> & d, std::mt19937_64 &) { fillConstant64(d, 42ull); }});

    // Test various bit widths — especially the b<=32 / b>32 boundary for hybrid format
    for (unsigned bw : {1u, 2u, 4u, 8u, 16u, 24u, 31u, 32u, 33u, 40u, 48u, 56u, 63u, 64u})
    {
        uint64_t max_val = (bw == 64) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << bw) - 1ull);
        patterns.push_back(
            {"random_bw" + std::to_string(bw), [max_val](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillRandom64(d, max_val, r); }});
    }

    // Exception patterns with values fitting in 32 bits (tests SIMD path in 128v64)
    patterns.push_back(
        {"exceptions_5pct_32b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillWithExceptions64(d, 255ull, 100000ull, 5u, r); }});
    patterns.push_back({"exceptions_10pct_32b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                            fillWithExceptions64(d, 255ull, 100000ull, 10u, r);
                        }});
    patterns.push_back({"exceptions_25pct_32b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                            fillWithExceptions64(d, 255ull, 100000ull, 25u, r);
                        }});

    // Exception patterns with 64-bit exception values (tests scalar fallback in 128v64)
    patterns.push_back({"exceptions_5pct_64b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                            fillWithExceptions64(d, 255ull, 0x100000000ull, 5u, r);
                        }});
    patterns.push_back({"exceptions_10pct_64b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                            fillWithExceptions64(d, 255ull, 0x100000000ull, 10u, r);
                        }});

    // Edge case: 63-bit values (tests 63→64 quirk)
    patterns.push_back(
        {"random_bw63_large", [](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillRandom64(d, 0x7FFFFFFFFFFFFFFFull, r); }});

    for (const auto & pattern : patterns)
    {
        const unsigned alloc_n = 128u;
        std::vector<uint64_t> input(alloc_n, 0ull);
        std::vector<unsigned char> cxx_scalar_buf(alloc_n * 10 + 512);
        std::vector<unsigned char> c_buf(alloc_n * 10 + 512);
        std::vector<uint64_t> out_cxx_scalar(alloc_n, 0ull);
        std::vector<uint64_t> out_c(alloc_n, 0ull);

        pattern.fill(input, rng);
        std::fill(cxx_scalar_buf.begin(), cxx_scalar_buf.end(), 0u);
        std::fill(c_buf.begin(), c_buf.end(), 0u);

        // Encode with both implementations
        unsigned char * cxx_scalar_end = turbopfor::scalar::p4Enc128v64(input.data(), n, cxx_scalar_buf.data());
        unsigned char * c_end = ::p4enc128v64(input.data(), n, c_buf.data());

        size_t cxx_scalar_len = cxx_scalar_end - cxx_scalar_buf.data();
        size_t c_len = c_end - c_buf.data();

        bool ok = true;

        // Compare encoded sizes
        if (cxx_scalar_len != c_len)
        {
            std::fprintf(stderr, "FAIL [n=%u %s]: size mismatch C=%zu C++(scalar)=%zu\n", n, pattern.name.c_str(), c_len, cxx_scalar_len);
            ++failed;
            ok = false;
        }
        else
        {
            // Normalize padding bits before comparison
            normalizeP4Enc64(c_buf.data(), n);
            normalizeP4Enc64(cxx_scalar_buf.data(), n);

            if (!std::equal(c_buf.begin(), c_buf.begin() + c_len, cxx_scalar_buf.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: encode byte mismatch\n", n, pattern.name.c_str());
                for (size_t i = 0; i < c_len; ++i)
                {
                    if (c_buf[i] != cxx_scalar_buf[i])
                    {
                        std::fprintf(
                            stderr,
                            "  first diff at byte %zu: C=0x%02X C++=0x%02X\n",
                            i,
                            static_cast<unsigned>(c_buf[i]),
                            static_cast<unsigned>(cxx_scalar_buf[i]));
                        break;
                    }
                }
                ++failed;
                ok = false;
            }
            else
            {
                // Cross-decode with C's p4dec128v64 (non-delta, since TurboPFor lacks p4d1dec128v64)
                ::p4dec128v64(c_buf.data(), n, out_c.data());
                ::p4dec128v64(cxx_scalar_buf.data(), n, out_cxx_scalar.data());

                if (!std::equal(out_c.begin(), out_c.begin() + n, out_cxx_scalar.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: C non-delta decode mismatch\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                // NOTE: We do NOT check that C's non-delta decode output matches the original input.
                // For b<=32, the 128v64 format intentionally reorders elements (pair-swap within
                // groups of 4, matching the IP32 SIMD shuffle). TurboPFor's p4dec128v64 does NOT
                // reverse this shuffle — it's designed for use with p4ndec128v64 where the shuffle
                // is part of the SIMD pipeline. Our bitunpack128v64Scalar DOES reverse the shuffle,
                // so our decode restores original order. But the C decode won't match input for b<=32.
                //
                // What matters:
                //   1. C encode == C++ encode (verified above)
                //   2. C decode of both produces same output (verified above)
                //   3. Our delta1 decoder roundtrips correctly (verified below)
                {
                    // Test our delta1 decoder via roundtrip: encode -> decode with delta1 -> verify
                    std::fill(out_cxx_scalar.begin(), out_cxx_scalar.end(), 0ull);
                    turbopfor::scalar::p4D1Dec128v64(cxx_scalar_buf.data(), n, out_cxx_scalar.data(), 0ull);

                    // Compute expected delta1 output
                    std::vector<uint64_t> expected(n);
                    uint64_t acc = 0;
                    for (unsigned i = 0; i < n; ++i)
                    {
                        acc += input[i] + 1ull;
                        expected[i] = acc;
                    }
                    if (!std::equal(out_cxx_scalar.begin(), out_cxx_scalar.begin() + n, expected.begin()))
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: delta1 decode mismatch\n", n, pattern.name.c_str());
                        for (unsigned i = 0; i < n; ++i)
                        {
                            if (out_cxx_scalar[i] != expected[i])
                            {
                                std::fprintf(
                                    stderr,
                                    "  first diff at index %u: got=0x%016llX expected=0x%016llX\n",
                                    i,
                                    static_cast<unsigned long long>(out_cxx_scalar[i]),
                                    static_cast<unsigned long long>(expected[i]));
                                break;
                            }
                        }
                        ++failed;
                        ok = false;
                    }

                    // Also test: C encode -> our delta1 decoder
                    if (ok)
                    {
                        std::fill(out_cxx_scalar.begin(), out_cxx_scalar.end(), 0ull);
                        turbopfor::scalar::p4D1Dec128v64(c_buf.data(), n, out_cxx_scalar.data(), 0ull);
                        if (!std::equal(out_cxx_scalar.begin(), out_cxx_scalar.begin() + n, expected.begin()))
                        {
                            std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode C->C++(scalar d1) mismatch\n", n, pattern.name.c_str());
                            ++failed;
                            ok = false;
                        }
                    }

                    // Test our SIMD p4Dec128v64 (non-delta) matches our scalar p4Dec128v64.
                    // Note: C's p4dec128v64 does NOT reverse the IP32 pair-swap, so it
                    // produces [v2,v3,v0,v1] output. Our scalar and SIMD decoders both
                    // reverse the pair-swap, producing correct [v0,v1,v2,v3] order.
                    if (ok)
                    {
                        std::vector<uint64_t> out_simd(alloc_n, 0ull);
                        std::vector<uint64_t> out_scalar_nondelta(alloc_n, 0ull);
                        turbopfor::simd::p4Dec128v64(c_buf.data(), n, out_simd.data());
                        turbopfor::scalar::p4Dec128v64(c_buf.data(), n, out_scalar_nondelta.data());
                        if (!std::equal(out_simd.begin(), out_simd.begin() + n, out_scalar_nondelta.begin()))
                        {
                            std::fprintf(stderr, "FAIL [n=%u %s]: SIMD p4Dec128v64 vs scalar p4Dec128v64 mismatch\n", n, pattern.name.c_str());
                            for (unsigned i = 0; i < n; ++i)
                            {
                                if (out_simd[i] != out_scalar_nondelta[i])
                                {
                                    std::fprintf(
                                        stderr,
                                        "  first diff at index %u: simd=0x%016llX scalar=0x%016llX\n",
                                        i,
                                        static_cast<unsigned long long>(out_simd[i]),
                                        static_cast<unsigned long long>(out_scalar_nondelta[i]));
                                    break;
                                }
                            }
                            ++failed;
                            ok = false;
                        }
                    }

                    // Test our SIMD p4D1Dec128v64 (delta1) matches our scalar p4D1Dec128v64
                    if (ok)
                    {
                        std::vector<uint64_t> out_simd_d1(alloc_n, 0ull);
                        std::vector<uint64_t> out_scalar_d1(alloc_n, 0ull);
                        turbopfor::simd::p4D1Dec128v64(c_buf.data(), n, out_simd_d1.data(), 0ull);
                        turbopfor::scalar::p4D1Dec128v64(c_buf.data(), n, out_scalar_d1.data(), 0ull);
                        if (!std::equal(out_simd_d1.begin(), out_simd_d1.begin() + n, out_scalar_d1.begin()))
                        {
                            std::fprintf(stderr, "FAIL [n=%u %s]: SIMD p4D1Dec128v64 vs scalar p4D1Dec128v64 mismatch\n", n, pattern.name.c_str());
                            for (unsigned i = 0; i < n; ++i)
                            {
                                if (out_simd_d1[i] != out_scalar_d1[i])
                                {
                                    std::fprintf(
                                        stderr,
                                        "  first diff at index %u: simd=0x%016llX scalar=0x%016llX\n",
                                        i,
                                        static_cast<unsigned long long>(out_simd_d1[i]),
                                        static_cast<unsigned long long>(out_scalar_d1[i]));
                                    break;
                                }
                            }
                            ++failed;
                            ok = false;
                        }
                    }
                }
            }
        }

        if (ok)
            ++passed;
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}

//
// Test 11: Roundtrip Compatibility Test (256v64)
// Verifies C++ p4Enc256v64/p4Dec256v64/p4D1Dec256v64 roundtrip correctness.

unsigned runBinaryCompatibility256v64Test()
{
    std::mt19937_64 rng(1337ull);
    const unsigned n = 256u;

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Roundtrip Compatibility Test (256v64) ===\n");
    std::printf("=== Verifying C++ p4Enc256v64/p4Dec256v64/p4D1Dec256v64 ===\n");
    std::printf("=== Testing n = 256 ===\n\n");

    struct TestPattern
    {
        std::string name;
        std::function<void(std::vector<uint64_t> &, std::mt19937_64 &)> fill;
    };

    std::vector<TestPattern> patterns;
    patterns.push_back({"sequential", [](std::vector<uint64_t> & d, std::mt19937_64 &) { fillSequential64(d, 0ull, 1ull); }});
    patterns.push_back({"all_zeros", [](std::vector<uint64_t> & d, std::mt19937_64 &) { fillConstant64(d, 0ull); }});
    patterns.push_back({"all_same", [](std::vector<uint64_t> & d, std::mt19937_64 &) { fillConstant64(d, 42ull); }});

    for (unsigned bw : {1u, 2u, 4u, 8u, 16u, 24u, 31u, 32u, 33u, 40u, 48u, 56u, 63u, 64u})
    {
        uint64_t max_val = (bw == 64u) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << bw) - 1ull);
        patterns.push_back(
            {"random_bw" + std::to_string(bw), [max_val](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillRandom64(d, max_val, r); }});
    }

    // Exception patterns with values fitting in 32 bits
    patterns.push_back(
        {"exceptions_5pct_32b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillWithExceptions64(d, 255ull, 100000ull, 5u, r); }});
    patterns.push_back({"exceptions_10pct_32b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                            fillWithExceptions64(d, 255ull, 100000ull, 10u, r);
                        }});
    patterns.push_back({"exceptions_25pct_32b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                            fillWithExceptions64(d, 255ull, 100000ull, 25u, r);
                        }});

    // Exception patterns with 64-bit exception values
    patterns.push_back({"exceptions_5pct_64b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                            fillWithExceptions64(d, 255ull, 0x100000000ull, 5u, r);
                        }});
    patterns.push_back({"exceptions_10pct_64b", [](std::vector<uint64_t> & d, std::mt19937_64 & r) {
                            fillWithExceptions64(d, 255ull, 0x100000000ull, 10u, r);
                        }});

    // Edge case: values requiring exactly 63 bits (tests 63→64 quirk)
    patterns.push_back(
        {"random_bw63_large", [](std::vector<uint64_t> & d, std::mt19937_64 & r) { fillRandom64(d, 0x7FFFFFFFFFFFFFFFull, r); }});

    for (const auto & pattern : patterns)
    {
        std::vector<uint64_t> input(n, 0ull);
        std::vector<unsigned char> enc_buf(n * 12 + 512, 0u);
        std::vector<uint64_t> out_non_delta(n, 0ull);
        std::vector<uint64_t> out_non_delta_scalar(n, 0ull);
        std::vector<uint64_t> out_delta(n, 0ull);
        std::vector<uint64_t> out_delta_scalar(n, 0ull);

        pattern.fill(input, rng);
        unsigned char * end = turbopfor::p4Enc256v64(input.data(), n, enc_buf.data());
        (void)end;

        bool ok = true;

        turbopfor::p4Dec256v64(enc_buf.data(), n, out_non_delta.data());
        turbopfor::scalar::p4Dec256v64(enc_buf.data(), n, out_non_delta_scalar.data());
        if (!std::equal(out_non_delta.begin(), out_non_delta.end(), out_non_delta_scalar.begin()))
        {
            std::fprintf(stderr, "FAIL [n=%u %s]: non-delta decode mismatch top-level vs scalar\n", n, pattern.name.c_str());
            ++failed;
            ok = false;
        }

        if (ok)
        {
            turbopfor::p4D1Dec256v64(enc_buf.data(), n, out_delta.data(), 0ull);
            turbopfor::scalar::p4D1Dec256v64(enc_buf.data(), n, out_delta_scalar.data(), 0ull);

            if (!std::equal(out_delta.begin(), out_delta.end(), out_delta_scalar.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: delta1 decode mismatch top-level vs scalar\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }

            uint64_t acc = 0ull;
            for (unsigned i = 0; ok && i < n; ++i)
            {
                acc += input[i] + 1ull;
                if (out_delta[i] != acc)
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: delta1 roundtrip mismatch at %u\n", n, pattern.name.c_str(), i);
                    ++failed;
                    ok = false;
                    break;
                }
            }
        }

        // Also test delta1 decode with non-zero start value (tests carry propagation)
        if (ok)
        {
            const uint64_t start_val = 1000000ull;
            std::vector<uint64_t> out_d1_nonzero(n, 0ull);
            std::vector<uint64_t> out_d1_nonzero_scalar(n, 0ull);

            turbopfor::p4D1Dec256v64(enc_buf.data(), n, out_d1_nonzero.data(), start_val);
            turbopfor::scalar::p4D1Dec256v64(enc_buf.data(), n, out_d1_nonzero_scalar.data(), start_val);

            if (!std::equal(out_d1_nonzero.begin(), out_d1_nonzero.end(), out_d1_nonzero_scalar.begin()))
            {
                std::fprintf(
                    stderr, "FAIL [n=%u %s]: delta1 decode (start=%llu) mismatch top-level vs scalar\n", n, pattern.name.c_str(),
                    static_cast<unsigned long long>(start_val));
                ++failed;
                ok = false;
            }
            else
            {
                uint64_t acc2 = start_val;
                for (unsigned i = 0; ok && i < n; ++i)
                {
                    acc2 += input[i] + 1ull;
                    if (out_d1_nonzero[i] != acc2)
                    {
                        std::fprintf(
                            stderr, "FAIL [n=%u %s]: delta1 roundtrip (start=%llu) mismatch at %u: got=0x%016llX expected=0x%016llX\n",
                            n, pattern.name.c_str(), static_cast<unsigned long long>(start_val), i,
                            static_cast<unsigned long long>(out_d1_nonzero[i]), static_cast<unsigned long long>(acc2));
                        ++failed;
                        ok = false;
                        break;
                    }
                }
            }
        }

        if (ok)
            ++passed;
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}


// 64-bit variable-byte encode/decode compatibility
namespace
{

int testVb64Compat(const uint64_t * values, unsigned n, const char * label)
{
    unsigned char c_buf[4096], cpp_buf[4096];
    uint64_t c_dec[256], cpp_dec[256];

    uint64_t * mut_values = const_cast<uint64_t *>(values);
    unsigned char * c_end = vbenc64(mut_values, n, c_buf);
    unsigned char * cpp_end = turbopfor::scalar::detail::vbEnc64(values, n, cpp_buf);

    int c_len = static_cast<int>(c_end - c_buf);
    int cpp_len = static_cast<int>(cpp_end - cpp_buf);

    if (c_len != cpp_len || std::memcmp(c_buf, cpp_buf, c_len) != 0)
    {
        std::printf("FAIL [%s] n=%u: C encoded %d bytes, C++ encoded %d bytes\n", label, n, c_len, cpp_len);
        return 1;
    }

    std::memset(cpp_dec, 0, sizeof(cpp_dec));
    unsigned char * dec_end = turbopfor::scalar::detail::vbDec64(c_buf, n, cpp_dec);
    int dec_len = static_cast<int>(dec_end - c_buf);

    if (dec_len != c_len)
    {
        std::printf("FAIL [%s] n=%u: C++ decoder consumed %d bytes, expected %d\n", label, n, dec_len, c_len);
        return 1;
    }

    for (unsigned i = 0; i < n; i++)
    {
        if (cpp_dec[i] != values[i])
        {
            std::printf("FAIL [%s] n=%u: value[%u] mismatch\n", label, n, i);
            return 1;
        }
    }

    std::memset(c_dec, 0, sizeof(c_dec));
    unsigned char * c_dec_end = vbdec64(cpp_buf, n, c_dec);
    int c_dec_len = static_cast<int>(c_dec_end - cpp_buf);

    if (c_dec_len != cpp_len)
    {
        std::printf("FAIL [%s] n=%u: C decoder consumed %d bytes from C++ encoded, expected %d\n", label, n, c_dec_len, cpp_len);
        return 1;
    }

    for (unsigned i = 0; i < n; i++)
    {
        if (c_dec[i] != values[i])
        {
            std::printf("FAIL [%s] n=%u: C-decoded value[%u] mismatch\n", label, n, i);
            return 1;
        }
    }

    return 0;
}

} // namespace

unsigned runVbyte64CompatibilityTest()
{
    unsigned failures = 0;

    std::printf("=== Vbyte64 Compatibility Test ===\n");
    std::printf("=== Verifying C vbenc64/vbdec64 <-> C++ vbEnc64/vbDec64 ===\n\n");

    uint64_t arr[128];

    for (int i = 0; i < 128; i++) arr[i] = i;
    failures += testVb64Compat(arr, 128, "1-byte sequential");

    for (int i = 0; i < 128; i++) arr[i] = 151;
    failures += testVb64Compat(arr, 128, "1-byte max");

    for (int i = 0; i < 128; i++) arr[i] = 152 + i;
    failures += testVb64Compat(arr, 128, "2-byte sequential");

    for (int i = 0; i < 128; i++) arr[i] = 16535;
    failures += testVb64Compat(arr, 128, "2-byte max");

    for (int i = 0; i < 128; i++) arr[i] = 16536 + i;
    failures += testVb64Compat(arr, 128, "3-byte sequential");

    for (int i = 0; i < 128; i++) arr[i] = 2113687;
    failures += testVb64Compat(arr, 128, "3-byte max");

    for (int i = 0; i < 128; i++) arr[i] = 2113688 + i;
    failures += testVb64Compat(arr, 128, "raw-3 sequential");

    for (int i = 0; i < 128; i++) arr[i] = 0xFFFFFF + i;
    failures += testVb64Compat(arr, 128, "raw-4 sequential");

    for (int i = 0; i < 128; i++) arr[i] = 0xFFFFFFFFULL + i;
    failures += testVb64Compat(arr, 128, "raw-5 sequential");

    for (int i = 0; i < 128; i++) arr[i] = 0xFFFFFFFFFFULL + i;
    failures += testVb64Compat(arr, 128, "raw-6 sequential");

    for (int i = 0; i < 128; i++) arr[i] = 0xFFFFFFFFFFFFULL + i;
    failures += testVb64Compat(arr, 128, "raw-7 sequential");

    for (int i = 0; i < 128; i++) arr[i] = 0xFFFFFFFFFFFFFFFFULL - i;
    failures += testVb64Compat(arr, 128, "raw-8 large");

    for (int i = 0; i < 128; i++)
    {
        switch (i % 8)
        {
            case 0: arr[i] = 0; break;
            case 1: arr[i] = 100; break;
            case 2: arr[i] = 500; break;
            case 3: arr[i] = 50000; break;
            case 4: arr[i] = 3000000; break;
            case 5: arr[i] = 0x12345678ULL; break;
            case 6: arr[i] = 0x123456789AULL; break;
            case 7: arr[i] = 0x123456789ABCDEFULL; break;
        }
    }
    failures += testVb64Compat(arr, 128, "mixed sizes");

    arr[0] = 42;
    failures += testVb64Compat(arr, 1, "n=1 small");

    arr[0] = 0xDEADBEEFCAFEULL;
    failures += testVb64Compat(arr, 1, "n=1 large");

    std::printf("%u failures\n\n", failures);
    return failures;
}
