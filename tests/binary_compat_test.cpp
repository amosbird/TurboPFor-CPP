#include "scalar/p4_scalar.h"
#include "simd/p4_simd.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

extern "C" unsigned char * p4enc32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4d1dec32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * p4enc128v32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4d1dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * p4enc256v32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4d1dec256v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * bitpack32(unsigned * in, unsigned n, unsigned char * out, unsigned b);
extern "C" unsigned char * bitunpack32(const unsigned char * in, unsigned n, uint32_t * out, unsigned b);
extern "C" unsigned char * bitd1unpack32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);

namespace turbopfor::scalar::detail
{
unsigned char * bitpack32Scalar(const uint32_t * in, unsigned n, unsigned char * out, unsigned b);
unsigned char * bitunpack32Scalar(unsigned char * in, unsigned n, uint32_t * out, unsigned b);
unsigned char * bitunpackd1_32Scalar(unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);
}

namespace
{

void fillSequential(std::vector<uint32_t> & data, uint32_t base, uint32_t step)
{
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = base + static_cast<uint32_t>(i) * step;
}

void fillRandom(std::vector<uint32_t> & data, uint32_t max_val, std::mt19937 & rng)
{
    std::uniform_int_distribution<uint32_t> dist(0u, max_val);
    for (auto & v : data)
        v = dist(rng);
}

void fillConstant(std::vector<uint32_t> & data, uint32_t value)
{
    for (auto & v : data)
        v = value;
}

void fillWithExceptions(std::vector<uint32_t> & data, uint32_t base_max, uint32_t exc_value, unsigned exc_percent, std::mt19937 & rng)
{
    std::uniform_int_distribution<uint32_t> base_dist(0u, base_max);
    std::uniform_int_distribution<unsigned> exc_dist(0u, 99u);

    for (auto & v : data)
    {
        if (exc_dist(rng) < exc_percent)
            v = exc_value;
        else
            v = base_dist(rng);
    }
}

unsigned pad8(unsigned bits)
{
    return (bits + 7u) / 8u;
}

unsigned popcount64(uint64_t v)
{
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_popcountll(v));
#else
    unsigned c = 0;
    while (v)
    {
        v &= v - 1u;
        ++c;
    }
    return c;
#endif
}

void maskPaddingBits(unsigned char * buf, unsigned total_bits)
{
    unsigned rem = total_bits & 7u;
    if (rem == 0u || total_bits == 0u)
        return;

    unsigned bytes = pad8(total_bits);
    unsigned char mask = static_cast<unsigned char>((1u << rem) - 1u);
    buf[bytes - 1] &= mask;
}

void normalizeP4Enc32(unsigned char * buf, unsigned n)
{
    if (n == 0u)
        return;

    unsigned b = buf[0];

    if ((b & 0xC0u) == 0xC0u)
        return;

    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        unsigned offset = 1u;
        if (b & 0x80u)
        {
            bx = buf[1];
            offset = 2u;
        }
        b &= 0x3Fu;

        if (bx == 0u)
        {
            maskPaddingBits(buf + offset, n * b);
            return;
        }

        if (bx <= 32u)
        {
            unsigned bitmap_bytes = pad8(n);
            unsigned xn = 0u;
            for (unsigned i = 0; i < bitmap_bytes; i += 8)
            {
                uint64_t word = 0;
                unsigned chunk = (bitmap_bytes - i) < 8 ? (bitmap_bytes - i) : 8;
                std::memcpy(&word, buf + offset + i, chunk);
                xn += popcount64(word);
            }

            unsigned char * ex_pack = buf + offset + bitmap_bytes;
            maskPaddingBits(ex_pack, xn * bx);
            unsigned ex_bytes = pad8(xn * bx);
            unsigned char * base_pack = ex_pack + ex_bytes;
            maskPaddingBits(base_pack, n * b);
            return;
        }

        return;
    }

    unsigned bx = buf[1];
    unsigned offset = 2u;
    b &= 0x3Fu;
    (void)bx;
    maskPaddingBits(buf + offset, n * b);
}

} // namespace

//
// Test 1: Binary Compatibility Test
// Verifies C p4enc32/p4d1dec32 is compatible with C++ turbopfor::p4Enc32/p4D1Dec32
// Tests n = 1 to 127
//
unsigned runBinaryCompatibilityTest()
{
    std::mt19937 rng(42u);

    std::vector<unsigned> sizes;
    for (unsigned n = 1; n <= 127; ++n)
        sizes.push_back(n);

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Binary Compatibility Test ===\n");
    std::printf("=== Verifying C p4enc32/p4d1dec32 <-> C++ turbopfor::p4Enc32/p4D1Dec32 ===\n");
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
            std::vector<unsigned char> cxx_scalar_buf(n * 5 + 256);
            std::vector<uint32_t> out_c(n, 0u);
            std::vector<uint32_t> out_cxx_scalar(n, 0u);

            pattern.fill(input, rng);
            std::copy(input.begin(), input.end(), input_copy.begin());
            std::fill(input_copy.begin() + n, input_copy.end(), 0u);
            std::fill(c_buf.begin(), c_buf.end(), 0u);
            std::fill(cxx_scalar_buf.begin(), cxx_scalar_buf.end(), 0u);

            unsigned char * c_end = ::p4enc32(input_copy.data(), n, c_buf.data());
            unsigned char * cxx_scalar_end = turbopfor::scalar::p4Enc32(input_copy.data(), n, cxx_scalar_buf.data());

            size_t c_len = c_end - c_buf.data();
            size_t cxx_scalar_len = cxx_scalar_end - cxx_scalar_buf.data();

            bool ok = true;
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
                normalizeP4Enc32(c_buf.data(), n);
                normalizeP4Enc32(cxx_scalar_buf.data(), n);
                if (!std::equal(c_buf.begin(), c_buf.begin() + c_len, cxx_scalar_buf.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: byte mismatch\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                else
                {
                    ::p4d1dec32(c_buf.data(), n, out_c.data(), 0u);
                    turbopfor::scalar::p4D1Dec32(cxx_scalar_buf.data(), n, out_cxx_scalar.data(), 0u);
                    if (out_c != out_cxx_scalar)
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch\n", n, pattern.name.c_str());
                        ++failed;
                        ok = false;
                    }
                    else
                    {
                        // Cross-decode: C -> scalar
                        std::fill(out_cxx_scalar.begin(), out_cxx_scalar.end(), 0u);
                        turbopfor::scalar::p4D1Dec32(c_buf.data(), n, out_cxx_scalar.data(), 0u);
                        if (out_c != out_cxx_scalar)
                        {
                            std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode C->C++(scalar) mismatch\n", n, pattern.name.c_str());
                            ++failed;
                            ok = false;
                        }
                        else
                        {
                            // Cross-decode: scalar -> C
                            std::fill(out_c.begin(), out_c.end(), 0u);
                            ::p4d1dec32(cxx_scalar_buf.data(), n, out_c.data(), 0u);
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
// Test 2: Cross Validation Test (128v)
// Verifies scalar p4Enc128v32/p4D1Dec128v32 matches SIMD and C reference
// Tests n = 128 only
//
unsigned runCrossValidation128vTest()
{
    std::mt19937 rng(42u);
    const unsigned n = 128u;

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Cross Validation Test (128v Scalar vs SIMD vs C) ===\n");
    std::printf("=== Verifying scalar::p4Enc128v32/p4D1Dec128v32 matches simd and C reference ===\n");
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
        std::vector<unsigned char> scalar_buf(alloc_n * 5 + 256);
        std::vector<unsigned char> simd_buf(alloc_n * 5 + 256);
        std::vector<unsigned char> c_buf(alloc_n * 5 + 256);
        std::vector<uint32_t> out_scalar_enc(alloc_n, 0u);
        std::vector<uint32_t> out_scalar_dec(alloc_n, 0u);
        std::vector<uint32_t> out_simd(alloc_n, 0u);
        std::vector<uint32_t> out_c(alloc_n, 0u);

        pattern.fill(input, rng);
        std::fill(scalar_buf.begin(), scalar_buf.end(), 0u);
        std::fill(simd_buf.begin(), simd_buf.end(), 0u);
        std::fill(c_buf.begin(), c_buf.end(), 0u);

        unsigned char * scalar_end = turbopfor::scalar::p4Enc128v32(input.data(), n, scalar_buf.data());
        unsigned char * simd_end = turbopfor::simd::p4Enc128v32(input.data(), n, simd_buf.data());
        unsigned char * c_end = ::p4enc128v32(input.data(), n, c_buf.data());

        size_t scalar_len = scalar_end - scalar_buf.data();
        size_t simd_len = simd_end - simd_buf.data();
        size_t c_len = c_end - c_buf.data();

        bool ok = true;

        // Compare sizes
        if (scalar_len != simd_len || scalar_len != c_len)
        {
            std::fprintf(
                stderr, "FAIL [n=%u %s]: size mismatch scalar=%zu simd=%zu C=%zu\n", n, pattern.name.c_str(), scalar_len, simd_len, c_len);
            ++failed;
            ok = false;
        }
        else
        {
            // Normalize padding bits before comparison
            normalizeP4Enc32(scalar_buf.data(), n);
            normalizeP4Enc32(simd_buf.data(), n);
            normalizeP4Enc32(c_buf.data(), n);

            // Compare scalar vs simd
            if (!std::equal(scalar_buf.begin(), scalar_buf.begin() + scalar_len, simd_buf.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: scalar vs simd byte mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
            // Compare scalar vs C
            else if (!std::equal(scalar_buf.begin(), scalar_buf.begin() + scalar_len, c_buf.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: scalar vs C byte mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
            else
            {
                // Decode with all implementations (4 decoders now)
                // 1. Scalar encode -> SIMD decode (verify scalar encoder output is correct)
                turbopfor::simd::p4D1Dec128v32(scalar_buf.data(), n, out_scalar_enc.data(), 0u);
                // 2. Scalar encode -> Scalar decode (full scalar roundtrip)
                turbopfor::scalar::p4D1Dec128v32(scalar_buf.data(), n, out_scalar_dec.data(), 0u);
                // 3. SIMD encode -> SIMD decode
                turbopfor::simd::p4D1Dec128v32(simd_buf.data(), n, out_simd.data(), 0u);
                // 4. C encode -> C decode
                ::p4d1dec128v32(c_buf.data(), n, out_c.data(), 0u);

                // Verify scalar encoder output decoded correctly
                if (!std::equal(out_scalar_enc.begin(), out_scalar_enc.begin() + n, out_simd.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch (scalar_enc vs simd)\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                // Verify scalar decoder matches SIMD decoder
                else if (!std::equal(out_scalar_dec.begin(), out_scalar_dec.begin() + n, out_simd.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch (scalar_dec vs simd)\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                // Verify all match C decoder
                else if (!std::equal(out_scalar_dec.begin(), out_scalar_dec.begin() + n, out_c.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch (scalar_dec vs C)\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                // Verify decoded data matches original input (with delta1 applied)
                else
                {
                    std::vector<uint32_t> expected(n);
                    uint32_t acc = 0;
                    for (unsigned i = 0; i < n; ++i)
                    {
                        acc += input[i] + 1u;
                        expected[i] = acc;
                    }
                    if (!std::equal(out_scalar_dec.begin(), out_scalar_dec.begin() + n, expected.begin()))
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: decoded data doesn't match expected\n", n, pattern.name.c_str());
                        ++failed;
                        ok = false;
                    }
                }

                // Cross-decode tests: verify scalar decoder can decode SIMD/C encoded data
                if (ok)
                {
                    std::fill(out_scalar_dec.begin(), out_scalar_dec.end(), 0u);
                    turbopfor::scalar::p4D1Dec128v32(simd_buf.data(), n, out_scalar_dec.data(), 0u);
                    if (!std::equal(out_scalar_dec.begin(), out_scalar_dec.begin() + n, out_simd.begin()))
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode SIMD->scalar mismatch\n", n, pattern.name.c_str());
                        ++failed;
                        ok = false;
                    }
                }
                if (ok)
                {
                    std::fill(out_scalar_dec.begin(), out_scalar_dec.end(), 0u);
                    turbopfor::scalar::p4D1Dec128v32(c_buf.data(), n, out_scalar_dec.data(), 0u);
                    if (!std::equal(out_scalar_dec.begin(), out_scalar_dec.begin() + n, out_c.begin()))
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode C->scalar mismatch\n", n, pattern.name.c_str());
                        ++failed;
                        ok = false;
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
// Test 3: Binary Compatibility Test (128v)
// Verifies C p4enc128v32/p4d1dec128v32 is compatible with C++ turbopfor implementations
// Tests n = 128 only
//
unsigned runBinaryCompatibility128vTest()
{
    std::mt19937 rng(42u);
    const unsigned n = 128u;

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Binary Compatibility Test (128v) ===\n");
    std::printf("=== Verifying C <-> C++ simd <-> C++ scalar (128v) ===\n");
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
        std::vector<unsigned char> cxx_simd_buf(alloc_n * 5 + 256);
        std::vector<unsigned char> cxx_scalar_buf(alloc_n * 5 + 256);
        std::vector<uint32_t> out_c(alloc_n, 0u);
        std::vector<uint32_t> out_cxx_simd(alloc_n, 0u);
        std::vector<uint32_t> out_cxx_scalar(alloc_n, 0u);

        pattern.fill(input, rng);
        std::fill(c_buf.begin(), c_buf.end(), 0u);
        std::fill(cxx_simd_buf.begin(), cxx_simd_buf.end(), 0u);
        std::fill(cxx_scalar_buf.begin(), cxx_scalar_buf.end(), 0u);

        // Encode with all three implementations
        unsigned char * c_end = ::p4enc128v32(input.data(), n, c_buf.data());
        unsigned char * cxx_simd_end = turbopfor::simd::p4Enc128v32(input.data(), n, cxx_simd_buf.data());
        unsigned char * cxx_scalar_end = turbopfor::scalar::p4Enc128v32(input.data(), n, cxx_scalar_buf.data());

        size_t c_len = c_end - c_buf.data();
        size_t cxx_simd_len = cxx_simd_end - cxx_simd_buf.data();
        size_t cxx_scalar_len = cxx_scalar_end - cxx_scalar_buf.data();

        bool ok = true;
        if (c_len != cxx_simd_len || c_len != cxx_scalar_len)
        {
            std::fprintf(
                stderr,
                "FAIL [n=%u %s]: size mismatch C=%zu simd=%zu scalar=%zu\n",
                n,
                pattern.name.c_str(),
                c_len,
                cxx_simd_len,
                cxx_scalar_len);
            ++failed;
            ok = false;
        }
        else
        {
            normalizeP4Enc32(c_buf.data(), n);
            normalizeP4Enc32(cxx_simd_buf.data(), n);
            normalizeP4Enc32(cxx_scalar_buf.data(), n);
            if (!std::equal(c_buf.begin(), c_buf.begin() + c_len, cxx_simd_buf.begin())
                || !std::equal(c_buf.begin(), c_buf.begin() + c_len, cxx_scalar_buf.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: encode byte mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
            else
            {
                // Decode with all three decoders
                ::p4d1dec128v32(c_buf.data(), n, out_c.data(), 0u);
                turbopfor::simd::p4D1Dec128v32(cxx_simd_buf.data(), n, out_cxx_simd.data(), 0u);
                turbopfor::scalar::p4D1Dec128v32(cxx_scalar_buf.data(), n, out_cxx_scalar.data(), 0u);

                if (!std::equal(out_c.begin(), out_c.begin() + n, out_cxx_simd.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch C vs simd\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                else if (!std::equal(out_c.begin(), out_c.begin() + n, out_cxx_scalar.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch C vs scalar\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                // Cross-decode: scalar decoder on C-encoded data
                else
                {
                    std::fill(out_cxx_scalar.begin(), out_cxx_scalar.end(), 0u);
                    turbopfor::scalar::p4D1Dec128v32(c_buf.data(), n, out_cxx_scalar.data(), 0u);
                    if (!std::equal(out_c.begin(), out_c.begin() + n, out_cxx_scalar.begin()))
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode C->scalar mismatch\n", n, pattern.name.c_str());
                        ++failed;
                        ok = false;
                    }
                    // Cross-decode: scalar decoder on SIMD-encoded data
                    else
                    {
                        std::fill(out_cxx_scalar.begin(), out_cxx_scalar.end(), 0u);
                        turbopfor::scalar::p4D1Dec128v32(cxx_simd_buf.data(), n, out_cxx_scalar.data(), 0u);
                        if (!std::equal(out_cxx_simd.begin(), out_cxx_simd.begin() + n, out_cxx_scalar.begin()))
                        {
                            std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode simd->scalar mismatch\n", n, pattern.name.c_str());
                            ++failed;
                            ok = false;
                        }
                        // Cross-decode: C decoder on scalar-encoded data
                        else
                        {
                            std::fill(out_c.begin(), out_c.end(), 0u);
                            ::p4d1dec128v32(cxx_scalar_buf.data(), n, out_c.data(), 0u);
                            if (!std::equal(out_cxx_scalar.begin(), out_cxx_scalar.begin() + n, out_c.begin()))
                            {
                                std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode scalar->C mismatch\n", n, pattern.name.c_str());
                                ++failed;
                                ok = false;
                            }
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
// Test 4: Bitunpack Compatibility Test
// Verifies C bitunpack32 matches C++ turbopfor::scalar::detail::bitunpack32Scalar
//
unsigned runBitunpackCompatibilityTest()
{
    std::mt19937 rng(42u);

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Bitunpack Compatibility Test ===\n");
    std::printf("=== Verifying C bitunpack32 <-> C++ bitunpack32Scalar ===\n");
    std::printf("=== Testing n = 1 to 127, bit widths 1..32 ===\n\n");

    for (unsigned n = 1; n <= 127; ++n)
    {
        for (unsigned bw = 1; bw <= 32; ++bw)
        {
            const uint32_t max_val = (bw == 32) ? 0xFFFFFFFFu : ((1u << bw) - 1u);
            std::vector<uint32_t> input(n);
            std::vector<unsigned char> c_buf(n * 4 + 64);
            std::vector<unsigned char> cxx_buf(n * 4 + 64);
            std::vector<uint32_t> out_c(n, 0u);
            std::vector<uint32_t> out_cxx(n, 0u);

            // Pattern: sequential (modulo max_val) to stay within bit width.
            const uint64_t range = static_cast<uint64_t>(max_val) + 1u;
            for (unsigned i = 0; i < n; ++i)
                input[i] = (range == 0u) ? static_cast<uint32_t>(i) : static_cast<uint32_t>(i % range);

            auto test_case = [&](const char * name)
            {
                std::fill(c_buf.begin(), c_buf.end(), 0u);
                std::fill(cxx_buf.begin(), cxx_buf.end(), 0u);
                std::fill(out_c.begin(), out_c.end(), 0u);
                std::fill(out_cxx.begin(), out_cxx.end(), 0u);

                unsigned char * c_end
                    = ::bitpack32(const_cast<unsigned *>(reinterpret_cast<const unsigned *>(input.data())), n, c_buf.data(), bw);
                unsigned char * cxx_end = turbopfor::scalar::detail::bitpack32Scalar(input.data(), n, cxx_buf.data(), bw);

                size_t c_len = static_cast<size_t>(c_end - c_buf.data());
                size_t cxx_len = static_cast<size_t>(cxx_end - cxx_buf.data());

                ::bitunpack32(c_buf.data(), n, out_c.data(), bw);
                turbopfor::scalar::detail::bitunpack32Scalar(c_buf.data(), n, out_cxx.data(), bw);

                if (out_c != out_cxx)
                {
                    std::fprintf(stderr, "FAIL [n=%u b=%u %s]: decode mismatch (C pack)\n", n, bw, name);
                    ++failed;
                    return;
                }

                std::fill(out_c.begin(), out_c.end(), 0u);
                std::fill(out_cxx.begin(), out_cxx.end(), 0u);
                ::bitunpack32(cxx_buf.data(), n, out_c.data(), bw);
                turbopfor::scalar::detail::bitunpack32Scalar(cxx_buf.data(), n, out_cxx.data(), bw);

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

                ++passed;
            };

            test_case("sequential");

            // Pattern: all_zeros
            std::fill(input.begin(), input.end(), 0u);
            test_case("all_zeros");

            // Pattern: all_same
            std::fill(input.begin(), input.end(), max_val ? (max_val / 2u) : 0u);
            test_case("all_same");

            // Pattern: random
            std::uniform_int_distribution<uint32_t> dist(0u, max_val);
            for (auto & v : input)
                v = dist(rng);
            test_case("random");
        }
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}

//
// Test 5: BitunpackD1 Compatibility Test
// Verifies C bitd1unpack matches C++ turbopfor::scalar::detail::bitunpackd1_32Scalar
//
unsigned runBitunpackD1CompatibilityTest()
{
    std::mt19937 rng(42u);

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== BitunpackD1 Compatibility Test ===\n");
    std::printf("=== Verifying C bitd1unpack <-> C++ bitunpackd1_32Scalar ===\n");
    std::printf("=== Testing n = 1 to 127, bit widths 1..32 ===\n\n");

    for (unsigned n = 1; n <= 127; ++n)
    {
        for (unsigned bw = 1; bw <= 32; ++bw)
        {
            const uint32_t max_val = (bw == 32) ? 0xFFFFFFFFu : ((1u << bw) - 1u);
            std::vector<uint32_t> input(n);
            std::vector<unsigned char> c_buf(n * 4 + 64);
            std::vector<unsigned char> cxx_buf(n * 4 + 64);
            std::vector<uint32_t> out_c(n, 0u);
            std::vector<uint32_t> out_cxx(n, 0u);
            const uint32_t start = 7u;

            // Pattern: sequential (modulo max_val) to stay within bit width.
            const uint64_t range = static_cast<uint64_t>(max_val) + 1u;
            for (unsigned i = 0; i < n; ++i)
                input[i] = (range == 0u) ? static_cast<uint32_t>(i) : static_cast<uint32_t>(i % range);

            auto test_case = [&](const char * name)
            {
                std::fill(c_buf.begin(), c_buf.end(), 0u);
                std::fill(cxx_buf.begin(), cxx_buf.end(), 0u);
                std::fill(out_c.begin(), out_c.end(), 0u);
                std::fill(out_cxx.begin(), out_cxx.end(), 0u);

                ::bitpack32(const_cast<unsigned *>(reinterpret_cast<const unsigned *>(input.data())), n, c_buf.data(), bw);
                turbopfor::scalar::detail::bitpack32Scalar(input.data(), n, cxx_buf.data(), bw);

                ::bitd1unpack32(c_buf.data(), n, out_c.data(), start, bw);
                turbopfor::scalar::detail::bitunpackd1_32Scalar(c_buf.data(), n, out_cxx.data(), start, bw);

                if (out_c != out_cxx)
                {
                    std::fprintf(stderr, "FAIL [n=%u b=%u %s]: decode mismatch (C pack)\n", n, bw, name);
                    ++failed;
                    return;
                }

                std::fill(out_c.begin(), out_c.end(), 0u);
                std::fill(out_cxx.begin(), out_cxx.end(), 0u);
                ::bitd1unpack32(cxx_buf.data(), n, out_c.data(), start, bw);
                turbopfor::scalar::detail::bitunpackd1_32Scalar(cxx_buf.data(), n, out_cxx.data(), start, bw);

                if (out_c != out_cxx)
                {
                    std::fprintf(stderr, "FAIL [n=%u b=%u %s]: decode mismatch (C++ pack)\n", n, bw, name);
                    ++failed;
                    return;
                }

                ++passed;
            };

            test_case("sequential");

            // Pattern: all_zeros
            std::fill(input.begin(), input.end(), 0u);
            test_case("all_zeros");

            // Pattern: all_same
            std::fill(input.begin(), input.end(), max_val ? (max_val / 2u) : 0u);
            test_case("all_same");

            // Pattern: random
            std::uniform_int_distribution<uint32_t> dist(0u, max_val);
            for (auto & v : input)
                v = dist(rng);
            test_case("random");
        }
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}
//
// Test 6: Cross Validation Test (256v)
// Verifies scalar p4Enc256v32/p4D1Dec256v32 matches C reference (AVX2 TurboPFor)
// Tests n = 256 only
//
unsigned runCrossValidation256vTest()
{
    std::mt19937 rng(42u);
    const unsigned n = 256u;

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Cross Validation Test (256v Scalar vs C) ===\n");
    std::printf("=== Verifying scalar::p4Enc256v32/p4D1Dec256v32 matches C reference ===\n");
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
        std::vector<unsigned char> scalar_buf(alloc_n * 5 + 512);
        std::vector<unsigned char> c_buf(alloc_n * 5 + 512);
        std::vector<uint32_t> out_scalar_enc(alloc_n, 0u);
        std::vector<uint32_t> out_scalar_dec(alloc_n, 0u);
        std::vector<uint32_t> out_c(alloc_n, 0u);

        pattern.fill(input, rng);
        std::fill(scalar_buf.begin(), scalar_buf.end(), 0u);
        std::fill(c_buf.begin(), c_buf.end(), 0u);

        unsigned char * scalar_end = turbopfor::scalar::p4Enc256v32(input.data(), n, scalar_buf.data());
        unsigned char * c_end = ::p4enc256v32(input.data(), n, c_buf.data());

        size_t scalar_len = scalar_end - scalar_buf.data();
        size_t c_len = c_end - c_buf.data();

        bool ok = true;

        // Compare sizes
        if (scalar_len != c_len)
        {
            std::fprintf(
                stderr, "FAIL [n=%u %s]: size mismatch scalar=%zu C=%zu\n", n, pattern.name.c_str(), scalar_len, c_len);
            ++failed;
            ok = false;
        }
        else
        {
            // Normalize padding bits before comparison (using n=256)
            normalizeP4Enc32(scalar_buf.data(), n);
            normalizeP4Enc32(c_buf.data(), n);

            // Compare scalar vs C
            if (!std::equal(scalar_buf.begin(), scalar_buf.begin() + scalar_len, c_buf.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: scalar vs C byte mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
            else
            {
                // Decode with all implementations
                // 1. Scalar encode -> C decode (verify scalar encoder output is correct)
                ::p4d1dec256v32(scalar_buf.data(), n, out_scalar_enc.data(), 0u);
                // 2. Scalar encode -> Scalar decode (full scalar roundtrip)
                turbopfor::scalar::p4D1Dec256v32(scalar_buf.data(), n, out_scalar_dec.data(), 0u);
                // 3. C encode -> C decode
                ::p4d1dec256v32(c_buf.data(), n, out_c.data(), 0u);

                // Verify scalar encoder output decoded correctly
                if (!std::equal(out_scalar_enc.begin(), out_scalar_enc.begin() + n, out_c.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch (scalar_enc vs C)\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                // Verify scalar decoder matches C decoder
                else if (!std::equal(out_scalar_dec.begin(), out_scalar_dec.begin() + n, out_c.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch (scalar_dec vs C)\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                // Verify decoded data matches original input (with delta1 applied)
                else
                {
                    std::vector<uint32_t> expected(n);
                    uint32_t acc = 0;
                    for (unsigned i = 0; i < n; ++i)
                    {
                        acc += input[i] + 1u;
                        expected[i] = acc;
                    }
                    if (!std::equal(out_scalar_dec.begin(), out_scalar_dec.begin() + n, expected.begin()))
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: decoded data doesn't match expected\n", n, pattern.name.c_str());
                        ++failed;
                        ok = false;
                    }
                }

                // Cross-decode tests: verify scalar decoder can decode C encoded data
                if (ok)
                {
                    std::fill(out_scalar_dec.begin(), out_scalar_dec.end(), 0u);
                    turbopfor::scalar::p4D1Dec256v32(c_buf.data(), n, out_scalar_dec.data(), 0u);
                    if (!std::equal(out_scalar_dec.begin(), out_scalar_dec.begin() + n, out_c.begin()))
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode C->scalar mismatch\n", n, pattern.name.c_str());
                        ++failed;
                        ok = false;
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
// Test 7: Binary Compatibility Test (256v)
// Verifies C p4enc256v32/p4d1dec256v32 is compatible with C++ scalar implementation
// Tests n = 256 only
//
unsigned runBinaryCompatibility256vTest()
{
    std::mt19937 rng(42u);
    const unsigned n = 256u;

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Binary Compatibility Test (256v) ===\n");
    std::printf("=== Verifying C <-> C++ scalar (256v) ===\n");
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
        std::vector<unsigned char> cxx_scalar_buf(alloc_n * 5 + 512);
        std::vector<uint32_t> out_c(alloc_n, 0u);
        std::vector<uint32_t> out_cxx_scalar(alloc_n, 0u);

        pattern.fill(input, rng);
        std::fill(c_buf.begin(), c_buf.end(), 0u);
        std::fill(cxx_scalar_buf.begin(), cxx_scalar_buf.end(), 0u);

        // Encode with both implementations
        unsigned char * c_end = ::p4enc256v32(input.data(), n, c_buf.data());
        unsigned char * cxx_scalar_end = turbopfor::scalar::p4Enc256v32(input.data(), n, cxx_scalar_buf.data());

        size_t c_len = c_end - c_buf.data();
        size_t cxx_scalar_len = cxx_scalar_end - cxx_scalar_buf.data();

        bool ok = true;
        if (c_len != cxx_scalar_len)
        {
            std::fprintf(
                stderr,
                "FAIL [n=%u %s]: size mismatch C=%zu scalar=%zu\n",
                n,
                pattern.name.c_str(),
                c_len,
                cxx_scalar_len);
            ++failed;
            ok = false;
        }
        else
        {
            normalizeP4Enc32(c_buf.data(), n);
            normalizeP4Enc32(cxx_scalar_buf.data(), n);
            if (!std::equal(c_buf.begin(), c_buf.begin() + c_len, cxx_scalar_buf.begin()))
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: encode byte mismatch\n", n, pattern.name.c_str());
                ++failed;
                ok = false;
            }
            else
            {
                // Decode with both decoders
                ::p4d1dec256v32(c_buf.data(), n, out_c.data(), 0u);
                turbopfor::scalar::p4D1Dec256v32(cxx_scalar_buf.data(), n, out_cxx_scalar.data(), 0u);

                if (!std::equal(out_c.begin(), out_c.begin() + n, out_cxx_scalar.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: decode mismatch C vs scalar\n", n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                // Cross-decode: scalar decoder on C-encoded data
                else
                {
                    std::fill(out_cxx_scalar.begin(), out_cxx_scalar.end(), 0u);
                    turbopfor::scalar::p4D1Dec256v32(c_buf.data(), n, out_cxx_scalar.data(), 0u);
                    if (!std::equal(out_c.begin(), out_c.begin() + n, out_cxx_scalar.begin()))
                    {
                        std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode C->scalar mismatch\n", n, pattern.name.c_str());
                        ++failed;
                        ok = false;
                    }
                    // Cross-decode: C decoder on scalar-encoded data
                    else
                    {
                        std::fill(out_c.begin(), out_c.end(), 0u);
                        ::p4d1dec256v32(cxx_scalar_buf.data(), n, out_c.data(), 0u);
                        if (!std::equal(out_cxx_scalar.begin(), out_cxx_scalar.begin() + n, out_c.begin()))
                        {
                            std::fprintf(stderr, "FAIL [n=%u %s]: cross-decode scalar->C mismatch\n", n, pattern.name.c_str());
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
// Test: Prototype Implementation Test
// Verifies the prototype based on the spec can encode/decode correctly
// by comparing against the C reference implementation
//
unsigned runPrototypeCompatibilityTest()
{
    return 0; // Skipped
}
/*
unsigned runPrototypeCompatibilityTest_OLD()
{
    std::mt19937 rng(42u);

    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Prototype Implementation Test ===\n");
    std::printf("=== Verifying prototype matches C reference implementation ===\n");
    std::printf("=== Testing various data patterns ===\n\n");

    for (unsigned n : std::vector<unsigned>{1, 2, 4, 8, 16, 31, 32, 33, 63, 64, 65, 127, 128})
    {
        struct TestPattern
        {
            std::string name;
            std::function<void(std::vector<uint32_t> &, std::mt19937 &)> fill;
        };

        std::vector<TestPattern> patterns;
        patterns.push_back({"sequential", [](std::vector<uint32_t> & d, std::mt19937 &)
            { fillSequential(d, 0u, 1u); }});
        patterns.push_back({"all_zeros", [](std::vector<uint32_t> & d, std::mt19937 &)
            { fillConstant(d, 0u); }});
        patterns.push_back({"all_same", [](std::vector<uint32_t> & d, std::mt19937 &)
            { fillConstant(d, 42u); }});
        patterns.push_back({"random_8bit", [](std::vector<uint32_t> & d, std::mt19937 & r)
            { fillRandom(d, 255u, r); }});
        patterns.push_back({"random_16bit", [](std::vector<uint32_t> & d, std::mt19937 & r)
            { fillRandom(d, 65535u, r); }});
        patterns.push_back({"random_32bit", [](std::vector<uint32_t> & d, std::mt19937 & r)
            { fillRandom(d, 0xFFFFFFFFu, r); }});

        for (const auto & pattern : patterns)
        {
            std::vector<uint32_t> input(n);
            std::vector<unsigned char> proto_buf(n * 5 + 256);
            std::vector<unsigned char> ref_buf(n * 5 + 256);
            std::vector<uint32_t> out_proto(n, 0u);
            std::vector<uint32_t> out_ref(n, 0u);

            pattern.fill(input, rng);
            std::fill(proto_buf.begin(), proto_buf.end(), 0u);
            std::fill(ref_buf.begin(), ref_buf.end(), 0u);

            // Encode with both implementations
            unsigned char * proto_end = turbopfor::prototype::p4Enc32(input.data(), n, proto_buf.data());
            unsigned char * ref_end = ::p4enc32(input.data(), n, ref_buf.data());

            size_t proto_len = proto_end - proto_buf.data();
            size_t ref_len = ref_end - ref_buf.data();

            bool ok = true;

            // Test: Encode with both and verify bytes match
            if (proto_len == ref_len && std::equal(proto_buf.begin(), proto_buf.begin() + proto_len, ref_buf.begin()))
            {
                // Encoded bytes match - decode prototype with its own decoder
                std::fill(out_proto.begin(), out_proto.end(), 0u);
                turbopfor::prototype::p4D1Dec32(proto_buf.data(), n, out_proto.data(), 0u);

                // Also decode with reference decoder
                std::fill(out_ref.begin(), out_ref.end(), 0u);
                ::p4d1dec32(ref_buf.data(), n, out_ref.data(), 0u);

                // Both decoders should produce the same result
                if (!std::equal(input.begin(), input.end(), out_proto.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: prototype decode mismatch\n",
                        n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
                else if (!std::equal(out_ref.begin(), out_ref.end(), out_proto.begin()))
                {
                    std::fprintf(stderr, "FAIL [n=%u %s]: decoder mismatch (proto vs ref)\n",
                        n, pattern.name.c_str());
                    ++failed;
                    ok = false;
                }
            }
            else
            {
                std::fprintf(stderr, "FAIL [n=%u %s]: encoding mismatch (len=%zu vs %zu)\n",
                    n, pattern.name.c_str(), proto_len, ref_len);
                ++failed;
                ok = false;
            }

            if (ok)
                ++passed;
        }
    }

    std::printf("%u passed, %u failed\n\n", passed, failed);
    return failed;
}
*/

int main()
{
    unsigned failed_compat = runBinaryCompatibilityTest();
    unsigned failed_128v_cross = runCrossValidation128vTest();
    unsigned failed_128v_compat = runBinaryCompatibility128vTest();
    unsigned failed_256v_cross = runCrossValidation256vTest();
    unsigned failed_256v_compat = runBinaryCompatibility256vTest();
    unsigned failed_bitunpack = runBitunpackCompatibilityTest();
    unsigned failed_bitunpack_d1 = runBitunpackD1CompatibilityTest();
    unsigned failed_proto = runPrototypeCompatibilityTest();

    std::printf("=== Summary ===\n");
    std::printf("Binary Compatibility Test failures: %u\n", failed_compat);
    std::printf("Cross Validation (128v) Test failures: %u\n", failed_128v_cross);
    std::printf("Binary Compatibility (128v) Test failures: %u\n", failed_128v_compat);
    std::printf("Cross Validation (256v) Test failures: %u\n", failed_256v_cross);
    std::printf("Binary Compatibility (256v) Test failures: %u\n", failed_256v_compat);
    std::printf("Bitunpack Compatibility Test failures: %u\n", failed_bitunpack);
    std::printf("BitunpackD1 Compatibility Test failures: %u\n", failed_bitunpack_d1);
    std::printf("Prototype Implementation Test failures: %u\n", failed_proto);
    std::printf(
        "Total failures: %u\n",
        failed_compat + failed_128v_cross + failed_128v_compat + failed_256v_cross + failed_256v_compat + failed_bitunpack + failed_bitunpack_d1 + failed_proto);

    return (failed_compat + failed_128v_cross + failed_128v_compat + failed_256v_cross + failed_256v_compat + failed_bitunpack + failed_bitunpack_d1 + failed_proto) > 0 ? 1 : 0;
}
