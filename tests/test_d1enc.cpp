#include "test_helpers.h"

namespace
{

void makeSorted32(std::vector<uint32_t> & data, std::mt19937 & rng, unsigned maxDelta, uint32_t start)
{
    std::uniform_int_distribution<uint32_t> dist(1u, maxDelta);
    uint32_t acc = start;
    for (auto & v : data)
    {
        acc += dist(rng);
        v = acc;
    }
}

void makeSorted64(std::vector<uint64_t> & data, std::mt19937_64 & rng, uint64_t maxDelta, uint64_t start)
{
    std::uniform_int_distribution<uint64_t> dist(1ull, maxDelta);
    uint64_t acc = start;
    for (auto & v : data)
    {
        acc += dist(rng);
        v = acc;
    }
}

} // namespace

unsigned runD1Enc32CompatibilityTest()
{
    std::mt19937 rng(12345u);
    unsigned passed = 0, failed = 0;

    std::printf("=== D1 Encode 32-bit Compatibility Test ===\n");
    std::printf("=== Verifying C p4d1enc32 <-> C++ p4D1Enc32 ===\n\n");

    for (unsigned n = 1; n <= 127; ++n)
    {
        for (unsigned maxDelta : {1u, 15u, 255u, 65535u})
        {
            uint32_t start = static_cast<uint32_t>(n * maxDelta % 1000u + 1u);
            std::vector<uint32_t> input(n);
            makeSorted32(input, rng, maxDelta, start);

            const unsigned extra = 32u;
            std::vector<uint32_t> in_c(n + extra, 0u);
            std::vector<uint32_t> in_cxx(n + extra, 0u);
            std::copy(input.begin(), input.end(), in_c.begin());
            std::copy(input.begin(), input.end(), in_cxx.begin());

            std::vector<unsigned char> c_buf(n * 5 + 256, 0u);
            std::vector<unsigned char> cxx_buf(n * 5 + 256, 0u);

            unsigned char * c_end = ::p4d1enc32(in_c.data(), n, c_buf.data(), start);
            unsigned char * cxx_end = turbopfor::p4D1Enc32(in_cxx.data(), n, cxx_buf.data(), start);

            size_t c_len = c_end - c_buf.data();
            size_t cxx_len = cxx_end - cxx_buf.data();

            normalizeP4Enc32(c_buf.data(), n);
            normalizeP4Enc32(cxx_buf.data(), n);

            bool ok = (c_len == cxx_len) && std::memcmp(c_buf.data(), cxx_buf.data(), c_len) == 0;

            if (ok)
            {
                std::vector<uint32_t> decoded(n, 0u);
                ::p4d1dec32(cxx_buf.data(), n, decoded.data(), start);
                if (decoded != input)
                    ok = false;
            }

            if (ok)
                ++passed;
            else
            {
                std::fprintf(stderr, "FAIL [p4D1Enc32 n=%u maxDelta=%u]: len C=%zu C++=%zu\n", n, maxDelta, c_len, cxx_len);
                ++failed;
            }
        }
    }

    std::printf("p4D1Enc32: %u passed, %u failed\n\n", passed, failed);
    return failed;
}

unsigned runD1Enc128v32CompatibilityTest()
{
    std::mt19937 rng(54321u);
    unsigned passed = 0, failed = 0;

    std::printf("=== D1 Encode 128v32 Compatibility Test ===\n");

    for (unsigned n : {128u})
    {
        for (unsigned maxDelta : {1u, 15u, 255u, 65535u})
        {
            uint32_t start = static_cast<uint32_t>(n * maxDelta % 1000u + 1u);
            std::vector<uint32_t> input(n);
            makeSorted32(input, rng, maxDelta, start);

            const unsigned extra = 32u;
            std::vector<uint32_t> in_c(n + extra, 0u);
            std::vector<uint32_t> in_cxx(n + extra, 0u);
            std::copy(input.begin(), input.end(), in_c.begin());
            std::copy(input.begin(), input.end(), in_cxx.begin());

            std::vector<unsigned char> c_buf(n * 5 + 256, 0u);
            std::vector<unsigned char> cxx_buf(n * 5 + 256, 0u);

            unsigned char * c_end = ::p4d1enc128v32(in_c.data(), n, c_buf.data(), start);
            unsigned char * cxx_end = turbopfor::p4D1Enc128v32(in_cxx.data(), n, cxx_buf.data(), start);

            size_t c_len = c_end - c_buf.data();
            size_t cxx_len = cxx_end - cxx_buf.data();

            normalizeP4Enc32(c_buf.data(), n);
            normalizeP4Enc32(cxx_buf.data(), n);

            bool ok = (c_len == cxx_len) && std::memcmp(c_buf.data(), cxx_buf.data(), c_len) == 0;

            if (ok)
            {
                std::vector<uint32_t> decoded(n, 0u);
                ::p4d1dec128v32(cxx_buf.data(), n, decoded.data(), start);
                if (decoded != input)
                    ok = false;
            }

            if (ok)
                ++passed;
            else
            {
                std::fprintf(stderr, "FAIL [p4D1Enc128v32 n=%u maxDelta=%u]: len C=%zu C++=%zu\n", n, maxDelta, c_len, cxx_len);
                ++failed;
            }
        }
    }

    std::printf("p4D1Enc128v32: %u passed, %u failed\n\n", passed, failed);
    return failed;
}

unsigned runD1Enc256v32CompatibilityTest()
{
    std::mt19937 rng(67890u);
    unsigned passed = 0, failed = 0;

    std::printf("=== D1 Encode 256v32 Compatibility Test ===\n");

    for (unsigned n : {256u})
    {
        for (unsigned maxDelta : {1u, 15u, 255u, 65535u})
        {
            uint32_t start = static_cast<uint32_t>(n * maxDelta % 1000u + 1u);
            std::vector<uint32_t> input(n);
            makeSorted32(input, rng, maxDelta, start);

            const unsigned extra = 32u;
            std::vector<uint32_t> in_c(n + extra, 0u);
            std::vector<uint32_t> in_cxx(n + extra, 0u);
            std::copy(input.begin(), input.end(), in_c.begin());
            std::copy(input.begin(), input.end(), in_cxx.begin());

            std::vector<unsigned char> c_buf(n * 5 + 256, 0u);
            std::vector<unsigned char> cxx_buf(n * 5 + 256, 0u);

            unsigned char * c_end = ::p4d1enc256v32(in_c.data(), n, c_buf.data(), start);
            unsigned char * cxx_end = turbopfor::p4D1Enc256v32(in_cxx.data(), n, cxx_buf.data(), start);

            size_t c_len = c_end - c_buf.data();
            size_t cxx_len = cxx_end - cxx_buf.data();

            normalizeP4Enc32(c_buf.data(), n);
            normalizeP4Enc32(cxx_buf.data(), n);

            bool ok = (c_len == cxx_len) && std::memcmp(c_buf.data(), cxx_buf.data(), c_len) == 0;

            if (ok)
            {
                std::vector<uint32_t> decoded(n, 0u);
                ::p4d1dec256v32(cxx_buf.data(), n, decoded.data(), start);
                if (decoded != input)
                    ok = false;
            }

            if (ok)
                ++passed;
            else
            {
                std::fprintf(stderr, "FAIL [p4D1Enc256v32 n=%u maxDelta=%u]: len C=%zu C++=%zu\n", n, maxDelta, c_len, cxx_len);
                ++failed;
            }
        }
    }

    std::printf("p4D1Enc256v32: %u passed, %u failed\n\n", passed, failed);
    return failed;
}

unsigned runD1Enc64CompatibilityTest()
{
    std::mt19937_64 rng(11111ull);
    unsigned passed = 0, failed = 0;

    std::printf("=== D1 Encode 64-bit Compatibility Test ===\n");

    for (unsigned n = 1; n <= 127; ++n)
    {
        for (uint64_t maxDelta : {1ull, 255ull, 65535ull, 0xFFFFFFFFull})
        {
            uint64_t start = static_cast<uint64_t>(n * 123u + 1u);
            std::vector<uint64_t> input(n);
            makeSorted64(input, rng, maxDelta, start);

            const unsigned extra = 32u;
            std::vector<uint64_t> in_c(n + extra, 0ull);
            std::vector<uint64_t> in_cxx(n + extra, 0ull);
            std::copy(input.begin(), input.end(), in_c.begin());
            std::copy(input.begin(), input.end(), in_cxx.begin());

            std::vector<unsigned char> c_buf(n * 10 + 256, 0u);
            std::vector<unsigned char> cxx_buf(n * 10 + 256, 0u);

            unsigned char * c_end = ::p4d1enc64(in_c.data(), n, c_buf.data(), start);
            unsigned char * cxx_end = turbopfor::p4D1Enc64(in_cxx.data(), n, cxx_buf.data(), start);

            size_t c_len = c_end - c_buf.data();
            size_t cxx_len = cxx_end - cxx_buf.data();

            normalizeP4Enc64(c_buf.data(), n);
            normalizeP4Enc64(cxx_buf.data(), n);

            bool ok = (c_len == cxx_len) && std::memcmp(c_buf.data(), cxx_buf.data(), c_len) == 0;

            if (ok)
            {
                std::vector<uint64_t> decoded(n, 0ull);
                ::p4d1dec64(reinterpret_cast<unsigned char *>(cxx_buf.data()), n, decoded.data(), start);
                if (decoded != input)
                    ok = false;
            }

            if (ok)
                ++passed;
            else
            {
                std::fprintf(stderr, "FAIL [p4D1Enc64 n=%u maxDelta=%llu]: len C=%zu C++=%zu\n", n, (unsigned long long)maxDelta, c_len, cxx_len);
                ++failed;
            }
        }
    }

    std::printf("p4D1Enc64: %u passed, %u failed\n\n", passed, failed);
    return failed;
}

unsigned runD1Enc128v64RoundtripTest()
{
    std::mt19937_64 rng(22222ull);
    unsigned passed = 0, failed = 0;

    std::printf("=== D1 Encode 128v64 Roundtrip Test ===\n");

    for (unsigned n : {128u})
    {
        for (uint64_t maxDelta : {1ull, 255ull, 65535ull, 0xFFFFFFFFull, 0xFFFFFFFFFFFFull})
        {
            uint64_t start = static_cast<uint64_t>(n * 123u + 1u);
            std::vector<uint64_t> input(n);
            makeSorted64(input, rng, maxDelta, start);

            const unsigned extra = 32u;
            std::vector<uint64_t> in_cxx(n + extra, 0ull);
            std::copy(input.begin(), input.end(), in_cxx.begin());

            std::vector<unsigned char> buf(n * 10 + 256, 0u);

            turbopfor::p4D1Enc128v64(in_cxx.data(), n, buf.data(), start);

            std::vector<uint64_t> decoded(n, 0ull);
            turbopfor::p4D1Dec128v64(buf.data(), n, decoded.data(), start);

            bool ok = (decoded == input);
            if (ok)
                ++passed;
            else
            {
                std::fprintf(stderr, "FAIL [p4D1Enc128v64 n=%u maxDelta=%llu]: roundtrip mismatch\n", n, (unsigned long long)maxDelta);
                ++failed;
            }
        }
    }

    std::printf("p4D1Enc128v64: %u passed, %u failed\n\n", passed, failed);
    return failed;
}

unsigned runD1Enc256v64RoundtripTest()
{
    std::mt19937_64 rng(33333ull);
    unsigned passed = 0, failed = 0;

    std::printf("=== D1 Encode 256v64 Roundtrip Test ===\n");

    for (unsigned n : {256u})
    {
        for (uint64_t maxDelta : {1ull, 255ull, 65535ull, 0xFFFFFFFFull, 0xFFFFFFFFFFFFull})
        {
            uint64_t start = static_cast<uint64_t>(n * 123u + 1u);
            std::vector<uint64_t> input(n);
            makeSorted64(input, rng, maxDelta, start);

            const unsigned extra = 32u;
            std::vector<uint64_t> in_cxx(n + extra, 0ull);
            std::copy(input.begin(), input.end(), in_cxx.begin());

            std::vector<unsigned char> buf(n * 10 + 256, 0u);

            turbopfor::p4D1Enc256v64(in_cxx.data(), n, buf.data(), start);

            std::vector<uint64_t> decoded(n, 0ull);
            turbopfor::p4D1Dec256v64(buf.data(), n, decoded.data(), start);

            bool ok = (decoded == input);
            if (ok)
                ++passed;
            else
            {
                std::fprintf(stderr, "FAIL [p4D1Enc256v64 n=%u maxDelta=%llu]: roundtrip mismatch\n", n, (unsigned long long)maxDelta);
                ++failed;
            }
        }
    }

    std::printf("p4D1Enc256v64: %u passed, %u failed\n\n", passed, failed);
    return failed;
}
