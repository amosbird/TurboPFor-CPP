#include "test_helpers.h"

/// Test that p4Dec256v32 / p4D1Dec256v32 SIMD decode matches scalar decode
/// for data patterns that trigger bitmap exception encoding with many exceptions.
///
/// The bug: encoder packs exceptions with scalar bitpack (horizontal layout),
/// but the old SIMD decoder unpacked them with vertical/interleaved SIMD tables.
/// This produced wrong results when b=0 and bx<=32 (bitmap exception path)
/// with >=1 exception.
///
/// Also tests p4Enc256v32 + p4Dec256v32 round-trip (non-delta).

namespace
{

struct BitmapExcTestCase
{
    std::string name;
    std::vector<uint32_t> data;
};

std::vector<BitmapExcTestCase> generateTestCases()
{
    std::vector<BitmapExcTestCase> cases;

    // Case 1: ClickHouse projection text index packed_block_ranges pattern
    // Alternating gaps 1 and 692 → b=0, bx=10, 129 exceptions
    {
        std::vector<uint32_t> d(256);
        d[0] = 19;
        for (int i = 1; i < 256; i++)
            d[i] = (i % 2 == 0) ? 0 : 691;
        cases.push_back({"alternating_0_691", std::move(d)});
    }

    // Case 2: half zeros half large → b=0, many exceptions
    {
        std::vector<uint32_t> d(256, 0);
        for (int i = 0; i < 128; i++)
            d[i * 2 + 1] = 1000;
        cases.push_back({"alternating_0_1000", std::move(d)});
    }

    // Case 3: mostly zeros with a few exceptions → b=0, few exceptions
    {
        std::vector<uint32_t> d(256, 0);
        d[0] = 42;
        d[100] = 100;
        d[200] = 200;
        cases.push_back({"sparse_exceptions", std::move(d)});
    }

    // Case 4: 129 exceptions at various bit widths
    for (unsigned bx : {1, 5, 10, 16, 20, 31, 32})
    {
        std::vector<uint32_t> d(256, 0);
        uint32_t exc_val = (1u << bx) - 1;
        for (int i = 0; i < 129; i++)
            d[i * 2 % 256] = exc_val;
        cases.push_back({"exc129_bx" + std::to_string(bx), std::move(d)});
    }

    // Case 5: all exceptions (256 non-zero values, b=0)
    {
        std::vector<uint32_t> d(256);
        for (int i = 0; i < 256; i++)
            d[i] = i + 1;
        cases.push_back({"all_nonzero_sequential", std::move(d)});
    }

    // Case 6: random data that triggers b=0
    {
        std::mt19937 rng(12345);
        std::vector<uint32_t> d(256);
        std::uniform_int_distribution<uint32_t> dist(0, 1);
        for (auto & v : d)
            v = dist(rng) ? 500 : 0;
        cases.push_back({"random_sparse_500", std::move(d)});
    }

    // Case 7: b=1 with exceptions (ensures base bits are also tested)
    {
        std::vector<uint32_t> d(256);
        for (int i = 0; i < 256; i++)
            d[i] = (i % 3 == 0) ? 1000 : (i % 2);
        cases.push_back({"b1_with_exceptions", std::move(d)});
    }

    // Case 8: monotonically increasing (for D1 decode test)
    {
        std::vector<uint32_t> d(256);
        d[0] = 20;
        for (int i = 1; i < 256; i++)
            d[i] = d[i-1] + ((i % 2 == 0) ? 1 : 693);
        cases.push_back({"monotone_alternating_gap", std::move(d)});
    }

    return cases;
}

} // namespace

unsigned runBitmapExceptionTest()
{
    unsigned passed = 0;
    unsigned failed = 0;

    std::printf("=== Bitmap Exception Decode Test (256v32) ===\n");
    std::printf("=== Verifying SIMD p4Dec256v32/p4D1Dec256v32 matches scalar for exception-heavy patterns ===\n\n");

    auto cases = generateTestCases();

    for (const auto & tc : cases)
    {
        alignas(64) uint32_t input[256];
        std::memcpy(input, tc.data.data(), 256 * sizeof(uint32_t));

        // --- p4Enc256v32 + p4Dec256v32 round-trip ---
        {
            alignas(64) uint8_t packed[4096] = {};
            uint8_t * end = turbopfor::p4Enc256v32(input, 256, packed);
            size_t bytes = static_cast<size_t>(end - packed);

            // SIMD decode (via dispatch — goes to SIMD on AVX2 machines)
            alignas(64) uint32_t dec_simd[256] = {};
            const uint8_t * consumed = turbopfor::p4Dec256v32(packed, 256, dec_simd);

            // Verify consumed bytes
            if (static_cast<size_t>(consumed - packed) != bytes)
            {
                std::printf("  FAIL [%s] p4Dec256v32: consumed %zu bytes, encoded %zu\n",
                            tc.name.c_str(), static_cast<size_t>(consumed - packed), bytes);
                ++failed;
            }

            // Verify values
            bool ok = true;
            for (int i = 0; i < 256; i++)
            {
                if (dec_simd[i] != input[i])
                {
                    if (ok)
                        std::printf("  FAIL [%s] p4Dec256v32: mismatch at [%d] expected=%u got=%u\n",
                                    tc.name.c_str(), i, input[i], dec_simd[i]);
                    ok = false;
                }
            }
            if (ok) ++passed; else ++failed;
        }

        // --- p4D1Enc256v32 + p4D1Dec256v32 round-trip (only for monotone data) ---
        {
            // Check if data is monotonically increasing
            bool monotone = true;
            for (int i = 1; i < 256 && monotone; i++)
                if (input[i] <= input[i-1]) monotone = false;

            if (monotone)
            {
                uint32_t start = input[0] > 0 ? input[0] - 1 : 0;
                alignas(64) uint8_t packed[4096] = {};
                uint8_t * end = turbopfor::p4D1Enc256v32(input, 256, packed, start);
                size_t bytes = static_cast<size_t>(end - packed);

                alignas(64) uint32_t dec_d1[256] = {};
                const uint8_t * consumed = turbopfor::p4D1Dec256v32(packed, 256, dec_d1, start);

                if (static_cast<size_t>(consumed - packed) != bytes)
                {
                    std::printf("  FAIL [%s] p4D1Dec256v32: consumed %zu bytes, encoded %zu\n",
                                tc.name.c_str(), static_cast<size_t>(consumed - packed), bytes);
                    ++failed;
                }

                bool ok = true;
                for (int i = 0; i < 256; i++)
                {
                    if (dec_d1[i] != input[i])
                    {
                        if (ok)
                            std::printf("  FAIL [%s] p4D1Dec256v32: mismatch at [%d] expected=%u got=%u\n",
                                        tc.name.c_str(), i, input[i], dec_d1[i]);
                        ok = false;
                    }
                }
                if (ok) ++passed; else ++failed;
            }
        }

        // --- Cross-validate: SIMD round-trip matches scalar round-trip ---
        {
            // SIMD encode + SIMD decode
            alignas(64) uint8_t packed_simd[4096] = {};
            turbopfor::p4Enc256v32(input, 256, packed_simd);
            alignas(64) uint32_t dec_simd[256] = {};
            turbopfor::p4Dec256v32(packed_simd, 256, dec_simd);

            // Scalar encode + scalar decode
            alignas(64) uint8_t packed_scalar[4096] = {};
            turbopfor::p4Enc32(input, 256, packed_scalar);
            alignas(64) uint32_t dec_scalar[256] = {};
            turbopfor::p4Dec32(packed_scalar, 256, dec_scalar);

            bool ok = true;
            for (int i = 0; i < 256; i++)
            {
                if (dec_simd[i] != dec_scalar[i])
                {
                    if (ok)
                        std::printf("  FAIL [%s] SIMD vs scalar round-trip: [%d] simd=%u scalar=%u input=%u\n",
                                    tc.name.c_str(), i, dec_simd[i], dec_scalar[i], input[i]);
                    ok = false;
                }
            }
            if (ok) ++passed; else ++failed;
        }
    }

    std::printf("\n%u passed, %u failed\n\n", passed, failed);
    return failed;
}
