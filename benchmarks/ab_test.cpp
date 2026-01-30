/// TurboPFor A/B Benchmark Suite
/// Benchmarks encode/decode operations and bit-packing operations
/// Supports scalar and SIMD implementations with various input patterns

#include "scalar/p4_scalar.h"
#include "simd/p4_simd.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Reference C implementations
extern "C" unsigned char * p4enc32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4d1dec32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * bitpack32(unsigned * in, unsigned n, unsigned char * out, unsigned b);
extern "C" unsigned char * bitunpack32(const unsigned char * in, unsigned n, uint32_t * out, unsigned b);
extern "C" unsigned char * bitd1unpack32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);

// SIMD Reference declarations
extern "C" unsigned char * p4enc128v32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4d1dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * p4enc256v32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4d1dec256v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

namespace turbopfor::scalar::detail
{
unsigned char * bitpack32Scalar(const uint32_t * in, unsigned n, unsigned char * out, unsigned b);
unsigned char * bitunpack32Scalar(unsigned char * in, unsigned n, uint32_t * out, unsigned b);
unsigned char * bitunpackd1_32Scalar(unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);
}

namespace
{

using Clock = std::chrono::high_resolution_clock;

// =============================================================================
// Timing Utilities
// =============================================================================

/// Returns elapsed time in seconds since the given timepoint
double secondsSince(Clock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - start).count();
}

// =============================================================================
// Data Structures
// =============================================================================

/// Result of a p4enc32/p4d1dec32 encode/decode benchmark with multiple SIMD variants
struct BenchResult
{
    double ref_enc_mb_s; ///< Reference implementation encoding throughput
    double our_enc_mb_s; ///< Our implementation encoding throughput
    double ref_dec_mb_s; ///< Reference implementation decoding throughput
    double our_dec_mb_s; ///< Our implementation decoding throughput
};

/// Result of a bitunpack32 benchmark
struct BitunpackResult
{
    double ref_mb_s; ///< Reference implementation throughput
    double our_mb_s; ///< Our implementation throughput
};

/// Result of a bitunpackd1_32 (delta-encoded) benchmark
struct BitunpackD1Result
{
    double ref_mb_s; ///< Reference implementation throughput
    double our_mb_s; ///< Our implementation throughput
};

/// Result of a bitpack32 benchmark
struct BitpackResult
{
    double ref_mb_s; ///< Reference implementation throughput
    double our_mb_s; ///< Our implementation throughput
};

/// Test scenario with exception percentage for controlled failure testing
struct Scenario
{
    double pct; ///< Exception percentage (-1.0 for random data, 0-100 for forced exceptions)
    const char * desc; ///< Human-readable scenario description
};

// =============================================================================
// Command Line Arguments Structure
// =============================================================================

/// Parsed command-line arguments for the benchmark suite
struct CommandLineArgs
{
    unsigned n_start = 1; ///< Starting element count
    unsigned n_end = 127; ///< Ending element count
    unsigned iters = 100000u; ///< Iterations per benchmark
    unsigned runs = 3u; ///< Runs per benchmark (take best)
    double exc_pct = -1.0; ///< Exception percentage (-1.0 = random)
    bool single_n = false; ///< Test only a single n value
    bool bitpack_only = false; ///< Test bitpack32 only
    bool bitunpack_only = false; ///< Test bitunpack32 only
    bool bitunpackd1_only = false; ///< Test bitunpackd1_32 only
    bool simd128 = false; ///< Test 128-bit SIMD variant
    bool simd256 = false; ///< Test 256-bit SIMD variant

    /// Validates argument consistency and prints errors if invalid
    bool validate() const
    {
        // Check SIMD exclusivity
        if (simd128 && simd256)
        {
            std::fprintf(stderr, "Error: Cannot run both --simd128 and --simd256 at the same time\n");
            return false;
        }

        // Check SIMD and bitpack test incompatibility
        if ((simd128 || simd256) && (bitpack_only || bitunpack_only || bitunpackd1_only))
        {
            std::fprintf(stderr, "Error: SIMD tests cannot be combined with bitpack/unpack tests\n");
            return false;
        }

        // Check element count range (only for non-SIMD tests)
        if (!simd128 && !simd256 && (n_start < 1 || n_end > 127 || n_start > n_end))
        {
            std::fprintf(stderr, "Error: n must be in range [1, 127] and start <= end\n");
            return false;
        }

        // Check bit operation test exclusivity
        unsigned bit_tests = (bitpack_only ? 1 : 0) + (bitunpack_only ? 1 : 0) + (bitunpackd1_only ? 1 : 0);
        if (bit_tests > 1)
        {
            std::fprintf(stderr, "Error: --bitpack, --bitunpack, and --bitunpackd1 are mutually exclusive\n");
            return false;
        }

        return true;
    }
};

// =============================================================================
// Argument Parsing
// =============================================================================

/// Parses command-line arguments into a CommandLineArgs struct
/// \param argc Number of command-line arguments
/// \param argv Array of command-line argument strings
/// \param args Output parameter for parsed arguments
/// \return true if parsing succeeded, false if an error occurred
bool parseArguments(int argc, char ** argv, CommandLineArgs & args)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            return false; // Signal to show help
        }
        else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc)
        {
            args.n_start = args.n_end = static_cast<unsigned>(std::atoi(argv[++i]));
            args.single_n = true;
        }
        else if (std::strcmp(argv[i], "--n-range") == 0 && i + 1 < argc)
        {
            if (std::sscanf(argv[++i], "%u-%u", &args.n_start, &args.n_end) != 2)
            {
                std::fprintf(stderr, "Error: Invalid range format. Use: --n-range <start>-<end>\n");
                return false;
            }
        }
        else if (std::strcmp(argv[i], "--all") == 0)
        {
            args.n_start = 1;
            args.n_end = 127;
        }
        else if (std::strcmp(argv[i], "--bitpack") == 0)
        {
            args.bitpack_only = true;
        }
        else if (std::strcmp(argv[i], "--bitunpack") == 0)
        {
            args.bitunpack_only = true;
        }
        else if (std::strcmp(argv[i], "--bitunpackd1") == 0)
        {
            args.bitunpackd1_only = true;
        }
        else if (std::strcmp(argv[i], "--simd128") == 0)
        {
            args.simd128 = true;
        }
        else if (std::strcmp(argv[i], "--simd256") == 0)
        {
            args.simd256 = true;
        }
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
        {
            args.iters = static_cast<unsigned>(std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
        {
            args.runs = static_cast<unsigned>(std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--exc-pct") == 0 && i + 1 < argc)
        {
            args.exc_pct = std::atof(argv[++i]);
        }
        else
        {
            std::fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            return false;
        }
    }

    return true;
}

/// Prints usage information for the benchmark tool
void printUsage(const char * prog)
{
    std::printf("Usage: %s [options]\n", prog);
    std::printf("Options:\n");
    std::printf("  --n <value>        Test specific element count (1-127)\n");
    std::printf("  --n-range <start>-<end>  Test range of element counts\n");
    std::printf("  --all              Test all element counts from 1 to 127 (default)\n");
    std::printf("  --bitpack          Benchmark bitpack32 vs bitpack32Scalar\n");
    std::printf("  --bitunpack        Benchmark bitunpack32 vs bitunpack32Scalar\n");
    std::printf("  --bitunpackd1      Benchmark bitd1unpack32 vs bitunpackd1_32Scalar\n");
    std::printf("  --simd128          Test 128v SIMD (n=128)\n");
    std::printf("  --simd256          Test 256v SIMD (n=256)\n");
    std::printf("  --iters <count>    Number of iterations (default: 100000)\n");
    std::printf("  --runs <count>     Number of runs per test (default: 3)\n");
    std::printf("  --exc-pct <pct>    Force percentage of exceptions (values > 2^bw)\n");
    std::printf("Note: p4enc32/p4d1dec32 operate on 32-bit integers.\n");
    std::printf("      n = number of 32-bit elements (not bit width)\n");
    std::printf("Examples:\n");
    std::printf("  %s --n 32              # Test with 32 elements\n", prog);
    std::printf("  %s --n-range 8-16      # Test with 8 to 16 elements\n", prog);
    std::printf("  %s --all --iters 50000 # Test all with 50k iters\n", prog);
    std::printf("  %s --simd128           # Test 128v SIMD\n", prog);
}

// =============================================================================
// Benchmark Functions for Bit-packing Operations
// =============================================================================

/// Benchmarks bitpack32 (reference) vs bitpack32Scalar (ours)
/// \param input Random input data to pack
/// \param bit_width Bit width for packing
/// \param iters Total iterations to run
/// \return BitpackResult with throughput for both implementations
BitpackResult runBitpackBenchmark(const std::vector<uint32_t> & input, unsigned bit_width, unsigned iters)
{
    const unsigned num_elements = static_cast<unsigned>(input.size());
    std::vector<unsigned char> ref_buf(num_elements * 4u + 64u, 0u);
    std::vector<unsigned char> our_buf(num_elements * 4u + 64u, 0u);

    // Warmup phase to stabilize cache and branch predictors
    for (unsigned i = 0; i < 1000; ++i)
    {
        ::bitpack32(const_cast<unsigned *>(reinterpret_cast<const unsigned *>(input.data())), num_elements, ref_buf.data(), bit_width);
        turbopfor::scalar::detail::bitpack32Scalar(input.data(), num_elements, our_buf.data(), bit_width);
    }

    // Encode benchmark - interleaved to reduce measurement bias
    double ref_sec = 0.0;
    double our_sec = 0.0;
    size_t ref_bytes = 0;
    size_t our_bytes = 0;

    const unsigned chunk = 10000;
    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            unsigned char * end = ::bitpack32(
                const_cast<unsigned *>(reinterpret_cast<const unsigned *>(input.data())), num_elements, ref_buf.data(), bit_width);
            ref_bytes += static_cast<size_t>(end - ref_buf.data());
        }
        ref_sec += secondsSince(start);

        start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            unsigned char * end = turbopfor::scalar::detail::bitpack32Scalar(input.data(), num_elements, our_buf.data(), bit_width);
            our_bytes += static_cast<size_t>(end - our_buf.data());
        }
        our_sec += secondsSince(start);
    }

    BitpackResult result;
    result.ref_mb_s = ref_bytes / (1024.0 * 1024.0) / ref_sec;
    result.our_mb_s = our_bytes / (1024.0 * 1024.0) / our_sec;
    return result;
}

/// Benchmarks bitunpack32 (reference) vs bitunpack32Scalar (ours)
/// \param input Random input data to pack then unpack
/// \param bit_width Bit width for unpacking
/// \param iters Total iterations to run
/// \return BitunpackResult with throughput for both implementations
BitunpackResult runBitunpackBenchmark(const std::vector<uint32_t> & input, unsigned bit_width, unsigned iters)
{
    const unsigned num_elements = static_cast<unsigned>(input.size());
    std::vector<unsigned char> buf(num_elements * 4u + 64u, 0u);
    std::vector<uint32_t> out(num_elements, 0u);

    // Pack input data first
    unsigned char * packed_end
        = ::bitpack32(const_cast<unsigned *>(reinterpret_cast<const unsigned *>(input.data())), num_elements, buf.data(), bit_width);
    const size_t packed_bytes = static_cast<size_t>(packed_end - buf.data());

    // Warmup phase to stabilize cache and branch predictors
    for (unsigned i = 0; i < 1000; ++i)
    {
        ::bitunpack32(buf.data(), num_elements, out.data(), bit_width);
        turbopfor::scalar::detail::bitunpack32Scalar(buf.data(), num_elements, out.data(), bit_width);
    }

    // Decode benchmark - interleaved to reduce measurement bias
    double ref_sec = 0.0;
    double our_sec = 0.0;
    size_t total_bytes = 0;

    const unsigned chunk = 10000;
    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
            ::bitunpack32(buf.data(), num_elements, out.data(), bit_width);
        ref_sec += secondsSince(start);

        start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
            turbopfor::scalar::detail::bitunpack32Scalar(buf.data(), num_elements, out.data(), bit_width);
        our_sec += secondsSince(start);

        total_bytes += packed_bytes * count;
    }

    BitunpackResult result;
    result.ref_mb_s = total_bytes / (1024.0 * 1024.0) / ref_sec;
    result.our_mb_s = total_bytes / (1024.0 * 1024.0) / our_sec;
    return result;
}

/// Benchmarks bitd1unpack32 (reference) vs bitunpackd1_32Scalar (ours) - delta-encoded
/// \param input Random input data to pack then decode
/// \param bit_width Bit width for unpacking
/// \param iters Total iterations to run
/// \param start Initial value for delta decoding
/// \return BitunpackD1Result with throughput for both implementations
BitunpackD1Result runBitunpackD1Benchmark(const std::vector<uint32_t> & input, unsigned bit_width, unsigned iters, uint32_t start)
{
    const unsigned num_elements = static_cast<unsigned>(input.size());
    std::vector<unsigned char> buf(num_elements * 4u + 64u, 0u);
    std::vector<uint32_t> out(num_elements, 0u);

    // Pack input data first
    unsigned char * packed_end
        = ::bitpack32(const_cast<unsigned *>(reinterpret_cast<const unsigned *>(input.data())), num_elements, buf.data(), bit_width);
    const size_t packed_bytes = static_cast<size_t>(packed_end - buf.data());

    // Warmup phase to stabilize cache and branch predictors
    for (unsigned i = 0; i < 1000; ++i)
    {
        ::bitd1unpack32(buf.data(), num_elements, out.data(), start, bit_width);
        turbopfor::scalar::detail::bitunpackd1_32Scalar(buf.data(), num_elements, out.data(), start, bit_width);
    }

    // Decode benchmark - interleaved to reduce measurement bias
    double ref_sec = 0.0;
    double our_sec = 0.0;
    size_t total_bytes = 0;

    const unsigned chunk = 10000;
    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto start_time = Clock::now();
        for (unsigned i = 0; i < count; ++i)
            ::bitd1unpack32(buf.data(), num_elements, out.data(), start, bit_width);
        ref_sec += secondsSince(start_time);

        start_time = Clock::now();
        for (unsigned i = 0; i < count; ++i)
            turbopfor::scalar::detail::bitunpackd1_32Scalar(buf.data(), num_elements, out.data(), start, bit_width);
        our_sec += secondsSince(start_time);

        total_bytes += packed_bytes * count;
    }

    BitunpackD1Result result;
    result.ref_mb_s = total_bytes / (1024.0 * 1024.0) / ref_sec;
    result.our_mb_s = total_bytes / (1024.0 * 1024.0) / our_sec;
    return result;
}

// =============================================================================
// Benchmark Functions for P4 Encoding/Decoding
// =============================================================================

/// Benchmarks p4enc/p4d1dec operations (scalar, 128-bit SIMD, or 256-bit SIMD)
/// \param input Random input data to encode then decode
/// \param iters Total iterations to run
/// \param simd128 If true, benchmark 128-bit SIMD; if false and simd256 false, use scalar
/// \param simd256 If true, benchmark 256-bit SIMD; if false and simd128 false, use scalar
/// \return BenchResult with encode/decode throughput for both implementations
BenchResult runBenchmark(const std::vector<uint32_t> & input, unsigned iters, bool simd128, bool simd256)
{
    const unsigned num_elements = static_cast<unsigned>(input.size());

    // Helper to align pointer to 32-byte boundary
    auto get_aligned_ptr = [](std::vector<unsigned char> & buf) -> unsigned char *
    {
        unsigned char * ptr = buf.data();
        size_t remainder = reinterpret_cast<uintptr_t>(ptr) % 32;
        if (remainder)
            ptr += (32 - remainder);
        return ptr;
    };

    // Helper to align uint32_t pointer to 32-byte boundary
    auto get_aligned_u32_ptr = [](std::vector<uint32_t> & buf) -> uint32_t *
    {
        uint32_t * ptr = buf.data();
        size_t remainder = reinterpret_cast<uintptr_t>(ptr) % 32;
        if (remainder)
        {
            size_t bytes_needed = 32 - remainder;
            ptr += bytes_needed / 4;
        }
        return ptr;
    };

    // Prepare aligned buffers with extra padding
    std::vector<uint32_t> input_copy = input;
    input_copy.resize(num_elements + 64, 0u);

    std::vector<unsigned char> ref_buf_vec(num_elements * 5 + 512, 0u);
    unsigned char * ref_buf = get_aligned_ptr(ref_buf_vec);

    std::vector<unsigned char> our_buf_vec(num_elements * 5 + 512, 0u);
    unsigned char * our_buf = get_aligned_ptr(our_buf_vec);

    std::vector<uint32_t> out_vec(num_elements + 128, 0u);
    uint32_t * out = get_aligned_u32_ptr(out_vec);

    // Warmup phase to stabilize cache and branch predictors
    for (unsigned i = 0; i < 1000; ++i)
    {
        if (simd128)
        {
            ::p4enc128v32(input_copy.data(), num_elements, ref_buf);
            turbopfor::simd::p4Enc128v32(input_copy.data(), num_elements, our_buf);
            ::p4d1dec128v32(ref_buf, num_elements, out, 0u);
            turbopfor::simd::p4D1Dec128v32(our_buf, num_elements, out, 0u);
        }
        else if (simd256)
        {
            ::p4enc256v32(input_copy.data(), num_elements, ref_buf);
            turbopfor::simd::p4Enc256v32(input_copy.data(), num_elements, our_buf);
            ::p4d1dec256v32(ref_buf, num_elements, out, 0u);
            turbopfor::simd::p4D1Dec256v32(our_buf, num_elements, out, 0u);
        }
        else
        {
            ::p4enc32(input_copy.data(), num_elements, ref_buf);
            turbopfor::scalar::p4Enc32(input_copy.data(), num_elements, our_buf);
            ::p4d1dec32(ref_buf, num_elements, out, 0u);
            turbopfor::scalar::p4D1Dec32(our_buf, num_elements, out, 0u);
        }
    }

    // Encode benchmark - interleaved to reduce measurement bias
    double ref_enc_sec = 0.0;
    double our_enc_sec = 0.0;
    size_t ref_bytes = 0;
    size_t our_bytes = 0;

    const unsigned chunk = 10000;
    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            unsigned char * end = nullptr;
            if (simd128)
                end = ::p4enc128v32(input_copy.data(), num_elements, ref_buf);
            else if (simd256)
                end = ::p4enc256v32(input_copy.data(), num_elements, ref_buf);
            else
                end = ::p4enc32(input_copy.data(), num_elements, ref_buf);

            ref_bytes += static_cast<size_t>(end - ref_buf);
        }
        ref_enc_sec += secondsSince(start);

        start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            unsigned char * end = nullptr;
            if (simd128)
                end = turbopfor::simd::p4Enc128v32(input_copy.data(), num_elements, our_buf);
            else if (simd256)
                end = turbopfor::simd::p4Enc256v32(input_copy.data(), num_elements, our_buf);
            else
                end = turbopfor::scalar::p4Enc32(input_copy.data(), num_elements, our_buf);

            our_bytes += static_cast<size_t>(end - our_buf);
        }
        our_enc_sec += secondsSince(start);
    }

    // Decode benchmark - interleaved to reduce measurement bias
    double ref_dec_sec = 0.0;
    double our_dec_sec = 0.0;

    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            if (simd128)
                ::p4d1dec128v32(ref_buf, num_elements, out, 0u);
            else if (simd256)
                ::p4d1dec256v32(ref_buf, num_elements, out, 0u);
            else
                ::p4d1dec32(ref_buf, num_elements, out, 0u);
        }
        ref_dec_sec += secondsSince(start);

        start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            if (simd128)
                turbopfor::simd::p4D1Dec128v32(our_buf, num_elements, out, 0u);
            else if (simd256)
                turbopfor::simd::p4D1Dec256v32(our_buf, num_elements, out, 0u);
            else
                turbopfor::scalar::p4D1Dec32(our_buf, num_elements, out, 0u);
        }
        our_dec_sec += secondsSince(start);
    }

    BenchResult result;
    result.ref_enc_mb_s = ref_bytes / (1024.0 * 1024.0) / ref_enc_sec;
    result.our_enc_mb_s = our_bytes / (1024.0 * 1024.0) / our_enc_sec;
    result.ref_dec_mb_s = ref_bytes / (1024.0 * 1024.0) / ref_dec_sec;
    result.our_dec_mb_s = our_bytes / (1024.0 * 1024.0) / our_dec_sec;

    return result;
}

// =============================================================================
// Output Formatting Helpers
// =============================================================================

/// Prints the table header for the benchmark results
/// \param bitpack_only If true, print bitpack-specific header
/// \param bitunpack_only If true, print bitunpack-specific header
/// \param bitunpackd1_only If true, print bitunpackd1-specific header
void printTableHeader(bool bitpack_only, bool bitunpack_only, bool bitunpackd1_only)
{
    if (bitpack_only)
    {
        std::printf("  n  | BitWidth | Bitpack (MB/s)\n");
        std::printf("     |          |   Ref      Ours     Diff\n");
        std::printf("-----|----------|--------------------------\n");
    }
    else if (bitunpackd1_only)
    {
        std::printf("  n  | BitWidth | BitunpackD1 (MB/s)\n");
        std::printf("     |          |   Ref      Ours     Diff\n");
        std::printf("-----|----------|--------------------------\n");
    }
    else if (bitunpack_only)
    {
        std::printf("  n  | BitWidth | Bitunpack (MB/s)\n");
        std::printf("     |          |   Ref      Ours     Diff\n");
        std::printf("-----|----------|--------------------------\n");
    }
    else
    {
        std::printf("  n  | BitWidth | Encode (MB/s)             | Decode (MB/s)\n");
        std::printf("     |          |   Ref      Ours     Diff  |   Ref      Ours     Diff\n");
        std::printf("-----|----------|--------------------------|---------------------------\n");
    }
}

/// Prints the separator/footer line for a test group
/// \param bitpack_only If true, print bitpack-specific separator
/// \param bitunpack_only If true, print bitunpack-specific separator
/// \param bitunpackd1_only If true, print bitunpackd1-specific separator
void printTableSeparator(bool bitpack_only, bool bitunpack_only, bool bitunpackd1_only)
{
    if (bitpack_only || bitunpack_only || bitunpackd1_only)
    {
        std::printf("-----|----------|--------------------------\n");
    }
    else
    {
        std::printf("-----|----------|--------------------------|---------------------------\n");
    }
}

/// Prints the average result for a single element count
/// \param n Element count
/// \param avg_diff Average performance difference
/// \param bitpack_only If true, format for bitpack
/// \param bitunpack_only If true, format for bitunpack
/// \param bitunpackd1_only If true, format for bitunpackd1
void printAverageResult(unsigned n, double avg_diff, bool bitpack_only, bool bitunpack_only, bool bitunpackd1_only)
{
    if (bitpack_only || bitunpack_only || bitunpackd1_only)
    {
        std::printf("Avg(%3u) |          |                 %+6.1f%%\n", n, avg_diff);
    }
    else
    {
        std::printf("Avg(%3u) |          |                 %+6.1f%% |                 %+6.1f%%\n", n, avg_diff, avg_diff);
    }
}

/// Generates test scenarios based on command-line arguments
/// \param exc_pct Exception percentage from arguments
/// \param simd128 If true, running SIMD128 tests
/// \param simd256 If true, running SIMD256 tests
/// \return Vector of scenarios to test
std::vector<Scenario> generateScenarios(double exc_pct, bool simd128, bool simd256)
{
    std::vector<Scenario> scenarios;

    if (exc_pct >= 0.0)
    {
        // User provided specific exception percentage
        scenarios.push_back({exc_pct, "Explicit"});
    }
    else
    {
        // Default suite
        scenarios.push_back({-1.0, "Random"});
        // If testing SIMD, include exception cases
        if (simd128 || simd256)
        {
            scenarios.push_back({10.0, "Exc 10%"});
            scenarios.push_back({30.0, "Exc 30%"});
            scenarios.push_back({50.0, "Exc 50%"});
            scenarios.push_back({80.0, "Exc 80%"});
        }
    }

    return scenarios;
}

} // namespace

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char ** argv)
{
    CommandLineArgs args;

    // Parse command-line arguments
    if (!parseArguments(argc, argv, args))
    {
        printUsage(argv[0]);
        return (argc > 1 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) ? 0 : 1;
    }

    // Validate parsed arguments
    if (!args.validate())
    {
        return 1;
    }

    // Configure SIMD mode if requested
    if (args.simd128 || args.simd256)
    {
        if (args.simd128)
        {
            args.n_start = args.n_end = 128;
            std::printf("=== TurboPFor A/B Performance Test - 128v SIMD (n=128) ===\n");
        }
        else
        {
            args.n_start = args.n_end = 256;
            std::printf("=== TurboPFor A/B Performance Test - 256v SIMD (n=256) ===\n");
        }
    }
    else
    {
        // Print test mode based on bit operation flags
        if (args.bitpack_only)
            std::printf("=== TurboPFor A/B Performance Test - bitpack32 ===\n");
        else if (args.bitunpackd1_only)
            std::printf("=== TurboPFor A/B Performance Test - bitd1unpack32 ===\n");
        else if (args.bitunpack_only)
            std::printf("=== TurboPFor A/B Performance Test - bitunpack32 ===\n");
        else
            std::printf("=== TurboPFor A/B Performance Test - p4enc32/p4d1dec32 ===\n");
    }

    // Print test parameters
    std::printf("=== %u iterations x %u runs per bit width ===\n", args.iters, args.runs);
    if (args.simd128 || args.simd256 || args.single_n)
        std::printf("=== Testing n=%u ===\n\n", args.n_start);
    else
        std::printf("=== Testing n=%u to %u ===\n\n", args.n_start, args.n_end);

    // Print table header
    printTableHeader(args.bitpack_only, args.bitunpack_only, args.bitunpackd1_only);

    // Initialize result aggregation
    double grand_total_enc_diff = 0.0;
    double grand_total_dec_diff = 0.0;
    double grand_total_bitpack_diff = 0.0;
    double grand_total_bitunpack_diff = 0.0;
    double grand_total_bitunpackd1_diff = 0.0;
    unsigned total_tests = 0;

    // Generate test scenarios
    std::vector<Scenario> scenarios = generateScenarios(args.exc_pct, args.simd128, args.simd256);

    // Test loop over element counts
    for (unsigned n = args.n_start; n <= args.n_end; ++n)
    {
        // Inner loop over scenarios
        for (const auto & scenario : scenarios)
        {
            double current_exc_pct = scenario.pct;

            // Print scenario header if multiple scenarios
            if (scenarios.size() > 1)
            {
                std::printf("\n--- Scenario: %s (n=%u) ---\n", scenario.desc, n);
                printTableHeader(args.bitpack_only, args.bitunpack_only, args.bitunpackd1_only);
            }

            // Initialize per-scenario aggregation
            double total_enc_diff = 0.0;
            double total_dec_diff = 0.0;
            double total_bitpack_diff = 0.0;
            double total_bitunpack_diff = 0.0;
            double total_bitunpackd1_diff = 0.0;
            unsigned tests_in_scenario = 0;

            // Loop over bit widths
            for (unsigned bw = 1; bw <= 32; ++bw)
            {
                // Skip high bit widths if forcing exceptions
                if (current_exc_pct >= 0.0 && bw > 28)
                    continue;

                // Generate random input data with controlled exception percentage
                std::vector<uint32_t> input(n);
                std::mt19937 rng(42u + bw + n);
                uint32_t max_val = (bw == 32) ? 0xFFFFFFFFu : ((1u << bw) - 1u);
                std::uniform_int_distribution<uint32_t> dist(0u, max_val);
                std::uniform_real_distribution<double> dist_prob(0.0, 100.0);
                std::uniform_int_distribution<uint32_t> dist_exc((1u << bw), 0xFFFFFFFFu);

                if (current_exc_pct >= 0.0)
                {
                    for (auto & v : input)
                    {
                        if (dist_prob(rng) < current_exc_pct)
                        {
                            v = dist_exc(rng);
                        }
                        else
                        {
                            v = dist(rng);
                        }
                    }
                }
                else
                {
                    for (auto & v : input)
                        v = dist(rng);
                }

                // Run appropriate benchmark type
                if (args.bitpack_only)
                {
                    // Bitpack benchmark - run multiple times and take best
                    BitpackResult best{};
                    for (unsigned r = 0; r < args.runs; ++r)
                    {
                        auto result = runBitpackBenchmark(input, bw, args.iters);
                        if (r == 0 || result.ref_mb_s > best.ref_mb_s)
                            best.ref_mb_s = result.ref_mb_s;
                        if (r == 0 || result.our_mb_s > best.our_mb_s)
                            best.our_mb_s = result.our_mb_s;
                    }

                    double diff = (best.our_mb_s / best.ref_mb_s - 1.0) * 100.0;
                    total_bitpack_diff += diff;

                    std::printf(" %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n", n, bw, best.ref_mb_s, best.our_mb_s, diff);
                }
                else if (args.bitunpackd1_only)
                {
                    // BitunpackD1 benchmark - run multiple times and take best
                    BitunpackD1Result best{};
                    for (unsigned r = 0; r < args.runs; ++r)
                    {
                        auto result = runBitunpackD1Benchmark(input, bw, args.iters, 0u);
                        if (r == 0 || result.ref_mb_s > best.ref_mb_s)
                            best.ref_mb_s = result.ref_mb_s;
                        if (r == 0 || result.our_mb_s > best.our_mb_s)
                            best.our_mb_s = result.our_mb_s;
                    }

                    double diff = (best.our_mb_s / best.ref_mb_s - 1.0) * 100.0;
                    total_bitunpackd1_diff += diff;

                    std::printf(" %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n", n, bw, best.ref_mb_s, best.our_mb_s, diff);
                }
                else if (args.bitunpack_only)
                {
                    // Bitunpack benchmark - run multiple times and take best
                    BitunpackResult best{};
                    for (unsigned r = 0; r < args.runs; ++r)
                    {
                        auto result = runBitunpackBenchmark(input, bw, args.iters);
                        if (r == 0 || result.ref_mb_s > best.ref_mb_s)
                            best.ref_mb_s = result.ref_mb_s;
                        if (r == 0 || result.our_mb_s > best.our_mb_s)
                            best.our_mb_s = result.our_mb_s;
                    }

                    double diff = (best.our_mb_s / best.ref_mb_s - 1.0) * 100.0;
                    total_bitunpack_diff += diff;

                    std::printf(" %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n", n, bw, best.ref_mb_s, best.our_mb_s, diff);
                }
                else
                {
                    // P4 benchmark - run multiple times and take best
                    BenchResult best{};
                    for (unsigned r = 0; r < args.runs; ++r)
                    {
                        auto result = runBenchmark(input, args.iters, args.simd128, args.simd256);
                        if (r == 0 || result.ref_enc_mb_s > best.ref_enc_mb_s)
                            best.ref_enc_mb_s = result.ref_enc_mb_s;
                        if (r == 0 || result.our_enc_mb_s > best.our_enc_mb_s)
                            best.our_enc_mb_s = result.our_enc_mb_s;
                        if (r == 0 || result.ref_dec_mb_s > best.ref_dec_mb_s)
                            best.ref_dec_mb_s = result.ref_dec_mb_s;
                        if (r == 0 || result.our_dec_mb_s > best.our_dec_mb_s)
                            best.our_dec_mb_s = result.our_dec_mb_s;
                    }

                    double enc_diff = (best.our_enc_mb_s / best.ref_enc_mb_s - 1.0) * 100.0;
                    double dec_diff = (best.our_dec_mb_s / best.ref_dec_mb_s - 1.0) * 100.0;

                    total_enc_diff += enc_diff;
                    total_dec_diff += dec_diff;

                    std::printf(
                        " %3u |   %2u     | %6.1f   %6.1f   %+6.1f%% | %6.1f   %6.1f   %+6.1f%%\n",
                        n,
                        bw,
                        best.ref_enc_mb_s,
                        best.our_enc_mb_s,
                        enc_diff,
                        best.ref_dec_mb_s,
                        best.our_dec_mb_s,
                        dec_diff);
                }

                tests_in_scenario++;
            }

            // Print per-scenario summary if we have results
            if (tests_in_scenario > 0)
            {
                grand_total_enc_diff += total_enc_diff;
                grand_total_dec_diff += total_dec_diff;
                grand_total_bitpack_diff += total_bitpack_diff;
                grand_total_bitunpack_diff += total_bitunpack_diff;
                grand_total_bitunpackd1_diff += total_bitunpackd1_diff;
                total_tests += tests_in_scenario;

                printTableSeparator(args.bitpack_only, args.bitunpack_only, args.bitunpackd1_only);

                if (scenarios.size() > 1)
                {
                    // Multiple scenarios - print per-scenario average
                    if (!args.bitpack_only && !args.bitunpack_only && !args.bitunpackd1_only)
                    {
                        std::printf(
                            "Avg  |          |                 %+6.1f%% |                 %+6.1f%%\n",
                            total_enc_diff / tests_in_scenario,
                            total_dec_diff / tests_in_scenario);
                    }
                    printTableSeparator(args.bitpack_only, args.bitunpack_only, args.bitunpackd1_only);
                }
                else
                {
                    // Single scenario - print per-n average
                    if (args.bitpack_only)
                    {
                        std::printf("Avg(%3u) |          |                 %+6.1f%%\n", n, total_bitpack_diff / tests_in_scenario);
                    }
                    else if (args.bitunpackd1_only)
                    {
                        std::printf("Avg(%3u) |          |                 %+6.1f%%\n", n, total_bitunpackd1_diff / tests_in_scenario);
                    }
                    else if (args.bitunpack_only)
                    {
                        std::printf("Avg(%3u) |          |                 %+6.1f%%\n", n, total_bitunpack_diff / tests_in_scenario);
                    }
                    else
                    {
                        std::printf(
                            "Avg(%3u) |          |                 %+6.1f%% |                 %+6.1f%%\n",
                            n,
                            total_enc_diff / tests_in_scenario,
                            total_dec_diff / tests_in_scenario);
                    }
                    printTableSeparator(args.bitpack_only, args.bitunpack_only, args.bitunpackd1_only);
                }
            }
        }
    }

    // Print grand summary if testing multiple element counts or SIMD
    if (args.n_end > args.n_start || args.simd128 || args.simd256)
    {
        if (args.bitpack_only)
        {
            std::printf("Grand Avg|          |                 %+6.1f%%\n", grand_total_bitpack_diff / total_tests);
            std::printf("\n=== Summary ===\n");
            std::printf("Bitpack average diff: %+.2f%%\n", grand_total_bitpack_diff / total_tests);
        }
        else if (args.bitunpackd1_only)
        {
            std::printf("Grand Avg|          |                 %+6.1f%%\n", grand_total_bitunpackd1_diff / total_tests);
            std::printf("\n=== Summary ===\n");
            std::printf("BitunpackD1 average diff: %+.2f%%\n", grand_total_bitunpackd1_diff / total_tests);
        }
        else if (args.bitunpack_only)
        {
            std::printf("Grand Avg|          |                 %+6.1f%%\n", grand_total_bitunpack_diff / total_tests);
            std::printf("\n=== Summary ===\n");
            std::printf("Bitunpack average diff: %+.2f%%\n", grand_total_bitunpack_diff / total_tests);
        }
        else
        {
            std::printf(
                "Grand Avg|          |                 %+6.1f%% |                 %+6.1f%%\n",
                grand_total_enc_diff / total_tests,
                grand_total_dec_diff / total_tests);
            std::printf("\n=== Summary ===\n");
            std::printf("Encode average diff: %+.2f%%\n", grand_total_enc_diff / total_tests);
            std::printf("Decode average diff: %+.2f%%\n", grand_total_dec_diff / total_tests);
        }
    }

    return 0;
}
