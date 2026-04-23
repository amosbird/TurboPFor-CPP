/// TurboPFor A/B Benchmark Suite
/// Benchmarks encode/decode operations and bit-packing operations
/// Supports scalar and SIMD implementations with various input patterns

#include "turbopfor.h"
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
extern "C" unsigned char * p4dec32(unsigned char * in, unsigned n, uint32_t * out);
extern "C" unsigned char * p4d1dec32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * bitpack32(unsigned * in, unsigned n, unsigned char * out, unsigned b);
extern "C" unsigned char * bitunpack32(const unsigned char * in, unsigned n, uint32_t * out, unsigned b);
extern "C" unsigned char * bitd1unpack32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);

// SIMD Reference declarations
extern "C" unsigned char * p4enc128v32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4dec128v32(unsigned char * in, unsigned n, uint32_t * out);
extern "C" unsigned char * p4d1dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * p4enc256v32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4dec256v32(unsigned char * in, unsigned n, uint32_t * out);
extern "C" unsigned char * p4d1dec256v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

// D1 encode reference
extern "C" unsigned char * p4d1enc32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start);
extern "C" unsigned char * p4d1enc128v32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start);
extern "C" unsigned char * p4d1enc256v32(uint32_t * in, unsigned n, unsigned char * out, uint32_t start);
extern "C" unsigned char * p4d1enc64(uint64_t * in, unsigned n, unsigned char * out, uint64_t start);

// 64-bit Reference C implementations
extern "C" unsigned char * p4enc64(uint64_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4dec64(unsigned char * in, unsigned n, uint64_t * out);
extern "C" unsigned char * p4d1dec64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start);
extern "C" unsigned char * bitpack64(uint64_t * in, unsigned n, unsigned char * out, unsigned b);
extern "C" unsigned char * bitunpack64(const unsigned char * in, unsigned n, uint64_t * out, unsigned b);
extern "C" unsigned char * bitd1unpack64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b);
extern "C" unsigned char * p4enc128v64(uint64_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4dec128v64(unsigned char * in, unsigned n, uint64_t * out);

extern "C" void bitd1dec64(uint64_t * p, unsigned n, uint64_t start);

namespace turbopfor::scalar::detail
{
unsigned char * bitpack32Scalar(const uint32_t * in, unsigned n, unsigned char * out, unsigned b);
const unsigned char * bitunpack32Scalar(const unsigned char * in, unsigned n, uint32_t * out, unsigned b);
const unsigned char * bitunpackd1_32Scalar(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);
unsigned char * bitpack64Scalar(const uint64_t * in, unsigned n, unsigned char * out, unsigned b);
const unsigned char * bitunpack64Scalar(const unsigned char * in, unsigned n, uint64_t * out, unsigned b);
const unsigned char * bitunpackd1_64Scalar(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b);
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
    unsigned bw_min = 0; ///< Minimum bit width to test (0 = default)
    unsigned bw_max = 0; ///< Maximum bit width to test (0 = default)
    bool single_n = false; ///< Test only a single n value
    bool bitpack_only = false; ///< Test bitpack32 only
    bool bitunpack_only = false; ///< Test bitunpack32 only
    bool bitunpackd1_only = false; ///< Test bitunpackd1_32 only
    bool simd128 = false; ///< Test 128-bit SIMD variant
    bool simd256 = false; ///< Test 256-bit SIMD variant
    bool p64 = false; ///< Test 64-bit scalar p4enc64/p4d1dec64
    bool bitpack64_only = false; ///< Test bitpack64 only
    bool bitunpack64_only = false; ///< Test bitunpack64 only
    bool bitunpackd1_64_only = false; ///< Test bitunpackd1_64 only
    bool simd128v64 = false; ///< Test 128v64 variant (non-delta decode)
    bool simd128v64d1 = false; ///< Test 128v64 variant (delta1 decode)
    bool simd256v64d1 = false; ///< Test 256v64 variant (delta1 decode)
    bool p4dec = false; ///< Test non-delta p4dec32 (n=1..127)
    bool p4dec64 = false; ///< Test non-delta p4dec64 (n=1..127)
    bool simd128dec = false; ///< Test non-delta 128v SIMD decode (n=128)
    bool simd256dec = false; ///< Test non-delta 256v SIMD decode (n=256)
    bool d1enc = false; ///< Test D1 encode (p4d1enc32/128v/256v)

    /// Validates argument consistency and prints errors if invalid
    bool validate() const
    {
        // Check SIMD exclusivity
        unsigned simd_modes
            = (simd128 ? 1u : 0u) + (simd256 ? 1u : 0u) + (simd128v64 ? 1u : 0u) + (simd128v64d1 ? 1u : 0u) + (simd256v64d1 ? 1u : 0u)
            + (simd128dec ? 1u : 0u) + (simd256dec ? 1u : 0u);
        if (simd_modes > 1)
        {
            std::fprintf(stderr, "Error: Cannot run multiple SIMD modes at the same time\n");
            return false;
        }

        // Check 64-bit mode exclusivity with 32-bit bitpack tests
        if ((p64 || p4dec64 || bitpack64_only || bitunpack64_only || bitunpackd1_64_only || simd128v64 || simd128v64d1 || simd256v64d1)
            && (bitpack_only || bitunpack_only || bitunpackd1_only || simd128 || simd256 || p4dec || simd128dec || simd256dec))
        {
            std::fprintf(stderr, "Error: 64-bit tests cannot be combined with 32-bit tests\n");
            return false;
        }

        // Check SIMD and bitpack test incompatibility
        if ((simd128 || simd256 || simd128v64 || simd128v64d1 || simd256v64d1 || simd128dec || simd256dec)
            && (bitpack_only || bitunpack_only || bitunpackd1_only || bitpack64_only || bitunpack64_only || bitunpackd1_64_only))
        {
            std::fprintf(stderr, "Error: SIMD tests cannot be combined with bitpack/unpack tests\n");
            return false;
        }

        // Check element count range (only for non-SIMD tests)
        if (!simd128 && !simd256 && !simd128v64 && !simd128v64d1 && !simd256v64d1 && !simd128dec && !simd256dec && (n_start < 1 || n_end > 127 || n_start > n_end))
        {
            std::fprintf(stderr, "Error: n must be in range [1, 127] and start <= end\n");
            return false;
        }

        // Check bit operation test exclusivity
        unsigned bit_tests = (bitpack_only ? 1u : 0u) + (bitunpack_only ? 1u : 0u) + (bitunpackd1_only ? 1u : 0u)
            + (bitpack64_only ? 1u : 0u) + (bitunpack64_only ? 1u : 0u) + (bitunpackd1_64_only ? 1u : 0u);
        if (bit_tests > 1)
        {
            std::fprintf(stderr, "Error: bitpack/bitunpack options are mutually exclusive\n");
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
        else if (std::strcmp(argv[i], "--p64") == 0)
        {
            args.p64 = true;
        }
        else if (std::strcmp(argv[i], "--bitpack64") == 0)
        {
            args.bitpack64_only = true;
        }
        else if (std::strcmp(argv[i], "--bitunpack64") == 0)
        {
            args.bitunpack64_only = true;
        }
        else if (std::strcmp(argv[i], "--bitunpackd1_64") == 0)
        {
            args.bitunpackd1_64_only = true;
        }
        else if (std::strcmp(argv[i], "--simd128v64") == 0)
        {
            args.simd128v64 = true;
        }
        else if (std::strcmp(argv[i], "--simd128v64d1") == 0)
        {
            args.simd128v64d1 = true;
        }
        else if (std::strcmp(argv[i], "--simd256v64d1") == 0)
        {
            args.simd256v64d1 = true;
        }
        else if (std::strcmp(argv[i], "--p4dec") == 0)
        {
            args.p4dec = true;
        }
        else if (std::strcmp(argv[i], "--p4dec64") == 0)
        {
            args.p4dec64 = true;
            args.p64 = true;
        }
        else if (std::strcmp(argv[i], "--simd128dec") == 0)
        {
            args.simd128dec = true;
        }
        else if (std::strcmp(argv[i], "--simd256dec") == 0)
        {
            args.simd256dec = true;
        }
        else if (std::strcmp(argv[i], "--d1enc") == 0)
        {
            args.d1enc = true;
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
        else if (std::strcmp(argv[i], "--bw") == 0 && i + 1 < argc)
        {
            unsigned bw = static_cast<unsigned>(std::atoi(argv[++i]));
            args.bw_min = bw;
            args.bw_max = bw;
        }
        else if (std::strcmp(argv[i], "--bw-range") == 0 && i + 1 < argc)
        {
            char * arg = argv[++i];
            char * dash = std::strchr(arg, '-');
            if (dash)
            {
                *dash = '\0';
                args.bw_min = static_cast<unsigned>(std::atoi(arg));
                args.bw_max = static_cast<unsigned>(std::atoi(dash + 1));
            }
            else
            {
                std::fprintf(stderr, "Error: --bw-range requires format <start>-<end>\n");
                return false;
            }
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
    std::printf("  --p64              Test 64-bit p4enc64/p4d1dec64 (n=1..127)\n");
    std::printf("  --bitpack64        Benchmark bitpack64 vs bitpack64Scalar\n");
    std::printf("  --bitunpack64      Benchmark bitunpack64 vs bitunpack64Scalar\n");
    std::printf("  --bitunpackd1_64   Benchmark bitd1unpack64 vs bitunpackd1_64Scalar\n");
    std::printf("  --simd128v64       Test 128v64 non-delta (n=128)\n");
    std::printf("  --simd128v64d1     Test 128v64 delta1 decode (n=128)\n");
    std::printf("  --simd256v64d1     Test 256v64 delta1 decode (n=256)\n");
    std::printf("  --iters <count>    Number of iterations (default: 100000)\n");
    std::printf("  --runs <count>     Number of runs per test (default: 3)\n");
    std::printf("  --exc-pct <pct>    Force percentage of exceptions (values > 2^bw)\n");
    std::printf("  --bw <value>       Test only a specific bit width\n");
    std::printf("  --bw-range <s>-<e> Test a range of bit widths\n");
    std::printf("Examples:\n");
    std::printf("  %s --n 32              # Test 32-bit with 32 elements\n", prog);
    std::printf("  %s --p64               # Test 64-bit scalar, n=1..127\n", prog);
    std::printf("  %s --simd128v64        # Test 128v64 non-delta (n=128)\n", prog);
    std::printf("  %s --simd128v64d1      # Test 128v64 delta1 (n=128)\n", prog);
    std::printf("  %s --simd256v64d1      # Test 256v64 delta1 (n=256)\n", prog);
    std::printf("  %s --bitpack64 --n 32  # Test bitpack64 with 32 elements\n", prog);
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

/// Benchmarks non-delta P4 decode: p4enc*/p4dec* (no delta1)
BenchResult runBenchmarkNoDelta(const std::vector<uint32_t> & input, unsigned iters, bool simd128, bool simd256)
{
    const unsigned num_elements = static_cast<unsigned>(input.size());

    auto get_aligned_ptr = [](std::vector<unsigned char> & buf) -> unsigned char *
    {
        unsigned char * ptr = buf.data();
        size_t remainder = reinterpret_cast<uintptr_t>(ptr) % 32;
        if (remainder)
            ptr += (32 - remainder);
        return ptr;
    };

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

    std::vector<uint32_t> input_copy = input;
    input_copy.resize(num_elements + 64, 0u);

    std::vector<unsigned char> ref_buf_vec(num_elements * 5 + 512, 0u);
    unsigned char * ref_buf = get_aligned_ptr(ref_buf_vec);

    std::vector<unsigned char> our_buf_vec(num_elements * 5 + 512, 0u);
    unsigned char * our_buf = get_aligned_ptr(our_buf_vec);

    std::vector<uint32_t> out_vec(num_elements + 128, 0u);
    uint32_t * out = get_aligned_u32_ptr(out_vec);

    for (unsigned i = 0; i < 1000; ++i)
    {
        if (simd128)
        {
            ::p4enc128v32(input_copy.data(), num_elements, ref_buf);
            turbopfor::simd::p4Enc128v32(input_copy.data(), num_elements, our_buf);
            ::p4dec128v32(ref_buf, num_elements, out);
            turbopfor::simd::p4Dec128v32(our_buf, num_elements, out);
        }
        else if (simd256)
        {
            ::p4enc256v32(input_copy.data(), num_elements, ref_buf);
            turbopfor::simd::p4Enc256v32(input_copy.data(), num_elements, our_buf);
            ::p4dec256v32(ref_buf, num_elements, out);
            turbopfor::simd::p4Dec256v32(our_buf, num_elements, out);
        }
        else
        {
            ::p4enc32(input_copy.data(), num_elements, ref_buf);
            turbopfor::scalar::p4Enc32(input_copy.data(), num_elements, our_buf);
            ::p4dec32(ref_buf, num_elements, out);
            turbopfor::scalar::p4Dec32(our_buf, num_elements, out);
        }
    }

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

    double ref_dec_sec = 0.0;
    double our_dec_sec = 0.0;

    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            if (simd128)
                ::p4dec128v32(ref_buf, num_elements, out);
            else if (simd256)
                ::p4dec256v32(ref_buf, num_elements, out);
            else
                ::p4dec32(ref_buf, num_elements, out);
        }
        ref_dec_sec += secondsSince(start);

        start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            if (simd128)
                turbopfor::simd::p4Dec128v32(our_buf, num_elements, out);
            else if (simd256)
                turbopfor::simd::p4Dec256v32(our_buf, num_elements, out);
            else
                turbopfor::scalar::p4Dec32(our_buf, num_elements, out);
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
/// \param single_column If true, print single throughput column (bitpack/unpack tests)
void printTableHeader(bool single_column, bool /*unused1*/ = false, bool /*unused2*/ = false)
{
    if (single_column)
    {
        std::printf("  n  | BitWidth | Throughput (MB/s)\n");
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
/// \param single_column If true, print single throughput column separator
void printTableSeparator(bool single_column, bool /*unused1*/ = false, bool /*unused2*/ = false)
{
    if (single_column)
    {
        std::printf("-----|----------|--------------------------\n");
    }
    else
    {
        std::printf("-----|----------|--------------------------|---------------------------\n");
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

// =============================================================================
// 64-bit Benchmark Functions
// =============================================================================

/// Benchmarks bitpack64 (reference) vs bitpack64Scalar (ours)
BitpackResult runBitpack64Benchmark(const std::vector<uint64_t> & input, unsigned bit_width, unsigned iters)
{
    const unsigned num_elements = static_cast<unsigned>(input.size());
    std::vector<unsigned char> ref_buf(num_elements * 8u + 64u, 0u);
    std::vector<unsigned char> our_buf(num_elements * 8u + 64u, 0u);

    for (unsigned i = 0; i < 1000; ++i)
    {
        ::bitpack64(const_cast<uint64_t *>(input.data()), num_elements, ref_buf.data(), bit_width);
        turbopfor::scalar::detail::bitpack64Scalar(input.data(), num_elements, our_buf.data(), bit_width);
    }

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
            unsigned char * end = ::bitpack64(const_cast<uint64_t *>(input.data()), num_elements, ref_buf.data(), bit_width);
            ref_bytes += static_cast<size_t>(end - ref_buf.data());
        }
        ref_sec += secondsSince(start);

        start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            unsigned char * end = turbopfor::scalar::detail::bitpack64Scalar(input.data(), num_elements, our_buf.data(), bit_width);
            our_bytes += static_cast<size_t>(end - our_buf.data());
        }
        our_sec += secondsSince(start);
    }

    BitpackResult result;
    result.ref_mb_s = ref_bytes / (1024.0 * 1024.0) / ref_sec;
    result.our_mb_s = our_bytes / (1024.0 * 1024.0) / our_sec;
    return result;
}

/// Benchmarks bitunpack64 (reference) vs bitunpack64Scalar (ours)
BitunpackResult runBitunpack64Benchmark(const std::vector<uint64_t> & input, unsigned bit_width, unsigned iters)
{
    const unsigned num_elements = static_cast<unsigned>(input.size());
    std::vector<unsigned char> buf(num_elements * 8u + 64u, 0u);
    std::vector<uint64_t> out(num_elements, 0ull);

    unsigned char * packed_end = ::bitpack64(const_cast<uint64_t *>(input.data()), num_elements, buf.data(), bit_width);
    const size_t packed_bytes = static_cast<size_t>(packed_end - buf.data());

    for (unsigned i = 0; i < 1000; ++i)
    {
        ::bitunpack64(buf.data(), num_elements, out.data(), bit_width);
        turbopfor::scalar::detail::bitunpack64Scalar(buf.data(), num_elements, out.data(), bit_width);
    }

    double ref_sec = 0.0;
    double our_sec = 0.0;
    size_t total_bytes = 0;

    const unsigned chunk = 10000;
    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
            ::bitunpack64(buf.data(), num_elements, out.data(), bit_width);
        ref_sec += secondsSince(start);

        start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
            turbopfor::scalar::detail::bitunpack64Scalar(buf.data(), num_elements, out.data(), bit_width);
        our_sec += secondsSince(start);

        total_bytes += packed_bytes * count;
    }

    BitunpackResult result;
    result.ref_mb_s = total_bytes / (1024.0 * 1024.0) / ref_sec;
    result.our_mb_s = total_bytes / (1024.0 * 1024.0) / our_sec;
    return result;
}

/// Benchmarks bitd1unpack64 (reference) vs bitunpackd1_64Scalar (ours)
BitunpackD1Result runBitunpackD1_64Benchmark(const std::vector<uint64_t> & input, unsigned bit_width, unsigned iters, uint64_t start_val)
{
    const unsigned num_elements = static_cast<unsigned>(input.size());
    std::vector<unsigned char> buf(num_elements * 8u + 64u, 0u);
    std::vector<uint64_t> out(num_elements, 0ull);

    unsigned char * packed_end = ::bitpack64(const_cast<uint64_t *>(input.data()), num_elements, buf.data(), bit_width);
    const size_t packed_bytes = static_cast<size_t>(packed_end - buf.data());

    for (unsigned i = 0; i < 1000; ++i)
    {
        ::bitd1unpack64(buf.data(), num_elements, out.data(), start_val, bit_width);
        turbopfor::scalar::detail::bitunpackd1_64Scalar(buf.data(), num_elements, out.data(), start_val, bit_width);
    }

    double ref_sec = 0.0;
    double our_sec = 0.0;
    size_t total_bytes = 0;

    const unsigned chunk = 10000;
    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto start_time = Clock::now();
        for (unsigned i = 0; i < count; ++i)
            ::bitd1unpack64(buf.data(), num_elements, out.data(), start_val, bit_width);
        ref_sec += secondsSince(start_time);

        start_time = Clock::now();
        for (unsigned i = 0; i < count; ++i)
            turbopfor::scalar::detail::bitunpackd1_64Scalar(buf.data(), num_elements, out.data(), start_val, bit_width);
        our_sec += secondsSince(start_time);

        total_bytes += packed_bytes * count;
    }

    BitunpackD1Result result;
    result.ref_mb_s = total_bytes / (1024.0 * 1024.0) / ref_sec;
    result.our_mb_s = total_bytes / (1024.0 * 1024.0) / our_sec;
    return result;
}

/// Benchmarks 64-bit p4enc/p4d1dec (scalar, 128v64, or 256v64)
BenchResult runBenchmark64(const std::vector<uint64_t> & input, unsigned iters, bool simd128v64, bool simd128v64d1 = false, bool simd256v64d1 = false, bool p4dec64 = false)
{
    const unsigned num_elements = static_cast<unsigned>(input.size());

    auto get_aligned_ptr = [](std::vector<unsigned char> & buf) -> unsigned char *
    {
        unsigned char * ptr = buf.data();
        size_t remainder = reinterpret_cast<uintptr_t>(ptr) % 32;
        if (remainder)
            ptr += (32 - remainder);
        return ptr;
    };

    auto get_aligned_u64_ptr = [](std::vector<uint64_t> & buf) -> uint64_t *
    {
        uint64_t * ptr = buf.data();
        size_t remainder = reinterpret_cast<uintptr_t>(ptr) % 32;
        if (remainder)
        {
            size_t bytes_needed = 32 - remainder;
            ptr += bytes_needed / 8;
        }
        return ptr;
    };

    std::vector<uint64_t> input_copy = input;
    input_copy.resize(num_elements + 64, 0ull);

    std::vector<unsigned char> ref_buf_vec(num_elements * 10 + 512, 0u);
    unsigned char * ref_buf = get_aligned_ptr(ref_buf_vec);

    std::vector<unsigned char> our_buf_vec(num_elements * 10 + 512, 0u);
    unsigned char * our_buf = get_aligned_ptr(our_buf_vec);

    std::vector<uint64_t> out_vec(num_elements + 128, 0ull);
    uint64_t * out = get_aligned_u64_ptr(out_vec);

    // Warmup
    for (unsigned i = 0; i < 1000; ++i)
    {
        if (simd256v64d1)
        {
            turbopfor::p4Enc256v64(input_copy.data(), num_elements, our_buf);
            turbopfor::p4D1Dec256v64(our_buf, num_elements, out, 0ull);
            {
                unsigned char * rp = ref_buf;
                for (unsigned off = 0; off < num_elements; off += 128)
                    rp = ::p4enc128v64(input_copy.data() + off, 128, rp);
                rp = ref_buf;
                for (unsigned off = 0; off < num_elements; off += 128)
                {
                    rp = ::p4dec128v64(rp, 128, out + off);
                    ::bitd1dec64(out + off, 128, 0ull);
                }
            }
        }
        else if (simd128v64 || simd128v64d1)
        {
            ::p4enc128v64(input_copy.data(), num_elements, ref_buf);
            turbopfor::simd::p4Enc128v64(input_copy.data(), num_elements, our_buf);
            if (simd128v64d1)
            {
                // Delta1 decode: C ref = p4dec128v64 + bitd1dec64, Ours = p4D1Dec128v64
                ::p4dec128v64(ref_buf, num_elements, out);
                ::bitd1dec64(out, num_elements, 0ull);
                turbopfor::simd::p4D1Dec128v64(our_buf, num_elements, out, 0ull);
            }
            else
            {
                // Non-delta decode
                ::p4dec128v64(ref_buf, num_elements, out);
                turbopfor::simd::p4Dec128v64(our_buf, num_elements, out);
            }
        }
        else
        {
            ::p4enc64(input_copy.data(), num_elements, ref_buf);
            turbopfor::scalar::p4Enc64(input_copy.data(), num_elements, our_buf);
            if (p4dec64)
            {
                ::p4dec64(ref_buf, num_elements, out);
                turbopfor::scalar::p4Dec64(our_buf, num_elements, out);
            }
            else
            {
                ::p4d1dec64(ref_buf, num_elements, out, 0ull);
                turbopfor::scalar::p4D1Dec64(our_buf, num_elements, out, 0ull);
            }
        }
    }

    // Encode benchmark
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
            unsigned char * end;
            if (simd256v64d1)
            {
                end = ref_buf;
                for (unsigned off = 0; off < num_elements; off += 128)
                    end = ::p4enc128v64(input_copy.data() + off, 128, end);
            }
            else if (simd128v64 || simd128v64d1)
                end = ::p4enc128v64(input_copy.data(), num_elements, ref_buf);
            else
                end = ::p4enc64(input_copy.data(), num_elements, ref_buf);
            ref_bytes += static_cast<size_t>(end - ref_buf);
        }
        ref_enc_sec += secondsSince(start);

        start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            unsigned char * end;
            if (simd256v64d1)
                end = turbopfor::p4Enc256v64(input_copy.data(), num_elements, our_buf);
            else if (simd128v64 || simd128v64d1)
                end = turbopfor::simd::p4Enc128v64(input_copy.data(), num_elements, our_buf);
            else
                end = turbopfor::scalar::p4Enc64(input_copy.data(), num_elements, our_buf);
            our_bytes += static_cast<size_t>(end - our_buf);
        }
        our_enc_sec += secondsSince(start);
    }

    // Decode benchmark
    double ref_dec_sec = 0.0;
    double our_dec_sec = 0.0;

    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            if (simd256v64d1)
            {
                unsigned char * rp = ref_buf;
                for (unsigned off = 0; off < num_elements; off += 128)
                {
                    rp = ::p4dec128v64(rp, 128, out + off);
                    ::bitd1dec64(out + off, 128, 0ull);
                }
            }
            else if (simd128v64d1)
            {
                ::p4dec128v64(ref_buf, num_elements, out);
                ::bitd1dec64(out, num_elements, 0ull);
            }
            else if (simd128v64)
                ::p4dec128v64(ref_buf, num_elements, out);
            else if (p4dec64)
                ::p4dec64(ref_buf, num_elements, out);
            else
                ::p4d1dec64(ref_buf, num_elements, out, 0ull);
        }
        ref_dec_sec += secondsSince(start);

        start = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            if (simd256v64d1)
                turbopfor::p4D1Dec256v64(our_buf, num_elements, out, 0ull);
            else if (simd128v64d1)
                turbopfor::simd::p4D1Dec128v64(our_buf, num_elements, out, 0ull);
            else if (simd128v64)
                turbopfor::simd::p4Dec128v64(our_buf, num_elements, out);
            else if (p4dec64)
                turbopfor::scalar::p4Dec64(our_buf, num_elements, out);
            else
                turbopfor::scalar::p4D1Dec64(our_buf, num_elements, out, 0ull);
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

/// Benchmarks D1 encode: p4d1enc* (delta-1 encode)
/// Compares C reference vs C++ implementation for sorted input data
BenchResult runD1EncBenchmark(const std::vector<uint32_t> & sorted_input, unsigned iters, bool simd128, bool simd256)
{
    const unsigned num_elements = static_cast<unsigned>(sorted_input.size());

    auto get_aligned_ptr = [](std::vector<unsigned char> & buf) -> unsigned char *
    {
        unsigned char * ptr = buf.data();
        size_t remainder = reinterpret_cast<uintptr_t>(ptr) % 32;
        if (remainder)
            ptr += (32 - remainder);
        return ptr;
    };

    std::vector<uint32_t> input_copy = sorted_input;
    input_copy.resize(num_elements + 64, 0u);
    // Note: p4d1enc/p4D1Enc do NOT modify the input array (delta-1 output goes to a temp buffer).
    // Input is copied once before timing to ensure alignment/padding.
    std::copy(sorted_input.begin(), sorted_input.end(), input_copy.begin());

    std::vector<unsigned char> ref_buf_vec(num_elements * 5 + 512, 0u);
    unsigned char * ref_buf = get_aligned_ptr(ref_buf_vec);

    std::vector<unsigned char> our_buf_vec(num_elements * 5 + 512, 0u);
    unsigned char * our_buf = get_aligned_ptr(our_buf_vec);

    uint32_t start = 0u;

    for (unsigned i = 0; i < 1000; ++i)
    {
        if (simd128)
            ::p4d1enc128v32(input_copy.data(), num_elements, ref_buf, start);
        else if (simd256)
            ::p4d1enc256v32(input_copy.data(), num_elements, ref_buf, start);
        else
            ::p4d1enc32(input_copy.data(), num_elements, ref_buf, start);

        if (simd128)
            turbopfor::p4D1Enc128v32(input_copy.data(), num_elements, our_buf, start);
        else if (simd256)
            turbopfor::p4D1Enc256v32(input_copy.data(), num_elements, our_buf, start);
        else
            turbopfor::p4D1Enc32(input_copy.data(), num_elements, our_buf, start);
    }

    double ref_enc_sec = 0.0;
    double our_enc_sec = 0.0;
    size_t ref_bytes = 0;
    size_t our_bytes = 0;

    const unsigned chunk = 10000;
    for (unsigned base = 0; base < iters; base += chunk)
    {
        unsigned count = std::min(chunk, iters - base);

        auto t0 = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            unsigned char * end = nullptr;
            if (simd128)
                end = ::p4d1enc128v32(input_copy.data(), num_elements, ref_buf, start);
            else if (simd256)
                end = ::p4d1enc256v32(input_copy.data(), num_elements, ref_buf, start);
            else
                end = ::p4d1enc32(input_copy.data(), num_elements, ref_buf, start);
            ref_bytes += static_cast<size_t>(end - ref_buf);
        }
        ref_enc_sec += secondsSince(t0);

        t0 = Clock::now();
        for (unsigned i = 0; i < count; ++i)
        {
            unsigned char * end = nullptr;
            if (simd128)
                end = turbopfor::p4D1Enc128v32(input_copy.data(), num_elements, our_buf, start);
            else if (simd256)
                end = turbopfor::p4D1Enc256v32(input_copy.data(), num_elements, our_buf, start);
            else
                end = turbopfor::p4D1Enc32(input_copy.data(), num_elements, our_buf, start);
            our_bytes += static_cast<size_t>(end - our_buf);
        }
        our_enc_sec += secondsSince(t0);
    }

    BenchResult result;
    result.ref_enc_mb_s = ref_bytes / (1024.0 * 1024.0) / ref_enc_sec;
    result.our_enc_mb_s = our_bytes / (1024.0 * 1024.0) / our_enc_sec;
    result.ref_dec_mb_s = 0.0;
    result.our_dec_mb_s = 0.0;
    return result;
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

    // Detect 64-bit mode
    bool is_64bit = args.p64 || args.bitpack64_only || args.bitunpack64_only || args.bitunpackd1_64_only || args.simd128v64
        || args.simd128v64d1 || args.simd256v64d1;

    // Determine if this is a bitpack-only style test (single throughput column)
    bool is_bitop_only = args.bitpack_only || args.bitunpack_only || args.bitunpackd1_only || args.bitpack64_only || args.bitunpack64_only
        || args.bitunpackd1_64_only || args.d1enc;

    // Configure SIMD/128v mode if requested
    if (args.simd128 || args.simd256 || args.simd128v64 || args.simd128v64d1 || args.simd256v64d1 || args.simd128dec || args.simd256dec)
    {
        if (args.simd128)
        {
            args.n_start = args.n_end = 128;
            std::printf("=== TurboPFor A/B Performance Test - 128v SIMD (n=128) ===\n");
        }
        else if (args.simd256)
        {
            args.n_start = args.n_end = 256;
            std::printf("=== TurboPFor A/B Performance Test - 256v SIMD (n=256) ===\n");
        }
        else if (args.simd128dec)
        {
            args.n_start = args.n_end = 128;
            std::printf("=== TurboPFor A/B Performance Test - 128v SIMD p4dec (no delta) (n=128) ===\n");
        }
        else if (args.simd256dec)
        {
            args.n_start = args.n_end = 256;
            std::printf("=== TurboPFor A/B Performance Test - 256v SIMD p4dec (no delta) (n=256) ===\n");
        }
        else if (args.simd128v64d1)
        {
            args.n_start = args.n_end = 128;
            std::printf("=== TurboPFor A/B Performance Test - 128v64 Delta1 (n=128) ===\n");
            std::printf("=== C ref: p4enc128v64/p4d1dec128v64, Ours: simd::p4Enc128v64/simd::p4D1Dec128v64 ===\n");
        }
        else if (args.simd256v64d1)
        {
            args.n_start = args.n_end = 256;
            std::printf("=== TurboPFor A/B Performance Test - 256v64 Delta1 (n=256) ===\n");
            std::printf("=== C ref: p4enc64/p4d1dec64, Ours: p4Enc256v64/p4D1Dec256v64 ===\n");
        }
        else
        {
            args.n_start = args.n_end = 128;
            std::printf("=== TurboPFor A/B Performance Test - 128v64 (n=128) ===\n");
            std::printf("=== C ref: p4enc128v64/p4dec128v64, Ours: simd::p4Enc128v64/simd::p4Dec128v64 ===\n");
        }
    }
    else
    {
        // Print test mode based on flags
        if (args.bitpack64_only)
            std::printf("=== TurboPFor A/B Performance Test - bitpack64 ===\n");
        else if (args.bitunpack64_only)
            std::printf("=== TurboPFor A/B Performance Test - bitunpack64 ===\n");
        else if (args.bitunpackd1_64_only)
            std::printf("=== TurboPFor A/B Performance Test - bitd1unpack64 ===\n");
        else if (args.p64)
            std::printf("=== TurboPFor A/B Performance Test - p4enc64/p4d1dec64 ===\n");
        else if (args.p4dec)
            std::printf("=== TurboPFor A/B Performance Test - p4enc32/p4dec32 (no delta) ===\n");
        else if (args.d1enc)
            std::printf("=== TurboPFor A/B Performance Test - p4d1enc32 (delta1 encode) ===\n");
        else if (args.bitpack_only)
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
    if (args.simd128 || args.simd256 || args.simd128v64 || args.simd128v64d1 || args.simd256v64d1 || args.simd128dec || args.simd256dec || args.single_n)
        std::printf("=== Testing n=%u ===\n\n", args.n_start);
    else
        std::printf("=== Testing n=%u to %u ===\n\n", args.n_start, args.n_end);

    // Max bit width depends on 32 vs 64 bit mode
    const unsigned max_bw = is_64bit ? 64u : 32u;
    // Max exception skip threshold
    const unsigned max_exc_bw = is_64bit ? 60u : 28u;

    // Print table header
    printTableHeader(is_bitop_only, false, false);

    // Initialize result aggregation
    double grand_total_enc_diff = 0.0;
    double grand_total_dec_diff = 0.0;
    double grand_total_bitop_diff = 0.0;
    unsigned total_tests = 0;

    // Generate test scenarios
    std::vector<Scenario> scenarios
        = generateScenarios(args.exc_pct, args.simd128 || args.simd128v64 || args.simd128v64d1 || args.simd256v64d1 || args.simd128dec, args.simd256 || args.simd256dec);

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
                printTableHeader(is_bitop_only, false, false);
            }

            // Initialize per-scenario aggregation
            double total_enc_diff = 0.0;
            double total_dec_diff = 0.0;
            double total_bitop_diff = 0.0;
            unsigned tests_in_scenario = 0;

            // Loop over bit widths
            for (unsigned bw = 1; bw <= max_bw; ++bw)
            {
                // Skip bit widths outside the requested range
                if (args.bw_min > 0 && bw < args.bw_min)
                    continue;
                if (args.bw_max > 0 && bw > args.bw_max)
                    continue;

                // Skip high bit widths if forcing exceptions (need room for exception values above max_val)
                if (current_exc_pct >= 0.0 && bw > max_exc_bw)
                    continue;

                if (is_64bit)
                {
                    // =========================================================
                    // 64-bit data generation
                    // =========================================================
                    std::vector<uint64_t> input(n);
                    std::mt19937_64 rng(42ull + bw + n);
                    uint64_t max_val = (bw == 64) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << bw) - 1ull);
                    std::uniform_int_distribution<uint64_t> dist(0ull, max_val);

                    if (current_exc_pct >= 0.0 && bw < 64)
                    {
                        std::uniform_real_distribution<double> dist_prob(0.0, 100.0);
                        std::uniform_int_distribution<uint64_t> dist_exc(max_val + 1, 0xFFFFFFFFFFFFFFFFull);
                        for (auto & v : input)
                        {
                            if (dist_prob(rng) < current_exc_pct)
                                v = dist_exc(rng);
                            else
                                v = dist(rng);
                        }
                    }
                    else
                    {
                        for (auto & v : input)
                            v = dist(rng);
                    }

                    // =========================================================
                    // 64-bit benchmark dispatch
                    // =========================================================
                    if (args.bitpack64_only)
                    {
                        BitpackResult best{};
                        for (unsigned r = 0; r < args.runs; ++r)
                        {
                            auto result = runBitpack64Benchmark(input, bw, args.iters);
                            if (r == 0 || result.ref_mb_s > best.ref_mb_s)
                                best.ref_mb_s = result.ref_mb_s;
                            if (r == 0 || result.our_mb_s > best.our_mb_s)
                                best.our_mb_s = result.our_mb_s;
                        }
                        double diff = (best.our_mb_s / best.ref_mb_s - 1.0) * 100.0;
                        total_bitop_diff += diff;
                        std::printf(" %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n", n, bw, best.ref_mb_s, best.our_mb_s, diff);
                    }
                    else if (args.bitunpack64_only)
                    {
                        BitunpackResult best{};
                        for (unsigned r = 0; r < args.runs; ++r)
                        {
                            auto result = runBitunpack64Benchmark(input, bw, args.iters);
                            if (r == 0 || result.ref_mb_s > best.ref_mb_s)
                                best.ref_mb_s = result.ref_mb_s;
                            if (r == 0 || result.our_mb_s > best.our_mb_s)
                                best.our_mb_s = result.our_mb_s;
                        }
                        double diff = (best.our_mb_s / best.ref_mb_s - 1.0) * 100.0;
                        total_bitop_diff += diff;
                        std::printf(" %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n", n, bw, best.ref_mb_s, best.our_mb_s, diff);
                    }
                    else if (args.bitunpackd1_64_only)
                    {
                        BitunpackD1Result best{};
                        for (unsigned r = 0; r < args.runs; ++r)
                        {
                            auto result = runBitunpackD1_64Benchmark(input, bw, args.iters, 0ull);
                            if (r == 0 || result.ref_mb_s > best.ref_mb_s)
                                best.ref_mb_s = result.ref_mb_s;
                            if (r == 0 || result.our_mb_s > best.our_mb_s)
                                best.our_mb_s = result.our_mb_s;
                        }
                        double diff = (best.our_mb_s / best.ref_mb_s - 1.0) * 100.0;
                        total_bitop_diff += diff;
                        std::printf(" %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n", n, bw, best.ref_mb_s, best.our_mb_s, diff);
                    }
                    else
                    {
                        // p4enc64/p4d1dec64 or p4enc128v64/p4dec128v64 or p4enc128v64/p4D1Dec128v64
                        BenchResult best{};
                        for (unsigned r = 0; r < args.runs; ++r)
                        {
                            auto result = runBenchmark64(input, args.iters, args.simd128v64, args.simd128v64d1, args.simd256v64d1, args.p4dec64);
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
                }
                else
                {
                    // =========================================================
                    // 32-bit data generation
                    // =========================================================
                    std::vector<uint32_t> input(n);
                    std::mt19937 rng(42u + bw + n);
                    uint32_t max_val = (bw == 32) ? 0xFFFFFFFFu : ((1u << bw) - 1u);
                    std::uniform_int_distribution<uint32_t> dist(0u, max_val);

                    if (current_exc_pct >= 0.0)
                    {
                        std::uniform_real_distribution<double> dist_prob(0.0, 100.0);
                        std::uniform_int_distribution<uint32_t> dist_exc((1u << bw), 0xFFFFFFFFu);
                        for (auto & v : input)
                        {
                            if (dist_prob(rng) < current_exc_pct)
                                v = dist_exc(rng);
                            else
                                v = dist(rng);
                        }
                    }
                    else
                    {
                        for (auto & v : input)
                            v = dist(rng);
                    }

                    // =========================================================
                    // 32-bit benchmark dispatch
                    // =========================================================
                    if (args.bitpack_only)
                    {
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
                        total_bitop_diff += diff;
                        std::printf(" %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n", n, bw, best.ref_mb_s, best.our_mb_s, diff);
                    }
                    else if (args.bitunpackd1_only)
                    {
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
                        total_bitop_diff += diff;
                        std::printf(" %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n", n, bw, best.ref_mb_s, best.our_mb_s, diff);
                    }
                    else if (args.bitunpack_only)
                    {
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
                        total_bitop_diff += diff;
                        std::printf(" %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n", n, bw, best.ref_mb_s, best.our_mb_s, diff);
                    }
                    else if (args.d1enc)
                    {
                        std::sort(input.begin(), input.end());
                        BenchResult best{};
                        for (unsigned r = 0; r < args.runs; ++r)
                        {
                            auto result = runD1EncBenchmark(input, args.iters, args.simd128, args.simd256);
                            if (r == 0 || result.ref_enc_mb_s > best.ref_enc_mb_s)
                                best.ref_enc_mb_s = result.ref_enc_mb_s;
                            if (r == 0 || result.our_enc_mb_s > best.our_enc_mb_s)
                                best.our_enc_mb_s = result.our_enc_mb_s;
                        }
                        double enc_diff = (best.our_enc_mb_s / best.ref_enc_mb_s - 1.0) * 100.0;
                        total_bitop_diff += enc_diff;
                        std::printf(
                            " %3u |   %2u     | %6.1f   %6.1f   %+6.1f%%\n",
                            n,
                            bw,
                            best.ref_enc_mb_s,
                            best.our_enc_mb_s,
                            enc_diff);
                    }
                    else
                    {
                        bool use_nodelta = args.p4dec || args.simd128dec || args.simd256dec;
                        BenchResult best{};
                        for (unsigned r = 0; r < args.runs; ++r)
                        {
                            auto result = use_nodelta
                                ? runBenchmarkNoDelta(input, args.iters, args.simd128dec, args.simd256dec)
                                : runBenchmark(input, args.iters, args.simd128, args.simd256);
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
                }

                tests_in_scenario++;
            }

            // Print per-scenario summary if we have results
            if (tests_in_scenario > 0)
            {
                grand_total_enc_diff += total_enc_diff;
                grand_total_dec_diff += total_dec_diff;
                grand_total_bitop_diff += total_bitop_diff;
                total_tests += tests_in_scenario;

                printTableSeparator(is_bitop_only, false, false);

                if (scenarios.size() > 1)
                {
                    // Multiple scenarios - print per-scenario average
                    if (is_bitop_only)
                    {
                        std::printf("Avg  |          |                 %+6.1f%%\n", total_bitop_diff / tests_in_scenario);
                    }
                    else
                    {
                        std::printf(
                            "Avg  |          |                 %+6.1f%% |                 %+6.1f%%\n",
                            total_enc_diff / tests_in_scenario,
                            total_dec_diff / tests_in_scenario);
                    }
                    printTableSeparator(is_bitop_only, false, false);
                }
                else
                {
                    // Single scenario - print per-n average
                    if (is_bitop_only)
                    {
                        std::printf("Avg(%3u) |          |                 %+6.1f%%\n", n, total_bitop_diff / tests_in_scenario);
                    }
                    else
                    {
                        std::printf(
                            "Avg(%3u) |          |                 %+6.1f%% |                 %+6.1f%%\n",
                            n,
                            total_enc_diff / tests_in_scenario,
                            total_dec_diff / tests_in_scenario);
                    }
                    printTableSeparator(is_bitop_only, false, false);
                }
            }
        }
    }

    // Print grand summary if testing multiple element counts or SIMD
    if (args.n_end > args.n_start || args.simd128 || args.simd256 || args.simd128v64 || args.simd128v64d1 || args.simd256v64d1 || args.simd128dec || args.simd256dec)
    {
        if (is_bitop_only)
        {
            std::printf("Grand Avg|          |                 %+6.1f%%\n", grand_total_bitop_diff / total_tests);
            std::printf("\n=== Summary ===\n");
            if (args.bitpack64_only)
                std::printf("Bitpack64 average diff: %+.2f%%\n", grand_total_bitop_diff / total_tests);
            else if (args.bitunpack64_only)
                std::printf("Bitunpack64 average diff: %+.2f%%\n", grand_total_bitop_diff / total_tests);
            else if (args.bitunpackd1_64_only)
                std::printf("BitunpackD1_64 average diff: %+.2f%%\n", grand_total_bitop_diff / total_tests);
            else if (args.bitpack_only)
                std::printf("Bitpack average diff: %+.2f%%\n", grand_total_bitop_diff / total_tests);
            else if (args.bitunpackd1_only)
                std::printf("BitunpackD1 average diff: %+.2f%%\n", grand_total_bitop_diff / total_tests);
            else
                std::printf("Bitunpack average diff: %+.2f%%\n", grand_total_bitop_diff / total_tests);
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
