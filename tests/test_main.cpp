#include <cstdio>

// 32-bit tests
unsigned runBinaryCompatibilityTest();
unsigned runCrossValidation128vTest();
unsigned runBinaryCompatibility128vTest();
unsigned runBitunpackCompatibilityTest();
unsigned runBitunpackD1CompatibilityTest();
unsigned runCrossValidation256vTest();
unsigned runBinaryCompatibility256vTest();

// 32-bit non-delta decode tests
unsigned runP4Dec32CompatibilityTest();
unsigned runP4Dec128v32CompatibilityTest();
unsigned runP4Dec256v32CompatibilityTest();

// 64-bit tests
unsigned runBitpack64CompatibilityTest();
unsigned runBinaryCompatibility64Test();
unsigned runBinaryCompatibility128v64Test();
unsigned runBinaryCompatibility256v64Test();
unsigned runVbyte64CompatibilityTest();

// D1 encode tests
unsigned runD1Enc32CompatibilityTest();
unsigned runD1Enc128v32CompatibilityTest();
unsigned runD1Enc256v32CompatibilityTest();
unsigned runD1Enc64CompatibilityTest();
unsigned runD1Enc128v64RoundtripTest();
unsigned runD1Enc256v64RoundtripTest();

int main()
{
    unsigned failed_p4_32 = runBinaryCompatibilityTest();
    unsigned failed_128v_cross = runCrossValidation128vTest();
    unsigned failed_128v_compat = runBinaryCompatibility128vTest();
    unsigned failed_bitunpack = runBitunpackCompatibilityTest();
    unsigned failed_bitunpack_d1 = runBitunpackD1CompatibilityTest();
    unsigned failed_256v_cross = runCrossValidation256vTest();
    unsigned failed_256v_compat = runBinaryCompatibility256vTest();

    unsigned failed_p4dec32 = runP4Dec32CompatibilityTest();
    unsigned failed_p4dec128v32 = runP4Dec128v32CompatibilityTest();
    unsigned failed_p4dec256v32 = runP4Dec256v32CompatibilityTest();

    unsigned failed_bitpack64 = runBitpack64CompatibilityTest();
    unsigned failed_p4_64 = runBinaryCompatibility64Test();
    unsigned failed_128v64 = runBinaryCompatibility128v64Test();
    unsigned failed_256v64 = runBinaryCompatibility256v64Test();
    unsigned failed_vbyte64 = runVbyte64CompatibilityTest();

    unsigned failed_d1enc32 = runD1Enc32CompatibilityTest();
    unsigned failed_d1enc128v32 = runD1Enc128v32CompatibilityTest();
    unsigned failed_d1enc256v32 = runD1Enc256v32CompatibilityTest();
    unsigned failed_d1enc64 = runD1Enc64CompatibilityTest();
    unsigned failed_d1enc128v64 = runD1Enc128v64RoundtripTest();
    unsigned failed_d1enc256v64 = runD1Enc256v64RoundtripTest();

    unsigned total = failed_p4_32 + failed_128v_cross + failed_128v_compat + failed_bitunpack + failed_bitunpack_d1 + failed_256v_cross
        + failed_256v_compat + failed_p4dec32 + failed_p4dec128v32 + failed_p4dec256v32 + failed_bitpack64 + failed_p4_64 + failed_128v64 + failed_256v64 + failed_vbyte64
        + failed_d1enc32 + failed_d1enc128v32 + failed_d1enc256v32 + failed_d1enc64 + failed_d1enc128v64 + failed_d1enc256v64;

    std::printf("=== Summary ===\n");
    std::printf("  p4enc/dec 32-bit:      %u failures\n", failed_p4_32);
    std::printf("  128v32 cross:          %u failures\n", failed_128v_cross);
    std::printf("  128v32 compat:         %u failures\n", failed_128v_compat);
    std::printf("  bitunpack32:           %u failures\n", failed_bitunpack);
    std::printf("  bitunpack32 d1:        %u failures\n", failed_bitunpack_d1);
    std::printf("  256v32 cross:          %u failures\n", failed_256v_cross);
    std::printf("  256v32 compat:         %u failures\n", failed_256v_compat);
    std::printf("  p4dec32:               %u failures\n", failed_p4dec32);
    std::printf("  p4dec128v32:           %u failures\n", failed_p4dec128v32);
    std::printf("  p4dec256v32:           %u failures\n", failed_p4dec256v32);
    std::printf("  bitpack64:             %u failures\n", failed_bitpack64);
    std::printf("  p4enc/dec 64-bit:      %u failures\n", failed_p4_64);
    std::printf("  128v64 compat:         %u failures\n", failed_128v64);
    std::printf("  256v64 roundtrip:      %u failures\n", failed_256v64);
    std::printf("  vbyte64:               %u failures\n", failed_vbyte64);
    std::printf("  p4D1Enc32:             %u failures\n", failed_d1enc32);
    std::printf("  p4D1Enc128v32:         %u failures\n", failed_d1enc128v32);
    std::printf("  p4D1Enc256v32:         %u failures\n", failed_d1enc256v32);
    std::printf("  p4D1Enc64:             %u failures\n", failed_d1enc64);
    std::printf("  p4D1Enc128v64:         %u failures\n", failed_d1enc128v64);
    std::printf("  p4D1Enc256v64:         %u failures\n", failed_d1enc256v64);
    std::printf("Total failures: %u\n", total);

    return total > 0 ? 1 : 0;
}
