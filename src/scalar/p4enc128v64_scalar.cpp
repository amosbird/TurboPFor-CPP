// Scalar implementation of P4 encoding for 128v64 format
//
// P4 (Patched Frame-of-Reference) encoding with 128-element hybrid bitpacking.
// 128v64 is a hybrid format:
//   - When b <= 32: uses 128v32 4-lane interleaved SIMD format (values fit in 32 bits)
//   - When b > 32: uses scalar 64-bit horizontal bitpacking
//
// The exception patch bits are ALWAYS encoded with scalar bitpack64, regardless of b.
// The base values use the hybrid bitpack128v64 format (delegating to 128v32 or scalar64).
//
// Format overview:
//   [header][payload]
//
// Header (1-2 bytes):
//   - Same format as standard P4, but with 63→64 bit mapping:
//     bit width 64 is stored as 63 in the header (only 6 bits available)
//   - Flags: 0x00 = bitpack only, 0x80 = PFOR, 0x40 = vbyte, 0xC0 = constant
//
// Payload depends on encoding strategy:
//   - b=0, bx=0: All zeros, no payload
//   - bx=0: Simple bitpacking using 128v64 hybrid format
//   - bx=66 (constant): Single value stored (up to 8 bytes)
//   - bx=1-64: Bitwise patching (bitmap + patch bits + base bits)
//   - bx=65: Variable-byte exceptions (count + base bits + vbyte values + positions)
//
// Binary compatibility: Output is bit-identical to TurboPFor's p4enc128v64()

#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

// Encode P4 block payload with exception handling for 128v64 format
//
// Handles two exception encoding strategies:
// 1. Bitwise patching (bx <= 64): bitmap + patch bits (bitpack64) + base bits (bitpack128v64)
// 2. Variable-byte (bx == 65): count + base bits (bitpack128v64) + vbyte64 values + positions
//
// Parameters:
//   in: Input array of n values (n <= 128)
//   n: Number of values
//   out: Output buffer
//   b: Base bit width (bits per value for base data)
//   bx: Exception strategy indicator
//
// Returns: Pointer to end of encoded data
unsigned char * p4Enc128v64PayloadExceptions(uint64_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Create mask for extracting base bits (low b bits of each value)
    const uint64_t base_mask = maskBits64(b);

    // Temporary arrays for exception handling
    uint64_t base[MAX_VALUES]; // Base values (low b bits)
    uint64_t exceptions[MAX_VALUES]; // Exception values (high bits)
    unsigned exception_positions[MAX_VALUES]; // Indices of exception values

    // Phase 1: Split values into base and exceptions
    //
    // For each value:
    // - Extract low b bits as base value
    // - If value > base_mask, it's an exception: record position and high bits
    //
    // Uses branchless technique from TurboPFor to avoid branch misprediction:
    // always write position, only increment count when it's an exception.
    unsigned exception_count = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        exception_positions[exception_count] = i;
        exception_count += (in[i] > base_mask) ? 1u : 0u;
        base[i] = in[i] & base_mask;
    }

    // Phase 2: Build exception data
    uint64_t bitmap[MAX_VALUES / 64] = {0}; // 1 bit per value: 1 = has exception
    for (unsigned i = 0; i < exception_count; ++i)
    {
        const unsigned pos = exception_positions[i];
        bitmap[pos >> 6] |= 1ull << (pos & 0x3Fu);
        exceptions[i] = in[pos] >> b;
    }

    // Phase 3: Encode based on exception strategy
    if (bx <= MAX_BITS_64)
    {
        // Strategy 1: Bitwise patching
        //
        // Format: [bitmap][patch bits][base bits]
        //
        // bitmap: n bits (padded to byte boundary), 1 = position has exception
        // patch bits: exception_count values packed with bitpack64 (scalar horizontal)
        // base bits: n values packed with bitpack128v64 (hybrid format)

        // Write exception bitmap
        const unsigned bitmap_words = (n + 63u) / 64u;
        for (unsigned i = 0; i < bitmap_words; ++i)
        {
            storeU64Fast(out + i * sizeof(uint64_t), bitmap[i]);
        }

        out += pad8(n); // Advance by bitmap size (padded to bytes)

        // Write patch bits using scalar 64-bit horizontal bitpacking
        // Note: exceptions ALWAYS use bitpack64 (not bitpack128v64)
        out = bitpack64Scalar(exceptions, exception_count, out, bx);

        // Write base values using 128v64 hybrid format
        out = bitpack128v64Scalar(base, out, b);

        return out;
    }

    // Strategy 2: Variable-byte exception encoding
    //
    // Format: [exception_count][base bits][vbyte exceptions][position list]
    //
    // exception_count: 1 byte
    // base bits: n values using 128v64 hybrid format
    // vbyte exceptions: variable-byte encoded 64-bit exception values
    // position list: exception_count bytes, each is position in [0, n)

    *out++ = static_cast<unsigned char>(exception_count);

    // Write base values using 128v64 hybrid format
    out = bitpack128v64Scalar(base, out, b);

    // Write exception values using 64-bit variable-byte encoding
    out = vbEnc64(exceptions, exception_count, out);

    // Write exception positions (1 byte each, since n <= 128)
    for (unsigned i = 0; i < exception_count; ++i)
        *out++ = static_cast<unsigned char>(exception_positions[i]);

    return out;
}

// Encode P4 block payload for 128v64 format
//
// Handles all encoding strategies based on (b, bx) determined by p4Bits64:
//   - b=0, bx=0: All zeros
//   - bx=0: Simple bitpacking (no exceptions)
//   - bx=66: Constant block (all values equal)
//   - bx=1-64: Bitwise patching
//   - bx=65: Variable-byte exceptions
//
// Parameters:
//   in: Input array of n values
//   n: Number of values (should be 128 for 128v64 format)
//   out: Output buffer
//   b: Base bit width
//   bx: Exception strategy
//
// Returns: Pointer to end of encoded data
unsigned char * p4Enc128v64Payload(uint64_t * in, unsigned n, unsigned char * out, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Case 1: No exceptions needed
    if (bx == 0u)
    {
        // Simple 128v64 hybrid bitpacking, no exception handling
        return bitpack128v64Scalar(in, out, b);
    }

    // Case 2: Constant block (all values are identical)
    // bx = MAX_BITS_64 + 2 = 66 indicates constant encoding
    if (bx == MAX_BITS_64 + 2u)
    {
        // Store single value using full-width write, advance by needed bytes
        // TurboPFor: ctou64(out) = in[0]; return out + (b+7)/8;
        storeU64Fast(out, in[0]);
        return out + (b + 7u) / 8u;
    }

    // Case 3: Exception handling required
    return p4Enc128v64PayloadExceptions(in, n, out, b, bx);
}

} // namespace

// Main P4 encoding entry point for 128v64 format
//
// Encodes up to 128 uint64_t values using P4 encoding with 128v64 hybrid bitpacking.
// The 128v64 hybrid format uses 128v32 (SIMD-friendly) when b<=32, and scalar 64-bit
// horizontal bitpacking when b>32.
//
// Encoding steps:
// 1. Analyze input to determine optimal bit width and exception strategy
// 2. Write header (1-2 bytes, with 64→63 clamping)
// 3. Write payload using appropriate strategy
//
// Parameters:
//   in: Input array of uint64_t values
//   n: Number of values (typically 128)
//   out: Output buffer (must have sufficient space)
//
// Returns: Pointer past end of encoded data
//
// Binary compatibility: Output is bit-identical to TurboPFor's p4enc128v64()
unsigned char * p4Enc128v64(uint64_t * in, unsigned n, unsigned char * out)
{
    using namespace turbopfor::scalar::detail;

    // Empty input produces no output
    if (n == 0u)
        return out;

    // Analyze input to find optimal encoding parameters
    // b: base bit width (minimum bits to represent most values)
    // bx: exception strategy (0=none, 1-64=patching, 65=vbyte, 66=constant)
    unsigned bx = 0;
    unsigned b = p4Bits64(in, n, &bx);

    // Write P4 header (1-2 bytes, with 64→63 clamping for header encoding)
    writeHeader64(out, b, bx);

    // Write payload using determined strategy
    return p4Enc128v64Payload(in, n, out, b, bx);
}

} // namespace turbopfor::scalar
