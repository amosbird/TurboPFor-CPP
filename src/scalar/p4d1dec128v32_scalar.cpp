// Scalar implementation of P4 decoding for 128v32 format
//
// This is the decode counterpart to p4enc128v32_scalar.cpp.
// Decodes P4-encoded data with 128v32 4-lane interleaved bitpacking format
// and applies delta1 decoding in a single pass.
//
// Format overview (matches encoder):
//   [header][payload]
//
// Header (1-2 bytes):
//   - Byte 0 high bits encode strategy
//   - 0xC0: constant block
//   - 0x40: variable-byte exceptions
//   - 0x80: bitmap exceptions
//   - Otherwise: simple bitpacking
//
// Payload depends on encoding strategy (see p4enc128v32_scalar.cpp for details)
// Bitpacking uses 128v32 format: 4-lane interleaved horizontal packing

#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

// Apply delta1 decoding to output array
//
// Delta1 format: out[i] = sum(in[0..i]) + (i+1) + start
// Each value is stored as (actual_value - previous_value - 1)
// So decoding adds 1 to each delta before prefix summing
//
// Parameters:
//   out: Array to decode in-place
//   n: Number of elements
//   start: Initial value (previous element before this block)
void applyDelta1(uint32_t * out, unsigned n, uint32_t start)
{
    if (n == 0u)
        return;

    uint32_t acc = start;
    for (unsigned i = 0; i < n; ++i)
    {
        acc += out[i] + 1u;
        out[i] = acc;
    }
}

// Decode P4 payload with bitmap exceptions for 128v32 format
//
// Format: [bitmap][patch bits][base bits]
//
// Parameters:
//   in: Input buffer positioned at bitmap
//   n: Number of elements (typically 128)
//   out: Output array
//   start: Delta1 start value
//   b: Base bit width (with 0x80 flag cleared)
//   bx: Exception bit width
//
// Returns: Pointer past end of consumed input
const unsigned char * p4D1Dec128PayloadBitmap(
    const unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Phase 1: Read and parse bitmap
    //
    // Bitmap has n bits (one per value), padded to byte boundary
    // Bit set = position has exception that needs patching
    uint64_t bitmap[MAX_VALUES / 64] = {0};
    const unsigned words = (n + 63u) / 64u;
    unsigned exception_count = 0;

    for (unsigned i = 0; i < words; ++i)
    {
        uint64_t word = loadU64(in + i * sizeof(uint64_t));

        // Mask off unused bits in last word
        if (i == words - 1u && (n & 0x3Fu))
            word &= (1ull << (n & 0x3Fu)) - 1ull;

        bitmap[i] = word;

        // Count exceptions using popcount
#if defined(__GNUC__) || defined(__clang__)
        exception_count += static_cast<unsigned>(__builtin_popcountll(word));
#else
        while (word)
        {
            ++exception_count;
            word &= word - 1ull;
        }
#endif
    }

    const unsigned char * ip = in + pad8(n);

    // Phase 2: Unpack exception values (horizontal bitpacking)
    uint32_t exceptions[MAX_VALUES + 64] = {0};
    ip = bitunpack32Scalar(const_cast<unsigned char *>(ip), exception_count, exceptions, bx);

    // Phase 3: Unpack base values (128v32 vertical bitpacking)
    ip = bitunpack128v32Scalar(const_cast<unsigned char *>(ip), out, b);

    // Phase 4: Apply patches
    //
    // For each set bit in bitmap, OR in the high bits from exceptions
    unsigned k = 0;
    for (unsigned i = 0; i < words; ++i)
    {
        uint64_t word = bitmap[i];
        while (word)
        {
            // Find lowest set bit
#if defined(__GNUC__) || defined(__clang__)
            unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
#else
            unsigned bit = 0;
            while (((word >> bit) & 1ull) == 0ull)
                ++bit;
#endif
            const unsigned idx = i * 64u + bit;
            out[idx] |= static_cast<uint32_t>(exceptions[k++] << b);
            word &= word - 1ull; // Clear lowest set bit
        }
    }

    // Phase 5: Apply delta1 decoding
    applyDelta1(out, n, start);

    return ip;
}

// Decode P4 payload for 128v32 format
//
// Handles all encoding strategies based on header flags:
//   - No exceptions (b without 0x80 flag): simple unpack + delta1
//   - Bitmap exceptions (b with 0x80 flag, bx > 0): bitmap + patch + base
//
// Parameters:
//   in: Input buffer positioned at payload
//   n: Number of elements (typically 128)
//   out: Output array
//   start: Delta1 start value
//   b: Base bit width (may have 0x80 flag)
//   bx: Exception strategy (0 = none, 1-31 = bitmap patching)
//
// Returns: Pointer past end of consumed input
const unsigned char * p4D1Dec128Payload(
    const unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b, unsigned bx)
{
    using namespace turbopfor::scalar::detail;

    // Check for exception flag in b
    if ((b & 0x80u) == 0u)
    {
        // No exceptions - simple unpack + delta1
        unsigned char * ip = bitunpack128v32Scalar(const_cast<unsigned char *>(in), out, b);
        applyDelta1(out, n, start);
        return ip;
    }

    // Clear exception flag
    b &= 0x7Fu;

    if (bx == 0u)
    {
        // Bitmap says no exceptions - simple unpack + delta1
        unsigned char * ip = bitunpack128v32Scalar(const_cast<unsigned char *>(in), out, b);
        applyDelta1(out, n, start);
        return ip;
    }

    // Bitmap exception handling
    return p4D1Dec128PayloadBitmap(in, n, out, start, b, bx);
}

} // namespace

// Main P4 decoding entry point for 128v32 format (scalar implementation)
//
// Decodes P4-encoded data and applies delta1 decoding.
// This is the decode counterpart to p4Enc128v32.
//
// Parameters:
//   in: Input buffer containing P4-encoded data
//   n: Number of values to decode (typically 128)
//   out: Output array for decoded values
//   start: Initial value for delta1 decoding
//
// Returns: Pointer past end of consumed input
//
// Binary compatibility: Correctly decodes data from both scalar and SIMD encoders
unsigned char * p4D1Dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start)
{
    using namespace turbopfor::scalar::detail;

    if (n == 0u)
        return in;

    unsigned char * ip = in;
    unsigned b = *ip++;

    // Case 1: Constant block (all values identical)
    //
    // Header byte has 0xC0 pattern
    // Payload: single value stored in minimal bytes
    if ((b & 0xC0u) == 0xC0u)
    {
        b &= 0x3Fu; // Extract bit width

        // Load value (only read bytes needed based on bit width)
        uint32_t value = loadU32(ip);
        if (b < MAX_BITS)
            value &= maskBits(b);

        // Fill output with constant
        for (unsigned i = 0; i < n; ++i)
            out[i] = value;

        // Apply delta1 decoding
        applyDelta1(out, n, start);

        return ip + (b + 7u) / 8u;
    }

    // Case 2: Standard bitpacking (possibly with bitmap exceptions)
    //
    // Header byte without 0x40 flag
    // 0x80 flag indicates exception info follows
    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        if (b & 0x80u)
            bx = *ip++; // Read exception bit width

        return const_cast<unsigned char *>(p4D1Dec128Payload(ip, n, out, start, b, bx));
    }

    // Case 3: Variable-byte exception encoding
    //
    // Header: 0x40 flag set
    // Format: [b|0x40][exception_count][base bits][vbyte exceptions][positions]
    unsigned bx = *ip++; // Exception count (reusing bx variable)
    b &= 0x3Fu;          // Extract base bit width

    // Unpack base values (128v32 format)
    ip = bitunpack128v32Scalar(ip, out, b);

    // Decode variable-byte exceptions
    uint32_t exceptions[MAX_VALUES + 64] = {0};
    ip = vbDec32(ip, bx, exceptions);

    // Apply patches using position list
    for (unsigned i = 0; i < bx; ++i)
        out[ip[i]] |= static_cast<uint32_t>(exceptions[i] << b);

    ip += bx;

    // Apply delta1 decoding
    applyDelta1(out, n, start);

    return ip;
}

} // namespace turbopfor::scalar
