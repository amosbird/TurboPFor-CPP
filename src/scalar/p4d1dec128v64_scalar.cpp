// Scalar implementation of P4 decoding for 128v64 format
//
// This is the decode counterpart to p4enc128v64_scalar.cpp.
// Decodes P4-encoded data with 128v64 hybrid bitpacking format
// and applies delta1 decoding in a single pass.
//
// Note: TurboPFor does NOT provide p4d1dec128v64 — this is our own addition
// for API symmetry. TurboPFor only has non-delta p4dec128v64, and delta1
// variants for 128v are limited to USIZE 16 and 32.
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
//   - 63→64 bit mapping applies for 64-bit variant
//
// Payload depends on encoding strategy (see p4enc128v64_scalar.cpp for details)
// Base values use 128v64 hybrid format (128v32 when b<=32, scalar64 when b>32)
// Exception values use scalar bitpack64 or vbyte64

#include "p4_scalar.h"
#include "p4_scalar_internal.h"

namespace turbopfor::scalar
{

namespace
{

// Apply delta1 decoding to 64-bit output array
//
// Delta1 format: out[i] = sum(in[0..i]) + (i+1) + start
// Each value is stored as (actual_value - previous_value - 1)
// So decoding adds 1 to each delta before prefix summing
void applyDelta1_64(uint64_t * out, unsigned n, uint64_t start)
{
    for (unsigned i = 0; i < n; ++i)
        out[i] = (start += out[i]) + (i + 1u);
}

// Decode P4 payload with bitmap exceptions for 128v64 format
//
// Format: [bitmap][patch bits (bitpack64)][base bits (bitpack128v64)]
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
const unsigned char *
p4D1Dec128v64PayloadBitmap(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b, unsigned bx)
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
        uint64_t word = loadU64Fast(in + i * sizeof(uint64_t));

        // Mask off unused bits in last word
        if (i == words - 1u && (n & 0x3Fu))
            word &= (1ull << (n & 0x3Fu)) - 1ull;

        bitmap[i] = word;

        // Count exceptions using popcount
#if defined(__GNUC__) || defined(__clang__)
        exception_count += static_cast<unsigned>(__builtin_popcountll(word));
#else
        uint64_t tmp = word;
        while (tmp)
        {
            ++exception_count;
            tmp &= tmp - 1ull;
        }
#endif
    }

    const unsigned char * ip = in + pad8(n);

    // Phase 2: Unpack exception values (scalar 64-bit horizontal bitpacking)
    //
    // Exceptions are ALWAYS packed with bitpack64 (scalar), regardless of b.
    // The encoder uses: out = bitpack64Scalar(exceptions, exception_count, out, bx)
    uint64_t exceptions[MAX_VALUES + 64] = {0};
    ip = bitunpack64Scalar(const_cast<unsigned char *>(ip), exception_count, exceptions, bx);

    // Phase 3: Unpack base values (128v64 hybrid format)
    //
    // Uses bitunpack128v64: 128v32 when b<=32, scalar64 when b>32
    ip = bitunpack128v64Scalar(const_cast<unsigned char *>(ip), out, b);

    // Phase 4: Apply patches
    //
    // For each set bit in bitmap, OR in the high bits from exceptions
    unsigned k = 0;
    for (unsigned i = 0; i < words; ++i)
    {
        uint64_t word = bitmap[i];
        while (word)
        {
#if defined(__GNUC__) || defined(__clang__)
            unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
#else
            unsigned bit = 0;
            while (((word >> bit) & 1ull) == 0ull)
                ++bit;
#endif
            const unsigned idx = i * 64u + bit;
            out[idx] |= exceptions[k++] << b;
            word &= word - 1ull; // Clear lowest set bit
        }
    }

    // Phase 5: Apply delta1 decoding
    applyDelta1_64(out, n, start);

    return ip;
}

} // namespace

// Main P4 decoding entry point for 128v64 format with delta1 (scalar implementation)
//
// Decodes P4-encoded data (produced by p4Enc128v64) and applies delta1 decoding.
// This function handles all P4 encoding strategies:
//   1. Simple bitpacking (no exceptions)
//   2. Bitmap exception patching (PFOR)
//   3. Variable-byte exceptions
//   4. Constant blocks
//
// The 63→64 bit mapping is applied when parsing the header.
//
// Parameters:
//   in: Input buffer containing P4-encoded data
//   n: Number of values to decode (typically 128)
//   out: Output array for decoded values
//   start: Initial value for delta1 decoding
//
// Returns: Pointer past end of consumed input
unsigned char * p4D1Dec128v64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start)
{
    using namespace turbopfor::scalar::detail;

    if (n == 0u)
        return in;

    unsigned char * ip = in;
    unsigned b = *ip++;

    // Case 1: Constant block (all values identical)
    //
    // Header byte has 0xC0 pattern
    // Payload: single value stored in minimal bytes (up to 8)
    if ((b & 0xC0u) == 0xC0u)
    {
        b &= 0x3Fu; // Extract bit width

        // 64-bit quirk: 63 in header means 64
        if (b == 63u)
            b = 64u;

        const unsigned bytes_stored = (b + 7u) / 8u;

        // Load constant value — TurboPFor reads full 8-byte word then masks
        uint64_t value = loadU64Fast(ip);
        if (b < 64u)
            value &= (1ull << b) - 1ull;

        // Fill output with constant + apply delta1
        for (unsigned i = 0; i < n; ++i)
            out[i] = (start += value) + (i + 1u);

        return ip + bytes_stored;
    }

    // Case 2: Standard bitpacking (possibly with bitmap exceptions)
    //
    // Header byte without 0x40 flag
    // 0x80 flag indicates exception info follows
    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        if (b & 0x80u)
        {
            bx = *ip++; // Read exception bit width
            b &= 0x7Fu; // Clear exception flag
        }

        // 64-bit quirk: 63 in header means 64
        if (b == 63u)
            b = 64u;

        // No exceptions — simple unpack + delta1
        if (bx == 0u)
        {
            ip = bitunpack128v64Scalar(ip, out, b);
            applyDelta1_64(out, n, start);
            return ip;
        }

        // Bitmap exception handling
        return const_cast<unsigned char *>(p4D1Dec128v64PayloadBitmap(ip, n, out, start, b, bx));
    }

    // Case 3: Variable-byte exception encoding
    //
    // Header: 0x40 flag set
    // Format: [b|0x40][exception_count][base bits (128v64)][vbyte64 exceptions][positions]
    b &= 0x3Fu; // Extract base bit width (remove 0x40 flag)

    // 64-bit quirk: 63 in header means 64
    if (b == 63u)
        b = 64u;

    const unsigned exception_count = *ip++;

    // Unpack base values (128v64 hybrid format)
    ip = bitunpack128v64Scalar(ip, out, b);

    // Decode variable-byte exceptions (64-bit)
    uint64_t exceptions[MAX_VALUES + 64] = {0};
    ip = vbDec64(ip, exception_count, exceptions);

    // Apply patches using position list
    for (unsigned i = 0; i < exception_count; ++i)
        out[ip[i]] |= exceptions[i] << b;

    ip += exception_count;

    // Apply delta1 decoding
    applyDelta1_64(out, n, start);

    return ip;
}

// Non-delta P4 decoding for 128v64 format (scalar implementation)
//
// Same as p4D1Dec128v64 but without delta1 prefix sum.
// Matches TurboPFor C's p4dec128v64 behavior.
unsigned char * p4Dec128v64(unsigned char * in, unsigned n, uint64_t * out)
{
    using namespace turbopfor::scalar::detail;

    if (n == 0u)
        return in;

    unsigned char * ip = in;
    unsigned b = *ip++;

    // Case 1: Constant block
    if ((b & 0xC0u) == 0xC0u)
    {
        b &= 0x3Fu;
        if (b == 63u)
            b = 64u;

        const unsigned bytes_stored = (b + 7u) / 8u;
        uint64_t value = loadU64Fast(ip);
        if (b < 64u)
            value &= (1ull << b) - 1ull;

        for (unsigned i = 0; i < n; ++i)
            out[i] = value;

        return ip + bytes_stored;
    }

    // Case 2: Standard bitpacking (possibly with bitmap exceptions)
    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        if (b & 0x80u)
        {
            bx = *ip++;
            b &= 0x7Fu;
        }

        if (b == 63u)
            b = 64u;

        if (bx == 0u)
        {
            ip = bitunpack128v64Scalar(ip, out, b);
            return ip;
        }

        // Bitmap exception handling (reuse internal helper without delta)
        // Inline here since the non-delta bitmap helper is local to this TU
        uint64_t bitmap[MAX_VALUES / 64] = {0};
        const unsigned words = (n + 63u) / 64u;
        unsigned exception_count = 0;

        for (unsigned i = 0; i < words; ++i)
        {
            uint64_t word = loadU64Fast(ip + i * sizeof(uint64_t));
            if (i == words - 1u && (n & 0x3Fu))
                word &= (1ull << (n & 0x3Fu)) - 1ull;

            bitmap[i] = word;
#if defined(__GNUC__) || defined(__clang__)
            exception_count += static_cast<unsigned>(__builtin_popcountll(word));
#else
            uint64_t tmp = word;
            while (tmp)
            {
                ++exception_count;
                tmp &= tmp - 1ull;
            }
#endif
        }

        ip += pad8(n);

        uint64_t exceptions[MAX_VALUES + 64] = {0};
        ip = bitunpack64Scalar(ip, exception_count, exceptions, bx);

        ip = bitunpack128v64Scalar(ip, out, b);

        unsigned k = 0;
        for (unsigned i = 0; i < words; ++i)
        {
            uint64_t word = bitmap[i];
            while (word)
            {
#if defined(__GNUC__) || defined(__clang__)
                unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
#else
                unsigned bit = 0;
                while (((word >> bit) & 1ull) == 0ull)
                    ++bit;
#endif
                const unsigned idx = i * 64u + bit;
                out[idx] |= exceptions[k++] << b;
                word &= word - 1ull;
            }
        }

        return ip;
    }

    // Case 3: Variable-byte exceptions
    b &= 0x3Fu;
    if (b == 63u)
        b = 64u;

    const unsigned exception_count = *ip++;

    ip = bitunpack128v64Scalar(ip, out, b);

    uint64_t exceptions[MAX_VALUES + 64] = {0};
    ip = vbDec64(ip, exception_count, exceptions);

    for (unsigned i = 0; i < exception_count; ++i)
        out[ip[i]] |= exceptions[i] << b;

    ip += exception_count;
    return ip;
}

} // namespace turbopfor::scalar
