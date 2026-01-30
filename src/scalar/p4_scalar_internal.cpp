#include "p4_scalar_internal.h"

#include <algorithm>

namespace turbopfor::scalar::detail
{

namespace
{

// Encode a single uint32_t value using variable-byte encoding scheme.
//
// Encoding scheme (4 size classes based on value magnitude):
//
// 1-byte:  [0x00..0x9B]           for values [0, 156)
//   Format: [value]
//   Direct storage of small values
//
// 2-byte:  [0x9C..0xDB][data]     for values [156, 16540)
//   Format: [marker][low_byte]
//   Marker encodes high bits: marker = 0x9C + (delta >> 8)
//   Data byte encodes low bits: data = delta & 0xFF
//   Where: delta = value - 156
//   Range: 64 marker values × 256 data values = 16384 values
//
// 3-byte:  [0xDC..0xFB][lo][hi]   for values [16540, 2113692)
//   Format: [marker][byte0][byte1]  (little-endian data bytes)
//   Marker encodes highest bits: marker = 0xDC + (delta >> 16)
//   Data bytes encode low 16 bits: [delta & 0xFF][(delta >> 8) & 0xFF]
//   Where: delta = value - 16540
//   Range: 32 marker values × 65536 data values = 2097152 values
//
// 4+ byte: [0xFC..0xFD][bytes...] for values [2113692, 2^32-1]
//   Format: [marker][3-4 raw bytes]
//   Marker indicates byte count: marker = 0xFC + (byte_count - 3)
//   Followed by raw little-endian bytes of the value
//   For uint32_t, only markers 0xFC (3 bytes) and 0xFD (4 bytes) are used
//   Note: Markers 0xFE and 0xFF reserved (0xFF used as uncompressed escape in vbEnc32)
//
// Design rationale:
// - Small values (most common) use 1 byte
// - Medium values use 2-3 bytes with marker encoding high bits
// - Large values use direct storage with minimal overhead
// - Self-describing: First byte indicates format (no separate length field needed)
//
// Note: Takes output pointer by reference; automatically advances 'out' to next position
void vbPut32(unsigned char *& out, uint32_t x)
{
    if (x < VBYTE_THRESHOLD_2BYTE) // x < 156
    {
        // 1-byte encoding: store value directly
        *out++ = static_cast<uint8_t>(x);
    }
    else if (x < VBYTE_THRESHOLD_3BYTE) // x < 16540
    {
        // 2-byte encoding: marker (0x9C-0xDB) + 1 data byte
        // Encode delta from threshold to maximize value range per byte
        const unsigned delta = x - VBYTE_THRESHOLD_2BYTE;
        *out++ = static_cast<uint8_t>(VBYTE_MARKER_2BYTE + (delta >> 8)); // High 6 bits in marker
        *out++ = static_cast<uint8_t>(delta & 0xFFu); // Low 8 bits in data
    }
    else if (x < VBYTE_THRESHOLD_4PLUS) // x < 2113692
    {
        // 3-byte encoding: marker (0xDC-0xFB) + 2 data bytes (little-endian)
        const unsigned delta = x - VBYTE_THRESHOLD_3BYTE;
        *out++ = static_cast<uint8_t>(VBYTE_MARKER_3BYTE + (delta >> 16)); // High 5 bits in marker
        *out++ = static_cast<uint8_t>(delta & 0xFFu); // Low byte
        *out++ = static_cast<uint8_t>((delta >> 8) & 0xFFu); // Middle byte
    }
    else
    {
        // 4+ byte encoding: values [2113692, 2^32-1]
        // For uint32_t, only 3-byte (marker 0xFC) or 4-byte (marker 0xFD) encodings are used
        // Threshold: 24-bit max = 0xFFFFFF = 16777215
        if (x <= 0xFFFFFFu) // 3-byte: values [2113692, 16777215]
        {
            *out++ = static_cast<uint8_t>(VBYTE_MARKER_4PLUS); // 0xFC
            *out++ = static_cast<uint8_t>(x);
            *out++ = static_cast<uint8_t>(x >> 8);
            *out++ = static_cast<uint8_t>(x >> 16);
        }
        else // 4-byte: values [16777216, 2^32-1]
        {
            *out++ = static_cast<uint8_t>(VBYTE_MARKER_4PLUS + 1u); // 0xFD
            storeU32Fast(out, x);
            out += 4;
        }
    }
}

// Decode a single variable-byte encoded uint32_t value.
//
// Decoding logic (inverse of vbPut32):
// - Read first byte (marker) to determine format
// - Extract encoded value based on marker range
// - Return pointer to next unread byte
//
// Marker ranges (matching vbPut32 encoding):
// - [0x00..0x9B]: 1-byte encoding (value = marker)
// - [0x9C..0xDB]: 2-byte encoding (value = 156 + decode(marker, data_byte))
// - [0xDC..0xFB]: 3-byte encoding (value = 16540 + decode(marker, data_bytes))
// - [0xFC..0xFD]: 4+ byte encoding (read raw bytes)
//
// Returns: Pointer to next byte after the decoded value
unsigned char * vbGet32(unsigned char * in, uint32_t & x)
{
    const unsigned marker = *in++;

    if (marker < VBYTE_MARKER_2BYTE) // marker < 0x9C
    {
        // 1-byte encoding: value stored directly in marker
        x = marker;
        return in;
    }

    if (marker < VBYTE_MARKER_3BYTE) // marker < 0xDC
    {
        // 2-byte encoding: reconstruct value from marker + 1 data byte
        // Inverse of: marker = 0x9C + (delta >> 8), data = delta & 0xFF
        const unsigned data_byte = *in++;
        const uint32_t delta = ((marker - VBYTE_MARKER_2BYTE) << 8) + data_byte;
        x = delta + VBYTE_THRESHOLD_2BYTE;
        return in;
    }

    if (marker < VBYTE_MARKER_4PLUS) // marker < 0xFC
    {
        const unsigned low16 = loadU16Fast(in);
        x = low16 + ((marker - VBYTE_MARKER_3BYTE) << 16) + VBYTE_THRESHOLD_3BYTE;
        return in + 2;
    }

    // 4+ byte encoding: 0xFC = 3 bytes, 0xFD = 4 bytes
    if (marker == VBYTE_MARKER_4PLUS)
    {
        x = loadU32Fast(in) & 0xFFFFFFu;
        return in + 3;
    }

    x = loadU32Fast(in);
    return in + 4;
}

} // namespace

// Encode array of n uint32_t values using adaptive variable-byte encoding.
//
// Strategy: Encodes all values using variable-byte format, then checks if compression
// is worthwhile. If compressed size doesn't provide significant savings (at least 32 bytes),
// stores uncompressed data instead to avoid decode overhead.
//
// Motivation:
// - Variable-byte decoding has computational cost (bit extraction, branching)
// - If compression saves only a few bytes, the decode overhead outweighs space savings
// - 32-byte threshold ensures compression is used only when benefit is significant
// - Uncompressed path uses fast memcpy, avoiding decode overhead for poorly-compressible data
//
// Format:
// - Compressed: [vbyte-encoded values...] (each value encoded by vbPut32)
// - Uncompressed: [0xFF escape marker][raw uint32_t values...]
//
// Returns: Pointer to end of encoded data
unsigned char * vbEnc32(const uint32_t * in, unsigned n, unsigned char * out)
{
    // Encode all values using variable-byte encoding
    // Note: vbPut32 takes 'unsigned char * &', so op advances automatically
    unsigned char * op = out;
    for (unsigned i = 0; i < n; ++i)
        vbPut32(op, in[i]);

    // Adaptive compression decision: Check if encoding provides significant space savings
    //
    // Condition breakdown:
    //   op             = pointer to end of encoded data (advanced by vbPut32 calls)
    //   op - out       = actual compressed size in bytes
    //   op + 32        = compressed size + 32-byte threshold
    //   out + n * 4    = pointer to end position if stored uncompressed
    //
    // If (op + 32 > out + n * 4), then:
    //   (op - out) + 32 > n * 4
    //   compressed_size + 32 > original_size
    //   compressed_size > original_size - 32
    //
    // This means compression saves less than 32 bytes, so use uncompressed format instead
    if (op + 32 > out + n * sizeof(uint32_t))
    {
        // Compression ineffective: Store uncompressed data with escape marker
        // Format: [0xFF][uint32_t values...]
        // Benefit: Decoder can use fast memcpy instead of expensive vbGet32 loop
        *out = VBYTE_ESCAPE_UNCOMPRESSED;
        copyU32ArrayToLe(out + 1, in, n);
        return out + 1 + n * sizeof(uint32_t);
    }

    // Compression effective: Return pointer to end of variable-byte encoded data
    return op;
}

// Decode array of n uint32_t values from adaptive variable-byte encoding.
//
// Handles two formats produced by vbEnc32:
// 1. Compressed format: Variable-byte encoded values (decoded by vbGet32)
// 2. Uncompressed format: [0xFF escape marker][raw uint32_t array]
//
// Format detection:
// - First byte == 0xFF: Uncompressed format, use fast memcpy
// - First byte != 0xFF: Compressed format, decode each value with vbGet32
//
// Performance notes:
// - Uncompressed path: Single memcpy, very fast (~memory bandwidth speed)
// - Compressed path: Loop with vbGet32 calls, slower but saves space
// - The 32-byte threshold in vbEnc32 ensures we only use compression when worthwhile
//
// Returns: Pointer to end of consumed input data
unsigned char * vbDec32(unsigned char * in, unsigned n, uint32_t * out)
{
    // Check format by examining first byte
    if (*in == VBYTE_ESCAPE_UNCOMPRESSED)
    {
        // Uncompressed format: [0xFF][n * uint32_t values...]
        // Fast path: Direct memory copy, no decoding needed
        // This is why vbEnc32 uses uncompressed for poorly-compressible data
        copyU32ArrayFromLe(out, in + 1, n);
        return in + 1 + n * sizeof(uint32_t);
    }

    // Compressed format: Variable-byte encoded values
    // Decode each value sequentially using vbGet32Inline
    // Note: vbGet32Inline returns updated input pointer after consuming variable bytes
    unsigned char * ip = in;
    for (unsigned i = 0; i < n; ++i)
    {
        ip = vbGet32Inline(ip, out[i]);
    }

    return ip;
}

// P4 bit width selection - Determines optimal bit width and exception handling strategy.
//
// P4 (Patched Frame-of-Reference) encoding uses a base bit width 'b' to encode most values,
// with "exceptions" (values needing more bits) handled separately. This function analyzes
// the input data to find the encoding strategy that minimizes total compressed size.
//
// Three encoding strategies are considered:
// 1. Simple bitpacking: All values encoded with bit width 'b', no exceptions (exception_bits = 0)
// 2. Exception encoding: Base bit width 'b' + variable-byte encoded exceptions (exception_bits = 33)
// 3. Bitwise patching: Base bit width 'b' + fixed-width patch bits for exceptions
//
// Algorithm overview:
// - Count bit width distribution (how many values need each bit width)
// - For each candidate base bit width, estimate compressed size:
//   * Simple: pad8(n * b) bytes
//   * With exceptions: pad8(n * b) + vbyte_size(exceptions) bytes
//   * With patches: pad8(n * b) + pad8(exception_count) + pad8(exception_count * patch_bits) bytes
// - Return the (b, exception_bits) pair that minimizes compressed size
//
// Parameters:
//   in: Input array of n uint32_t values
//   n: Number of elements (typically 1-256 for P4 block encoding)
//   out_exception_bits: Output parameter for exception handling strategy
//     - 0: No exceptions (simple bitpacking with width b)
//     - 1-31: Bitwise patching (exception_bits = original_max_bits - chosen_base_bits)
//     - 33 (32+1): Variable-byte exception encoding
//     - 34 (32+2): All values equal (constant block optimization)
//
// Returns: Optimal base bit width 'b' (0-32)
//
// Note: Uses negative array indexing trick for vbyte accumulators (vb array centered at offset)
unsigned p4Bits32(const uint32_t * in, unsigned n, unsigned * out_exception_bits)
{
    // Phase 1: Fast scan for special cases (zeros, constant)
    uint32_t bitwise_or = 0;
    const uint32_t first_value = in[0];
    unsigned equal_count = 0;

    for (unsigned i = 0; i < n; ++i)
    {
        bitwise_or |= in[i];
        equal_count += (in[i] == first_value);
    }

    // Special case: All zeros - early exit
    if (bitwise_or == 0u)
    {
        *out_exception_bits = 0u;
        return 0u;
    }

    unsigned max_bits = bitWidth32(bitwise_or);

    // Special case: All values equal (constant block) - early exit
    if (equal_count == n)
    {
        *out_exception_bits = MAX_BITS + 2u;
        return max_bits;
    }

    // Phase 2: Build bit width histogram (only for non-trivial cases)
    unsigned bit_width_count[MAX_BITS + 8u] = {0};

    for (unsigned i = 0; i < n; ++i)
    {
        ++bit_width_count[bitWidth32(in[i])];
    }

    // Variable-byte size accumulators
    unsigned vbyte_accumulator_storage[MAX_BITS * 2u + 64u + 16u] = {0};
    unsigned * vbyte_accumulator = vbyte_accumulator_storage + MAX_BITS + 16u;

    // Phase 2: Find optimal base bit width by evaluating all candidates

    unsigned optimal_base_bits = max_bits;
    unsigned exception_count = bit_width_count[max_bits]; // Values needing > base_bits
    unsigned min_size = pad8(n * max_bits) + 1u; // Start with simple encoding

    // Lambda: Update variable-byte size accumulators
    // For a given bit width difference, accumulate vbyte encoding sizes
    // The magic numbers (7, 15, 19, 25) correspond to vbyte encoding breakpoints:
    // - Difference <= 7 bits: 1 byte vbyte
    // - Difference <= 15 bits: 2 bytes vbyte
    // - Difference <= 19 bits: 3 bytes vbyte
    // - Difference <= 25 bits: 4 bytes vbyte
    auto update_vbyte_accumulator = [&vbyte_accumulator](unsigned count, unsigned bits)
    {
        vbyte_accumulator[static_cast<int>(bits) - 7] += count; // 1 byte vbyte threshold
        vbyte_accumulator[static_cast<int>(bits) - 15] += count * 2u; // 2 byte vbyte threshold
        vbyte_accumulator[static_cast<int>(bits) - 19] += count * 3u; // 3 byte vbyte threshold
        vbyte_accumulator[static_cast<int>(bits) - 25] += count * 4u; // 4 byte vbyte threshold
    };

    unsigned vbyte_size_accumulator = exception_count;
    update_vbyte_accumulator(exception_count, max_bits);

    unsigned use_vbyte_exceptions = 0; // Flag: 0 = bitwise patching, 1 = vbyte exceptions
    const unsigned bitmap_bytes = pad8(n); // Bytes needed for exception bitmap

    // Try each candidate base bit width from (max_bits-1) down to 0
    // Use unsigned directly and check underflow via explicit break
    unsigned base_bits = max_bits - 1u;
    while (true)
    {
        const unsigned patch_bits = max_bits - base_bits; // Extra bits needed for exceptions

        // Strategy 1: Variable-byte exception encoding
        const unsigned vbyte_size = pad8(n * base_bits) + 2u + exception_count + vbyte_size_accumulator;

        // Strategy 2: Bitwise patching
        const unsigned patching_size = pad8(n * base_bits) + 2u + bitmap_bytes + pad8(exception_count * patch_bits);

        // Update optimal choice - combined conditional for patching
        if (patching_size < min_size && patching_size <= vbyte_size)
        {
            min_size = patching_size;
            optimal_base_bits = base_bits;
            use_vbyte_exceptions = 0;
        }
        else if (vbyte_size < min_size)
        {
            min_size = vbyte_size;
            optimal_base_bits = base_bits;
            use_vbyte_exceptions = 1;
        }

        // Check if this is the last iteration (base_bits == 0)
        if (base_bits == 0)
            break;

        // Update state for next iteration
        exception_count += bit_width_count[base_bits];
        vbyte_size_accumulator += bit_width_count[base_bits] + vbyte_accumulator[static_cast<int>(base_bits)];
        update_vbyte_accumulator(bit_width_count[base_bits], base_bits);

        --base_bits;
    }

    // Return results
    // exception_bits encoding:
    // - 0: No exceptions (simple bitpacking)
    // - 1-31: Bitwise patching with patch_bits = exception_bits
    //   (max is 31 when max_bits=32, optimal_base_bits=1)
    // - 32: Reserved/unused (separates patching from special encodings)
    // - 33: Variable-byte exception encoding
    // - 34: Constant block (handled in p4Bits32 earlier, before this point)
    *out_exception_bits = use_vbyte_exceptions ? (MAX_BITS + 1u) : (max_bits - optimal_base_bits);
    return optimal_base_bits;
}

// Write P4 encoding header based on bit width (b) and exception handling strategy (bx)
//
// Header encoding format:
// 1. Simple bitpacking (bx = 0):
//    [b] - Single byte with base bit width
//
// 2. Bitwise patching (bx = 1-31):
//    [0x80 | b][bx] - Two bytes: flag + base_bits, then exception_bits
//
// 3. Variable-byte exceptions (bx = 33):
//    [0x40 | b] - Single byte with flag (0x40) + base_bits
//
// 4. Constant block (bx = 34):
//    [0xC0 | b] - Single byte with flag (0xC0) + bit_width
//
// The bx parameter values map to exception strategies:
// - 0: No exceptions
// - 1-31: Bitwise patching with patch_bits = bx
// - 32+1 (33): Variable-byte exception encoding
// - 32+2 (34): All values equal (constant block)
void writeHeader(unsigned char *& out, unsigned b, unsigned bx)
{
    if (bx == 0u)
    {
        // Simple bitpacking: just write the bit width
        *out++ = static_cast<unsigned char>(b);
    }
    else if (bx <= MAX_BITS)
    {
        // Bitwise patching: 0x80 flag + base_bits, then exception_bits
        *out++ = static_cast<unsigned char>(0x80u | b);
        *out++ = static_cast<unsigned char>(bx);
    }
    else
    {
        // Variable-byte exceptions (bx=33): 0x40 flag
        // Constant block (bx=34): 0xC0 flag
        const unsigned flag = (bx == MAX_BITS + 1u) ? 0x40u : 0xC0u;
        *out++ = static_cast<unsigned char>(flag | b);
    }
}

} // namespace turbopfor::scalar::detail
