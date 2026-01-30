#pragma once

#include <cstdint>
#include <cstring>

namespace turbopfor::scalar::detail
{

#if defined(__GNUC__) || defined(__clang__)
#    define TURBOPFOR_ALWAYS_INLINE __attribute__((always_inline)) inline
#    define TURBOPFOR_NOINLINE __attribute__((noinline))
#    define TURBOPFOR_LIKELY(x) __builtin_expect(!!(x), 1)
#    define TURBOPFOR_UNLIKELY(x) __builtin_expect(!!(x), 0)
using U64Alias = uint64_t __attribute__((__may_alias__));
using U32Alias = uint32_t __attribute__((__may_alias__));
using U16Alias = uint16_t __attribute__((__may_alias__));
#else
#    define TURBOPFOR_ALWAYS_INLINE inline
#    define TURBOPFOR_NOINLINE
#    define TURBOPFOR_LIKELY(x) (x)
#    define TURBOPFOR_UNLIKELY(x) (x)
using U64Alias = uint64_t;
using U32Alias = uint32_t;
using U16Alias = uint16_t;
#endif

// Endianness detection
// TurboPFor format uses little-endian byte order for all multi-byte values.
// On big-endian platforms, we need to byte-swap when loading/storing.
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
#    if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#        define TURBOPFOR_BIG_ENDIAN 1
#    else
#        define TURBOPFOR_BIG_ENDIAN 0
#    endif
#elif defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__THUMBEB__) || defined(__AARCH64EB__) || defined(_MIPSEB) \
    || defined(__MIPSEB) || defined(__MIPSEB__)
#    define TURBOPFOR_BIG_ENDIAN 1
#else
#    define TURBOPFOR_BIG_ENDIAN 0
#endif

// Byte swap functions for endianness conversion
TURBOPFOR_ALWAYS_INLINE constexpr uint16_t byteSwap16(uint16_t v)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap16(v);
#else
    return static_cast<uint16_t>((v >> 8) | (v << 8));
#endif
}

TURBOPFOR_ALWAYS_INLINE constexpr uint32_t byteSwap32(uint32_t v)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(v);
#else
    return ((v >> 24) & 0x000000FFu) | ((v >> 8) & 0x0000FF00u) | ((v << 8) & 0x00FF0000u) | ((v << 24) & 0xFF000000u);
#endif
}

TURBOPFOR_ALWAYS_INLINE constexpr uint64_t byteSwap64(uint64_t v)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(v);
#else
    return ((v >> 56) & 0x00000000000000FFull) | ((v >> 40) & 0x000000000000FF00ull) | ((v >> 24) & 0x0000000000FF0000ull)
        | ((v >> 8) & 0x00000000FF000000ull) | ((v << 8) & 0x000000FF00000000ull) | ((v << 24) & 0x0000FF0000000000ull)
        | ((v << 40) & 0x00FF000000000000ull) | ((v << 56) & 0xFF00000000000000ull);
#endif
}

// Convert from little-endian to native (no-op on little-endian platforms)
TURBOPFOR_ALWAYS_INLINE constexpr uint16_t leToNative16(uint16_t v)
{
#if TURBOPFOR_BIG_ENDIAN
    return byteSwap16(v);
#else
    return v;
#endif
}

TURBOPFOR_ALWAYS_INLINE constexpr uint32_t leToNative32(uint32_t v)
{
#if TURBOPFOR_BIG_ENDIAN
    return byteSwap32(v);
#else
    return v;
#endif
}

TURBOPFOR_ALWAYS_INLINE constexpr uint64_t leToNative64(uint64_t v)
{
#if TURBOPFOR_BIG_ENDIAN
    return byteSwap64(v);
#else
    return v;
#endif
}

// Convert from native to little-endian (same as leToNative, symmetric operation)
TURBOPFOR_ALWAYS_INLINE constexpr uint16_t nativeToLe16(uint16_t v)
{
    return leToNative16(v);
}

TURBOPFOR_ALWAYS_INLINE constexpr uint32_t nativeToLe32(uint32_t v)
{
    return leToNative32(v);
}

TURBOPFOR_ALWAYS_INLINE constexpr uint64_t nativeToLe64(uint64_t v)
{
    return leToNative64(v);
}

constexpr unsigned MAX_BITS = 32; // Maximum bits in uint32_t
constexpr unsigned MAX_VALUES = 256; // Maximum values per block

/// Round up to multiple of 8 bits
constexpr unsigned pad8(unsigned x)
{
    return (x + 7u) / 8u;
}

/// Bit scan reverse for 32-bit integer (returns highest set bit position + 1, or 0 if x is 0)
/// Returns value in range [0, 32]
///
/// Uses inline assembly on x86 for optimal codegen - the bsr instruction leaves the destination
/// unchanged when the source is 0, so initializing b=-1 gives us b+1=0 for x=0 without branching.
///
/// Returns unsigned (not uint8_t) to avoid zero-extension instructions (movzbl) when used as array index.
inline unsigned bsr32(uint32_t x)
{
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    int b = -1;
    asm("bsrl %1,%0" : "+r"(b) : "rm"(x));
    return static_cast<unsigned>(b + 1);
#elif defined(__GNUC__) || defined(__clang__)
    return x ? (32u - static_cast<unsigned>(__builtin_clz(x))) : 0u;
#else
    if (!x)
        return 0u;
    unsigned b = 0u;
    while (x)
    {
        ++b;
        x >>= 1u;
    }
    return b;
#endif
}

/// Bit width using lzcnt instruction (faster than bsr on modern CPUs)
/// Returns highest set bit position + 1, or 0 if x is 0
/// lzcnt is preferred when available because it handles x=0 natively (returns 32)
inline unsigned bitWidth32(uint32_t x)
{
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    unsigned lz;
    // lzcnt returns 32 for x=0, so 32-32=0 which is correct
    asm("lzcntl %1,%0" : "=r"(lz) : "rm"(x) : "cc");
    return 32u - lz;
#else
    return x ? (32u - static_cast<unsigned>(__builtin_clz(x))) : 0u;
#endif
}

/// Create a mask with b bits set
/// Parameter b: number of bits to set (0-32)
/// Returns: bitmask with b lowest bits set
inline uint32_t maskBits(unsigned b)
{
    if (b >= 32u)
        return 0xFFFFFFFFu;
    return b == 0u ? 0u : ((1u << b) - 1u);
}

/// Load unaligned 32-bit little-endian value and convert to native
inline uint32_t loadU32(const unsigned char * in)
{
    uint32_t v;
    memcpy(&v, in, sizeof(v));
    return leToNative32(v);
}

/// Load unaligned 16-bit little-endian value and convert to native
inline uint16_t loadU16(const unsigned char * in)
{
    uint16_t v;
    memcpy(&v, in, sizeof(v));
    return leToNative16(v);
}

/// Store native 32-bit value as unaligned little-endian
inline void storeU32(unsigned char * out, uint32_t v)
{
    v = nativeToLe32(v);
    memcpy(out, &v, sizeof(v));
}

/// Store native 16-bit value as unaligned little-endian
inline void storeU16(unsigned char * out, uint16_t v)
{
    v = nativeToLe16(v);
    memcpy(out, &v, sizeof(v));
}

/// Load unaligned 64-bit little-endian value and convert to native
inline uint64_t loadU64(const unsigned char * in)
{
    uint64_t v;
    memcpy(&v, in, sizeof(v));
    return leToNative64(v);
}

/// Store native 64-bit value as unaligned little-endian
inline void storeU64(unsigned char * out, uint64_t v)
{
    v = nativeToLe64(v);
    memcpy(out, &v, sizeof(v));
}

/// Fast unaligned loads/stores with little-endian conversion
/// On x86 (always little-endian), uses direct pointer access for speed.
/// On other architectures, uses memcpy + byte swap if needed.
inline uint64_t loadU64Fast(const unsigned char * in)
{
#if defined(__i386__) || defined(__x86_64__)
    // x86 is always little-endian, direct access is safe and fast
    return *reinterpret_cast<const U64Alias *>(in);
#else
    return loadU64(in);
#endif
}

inline void storeU64Fast(unsigned char * out, uint64_t v)
{
#if defined(__i386__) || defined(__x86_64__)
    // x86 is always little-endian, direct access is safe and fast
    *reinterpret_cast<U64Alias *>(out) = v;
#else
    storeU64(out, v);
#endif
}

inline uint32_t loadU32Fast(const unsigned char * in)
{
#if defined(__i386__) || defined(__x86_64__)
    // x86 is always little-endian, direct access is safe and fast
    return *reinterpret_cast<const U32Alias *>(in);
#else
    return loadU32(in);
#endif
}

inline void storeU32Fast(unsigned char * out, uint32_t v)
{
#if defined(__i386__) || defined(__x86_64__)
    // x86 is always little-endian, direct access is safe and fast
    *reinterpret_cast<U32Alias *>(out) = v;
#else
    storeU32(out, v);
#endif
}

inline uint16_t loadU16Fast(const unsigned char * in)
{
#if defined(__i386__) || defined(__x86_64__)
    // x86 is always little-endian, direct access is safe and fast
    return *reinterpret_cast<const U16Alias *>(in);
#else
    return loadU16(in);
#endif
}

inline uint32_t loadU24(const unsigned char * in)
{
    // Load 3 bytes as little-endian 24-bit value
    // Byte 0 is lowest, byte 2 is highest
    // On x86, use 16-bit load for better performance (fewer load ops)
#if defined(__i386__) || defined(__x86_64__)
    return static_cast<uint32_t>(*reinterpret_cast<const U16Alias *>(in)) | (static_cast<uint32_t>(in[2]) << 16);
#else
    return static_cast<uint32_t>(in[0]) | (static_cast<uint32_t>(in[1]) << 8) | (static_cast<uint32_t>(in[2]) << 16);
#endif
}

inline void storeU16Fast(unsigned char * out, uint16_t v)
{
#if defined(__i386__) || defined(__x86_64__)
    // x86 is always little-endian, direct access is safe and fast
    *reinterpret_cast<U16Alias *>(out) = v;
#else
    storeU16(out, v);
#endif
}

/// Copy n uint32_t values from native array to little-endian byte stream
/// On little-endian platforms, this is equivalent to memcpy
/// On big-endian platforms, each value is byte-swapped
TURBOPFOR_ALWAYS_INLINE void copyU32ArrayToLe(unsigned char * out, const uint32_t * in, unsigned n)
{
#if TURBOPFOR_BIG_ENDIAN
    for (unsigned i = 0; i < n; ++i)
    {
        storeU32Fast(out, in[i]);
        out += sizeof(uint32_t);
    }
#else
    memcpy(out, in, n * sizeof(uint32_t));
#endif
}

/// Copy n uint32_t values from little-endian byte stream to native array
/// On little-endian platforms, this is equivalent to memcpy
/// On big-endian platforms, each value is byte-swapped
TURBOPFOR_ALWAYS_INLINE void copyU32ArrayFromLe(uint32_t * out, const unsigned char * in, unsigned n)
{
#if TURBOPFOR_BIG_ENDIAN
    for (unsigned i = 0; i < n; ++i)
    {
        out[i] = loadU32Fast(in);
        in += sizeof(uint32_t);
    }
#else
    memcpy(out, in, n * sizeof(uint32_t));
#endif
}

/// Scalar bit packing/unpacking (horizontal format)
unsigned char * bitpack32Scalar(const uint32_t * in, unsigned n, unsigned char * out, unsigned b);
unsigned char * bitunpack32Scalar(unsigned char * in, unsigned n, uint32_t * out, unsigned b);

/// Fused unpack + delta1 decode (much faster than separate unpack + delta)
unsigned char * bitunpackd1_32Scalar(unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);

/// 128v32 bitpacking: 4-lane interleaved horizontal packing for 128 elements
/// Each output lane k contains bits from values: v[k], v[k+4], v[k+8], ...
unsigned char * bitpack128v32Scalar(const uint32_t * in, unsigned char * out, unsigned b);
unsigned char * bitunpack128v32Scalar(unsigned char * in, uint32_t * out, unsigned b);

/// 256v32 bitpacking: 8-lane interleaved horizontal packing for 256 elements
/// Each output lane k contains bits from values: v[k], v[k+8], v[k+16], ...
unsigned char * bitpack256v32Scalar(const uint32_t * in, unsigned char * out, unsigned b);
unsigned char * bitunpack256v32Scalar(unsigned char * in, uint32_t * out, unsigned b);

// Variable-byte encoding constants (matching TurboPFor vlcbyte.h scheme)
// These are used for compact encoding of small integer values.
constexpr unsigned VBYTE_ESCAPE_UNCOMPRESSED = 0xFFu; // Escape: uncompressed data follows
constexpr unsigned VBYTE_MARKER_MAX = 0xFDu; // Maximum marker used by vbPut32
constexpr unsigned VBYTE_MARKER_4PLUS = 0xFCu; // First marker for 4+ byte encoding
constexpr unsigned VBYTE_MARKER_3BYTE = 0xDCu; // First marker for 3-byte encoding
constexpr unsigned VBYTE_MARKER_2BYTE = 0x9Cu; // First marker for 2-byte encoding
constexpr unsigned VBYTE_THRESHOLD_2BYTE = 156u; // Values >= 156 need 2+ bytes
constexpr unsigned VBYTE_THRESHOLD_3BYTE = 16540u; // Values >= 16540 need 3+ bytes
constexpr unsigned VBYTE_THRESHOLD_4PLUS = 2113692u; // Values >= 2113692 need 4+ bytes

/// Inline single-value variable-byte decoder (matches TurboPFor _vbget32 macro)
/// Uses likely() hints for optimal branch prediction since small values are most common.
TURBOPFOR_ALWAYS_INLINE unsigned char * vbGet32Inline(unsigned char * in, uint32_t & x)
{
    const unsigned marker = *in++;

    if (TURBOPFOR_LIKELY(marker < VBYTE_MARKER_2BYTE)) // marker < 0x9C (most common)
    {
        // 1-byte encoding: value stored directly in marker
        x = marker;
        return in;
    }

    if (TURBOPFOR_LIKELY(marker < VBYTE_MARKER_3BYTE)) // marker < 0xDC
    {
        // 2-byte encoding: reconstruct value from marker + 1 data byte
        const unsigned data_byte = *in++;
        x = ((marker - VBYTE_MARKER_2BYTE) << 8) + data_byte + VBYTE_THRESHOLD_2BYTE;
        return in;
    }

    if (TURBOPFOR_LIKELY(marker < VBYTE_MARKER_4PLUS)) // marker < 0xFC
    {
        // 3-byte encoding
        const unsigned low16 = loadU16Fast(in);
        x = low16 + ((marker - VBYTE_MARKER_3BYTE) << 16) + VBYTE_THRESHOLD_3BYTE;
        return in + 2;
    }

    // 4+ byte encoding: 0xFC = 3 bytes, 0xFD = 4 bytes (rare)
    if (marker == VBYTE_MARKER_4PLUS)
    {
        x = loadU32Fast(in) & 0xFFFFFFu;
        return in + 3;
    }

    x = loadU32Fast(in);
    return in + 4;
}

/// Variable-byte encoding/decoding
unsigned char * vbEnc32(const uint32_t * in, unsigned n, unsigned char * out);
unsigned char * vbDec32(unsigned char * in, unsigned n, uint32_t * out);

/// P4 bit width selection
unsigned p4Bits32(const uint32_t * in, unsigned n, unsigned * out_exception_bits);

/// Write P4 header
void writeHeader(unsigned char *& out, unsigned b, unsigned bx);

} // namespace turbopfor::scalar::detail
