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

constexpr unsigned MAX_BITS_32 = 32; // Maximum bits in uint32_t
constexpr unsigned MAX_BITS_64 = 64; // Maximum bits in uint64_t
constexpr unsigned MAX_BITS = 32; // Default MAX_BITS (32-bit) for backwards compatibility
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

/// Bit scan reverse for 64-bit integer (returns highest set bit position + 1, or 0 if x is 0)
/// Returns value in range [0, 64]
inline unsigned bsr64(uint64_t x)
{
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    int64_t b = -1;
    asm("bsrq %1,%0" : "+r"(b) : "rm"(x));
    return static_cast<unsigned>(b + 1);
#elif defined(__GNUC__) || defined(__clang__)
    return x ? (64u - static_cast<unsigned>(__builtin_clzll(x))) : 0u;
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

/// Bit width using lzcnt instruction for 64-bit values
/// Returns highest set bit position + 1, or 0 if x is 0
inline unsigned bitWidth64(uint64_t x)
{
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    unsigned long long lz;
    // lzcnt returns 64 for x=0, so 64-64=0 which is correct
    asm("lzcntq %1,%0" : "=r"(lz) : "rm"(x) : "cc");
    return 64u - static_cast<unsigned>(lz);
#else
    return x ? (64u - static_cast<unsigned>(__builtin_clzll(x))) : 0u;
#endif
}

/// Create a mask with b bits set (32-bit version)
/// Parameter b: number of bits to set (0-32)
/// Returns: bitmask with b lowest bits set
inline uint32_t maskBits(unsigned b)
{
    if (b >= 32u)
        return 0xFFFFFFFFu;
    return b == 0u ? 0u : ((1u << b) - 1u);
}

/// Create a mask with b bits set (64-bit version)
/// Parameter b: number of bits to set (0-64)
/// Returns: bitmask with b lowest bits set
inline uint64_t maskBits64(unsigned b)
{
    if (b >= 64u)
        return 0xFFFFFFFFFFFFFFFFull;
    return b == 0u ? 0ull : ((1ull << b) - 1ull);
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

/// Copy n uint64_t values from native array to little-endian byte stream
/// On little-endian platforms, this is equivalent to memcpy
/// On big-endian platforms, each value is byte-swapped
TURBOPFOR_ALWAYS_INLINE void copyU64ArrayToLe(unsigned char * out, const uint64_t * in, unsigned n)
{
#if TURBOPFOR_BIG_ENDIAN
    for (unsigned i = 0; i < n; ++i)
    {
        storeU64Fast(out, in[i]);
        out += sizeof(uint64_t);
    }
#else
    memcpy(out, in, n * sizeof(uint64_t));
#endif
}

/// Copy n uint64_t values from little-endian byte stream to native array
/// On little-endian platforms, this is equivalent to memcpy
/// On big-endian platforms, each value is byte-swapped
TURBOPFOR_ALWAYS_INLINE void copyU64ArrayFromLe(uint64_t * out, const unsigned char * in, unsigned n)
{
#if TURBOPFOR_BIG_ENDIAN
    for (unsigned i = 0; i < n; ++i)
    {
        out[i] = loadU64Fast(in);
        in += sizeof(uint64_t);
    }
#else
    memcpy(out, in, n * sizeof(uint64_t));
#endif
}

/// Constexpr utilities for block size selection (shared by 32-bit and 64-bit bitpack/unpack)
constexpr unsigned gcd_u32(unsigned a, unsigned b)
{
    return b == 0u ? a : gcd_u32(b, a % b);
}

constexpr unsigned word_count_for(unsigned b, unsigned n)
{
    return (n * b + 63u) / 64u;
}

constexpr unsigned max_words_for_block()
{
    return 32u;
}

constexpr unsigned choose_block_size(unsigned b, unsigned n)
{
    if (n == 0u)
        return 0u;
    const unsigned g = gcd_u32(64u, b);
    unsigned period = 64u / g;
    unsigned max_words = max_words_for_block();

    unsigned k = period;
    while (k > 1u && k > n)
        k >>= 1u;
    if (k > n)
        k = 1u;

    for (;;)
    {
        if (word_count_for(b, k) <= max_words && ((k * b) % 8u == 0u))
            return k;
        if (k == 1u)
            break;
        k >>= 1u;
    }
    return n;
}

/// Store partial bytes at the end of output (1-7 bytes from a 64-bit word)
/// Used by bitpack templates when the last word doesn't fill a full 8 bytes
template <unsigned R>
static TURBOPFOR_ALWAYS_INLINE void store_partial(unsigned char *& op, uint64_t v)
{
    static_assert(R >= 1 && R <= 7);
    if constexpr (R >= 4)
    {
        storeU32Fast(op, static_cast<uint32_t>(v));
        op += 4u;
        if constexpr (R >= 6)
        {
            storeU16Fast(op, static_cast<uint16_t>(v >> 32));
            op += 2u;
            if constexpr (R == 7)
            {
                *op++ = static_cast<unsigned char>(v >> 48);
            }
        }
        else if constexpr (R == 5)
        {
            *op++ = static_cast<unsigned char>(v >> 32);
        }
    }
    else if constexpr (R >= 2)
    {
        storeU16Fast(op, static_cast<uint16_t>(v));
        op += 2u;
        if constexpr (R == 3)
        {
            *op++ = static_cast<unsigned char>(v >> 16);
        }
    }
    else
    {
        *op++ = static_cast<unsigned char>(v);
    }
}

/// Load partial bytes from input (1-7 bytes into a 64-bit word)
/// Used by bitunpack templates when the last word doesn't fill a full 8 bytes
template <unsigned R>
static TURBOPFOR_ALWAYS_INLINE uint64_t load_partial(const unsigned char *& ip)
{
    static_assert(R >= 1 && R <= 7);
    uint64_t v = 0;
    if constexpr (R >= 4)
    {
        v |= static_cast<uint64_t>(loadU32Fast(ip));
        ip += 4u;
        if constexpr (R >= 6)
        {
            v |= static_cast<uint64_t>(loadU16Fast(ip)) << 32;
            ip += 2u;
            if constexpr (R == 7)
            {
                v |= static_cast<uint64_t>(ip[0]) << 48;
                ip += 1u;
            }
        }
        else if constexpr (R == 5)
        {
            v |= static_cast<uint64_t>(ip[0]) << 32;
            ip += 1u;
        }
    }
    else if constexpr (R >= 2)
    {
        v |= static_cast<uint64_t>(loadU16Fast(ip));
        ip += 2u;
        if constexpr (R == 3)
        {
            v |= static_cast<uint64_t>(ip[0]) << 16;
            ip += 1u;
        }
    }
    else
    {
        v |= static_cast<uint64_t>(ip[0]);
        ip += 1u;
    }
    return v;
}

/// Scalar bit packing/unpacking (horizontal format)
unsigned char * bitpack32Scalar(const uint32_t * in, unsigned n, unsigned char * out, unsigned b);
const unsigned char * bitunpack32Scalar(const unsigned char * in, unsigned n, uint32_t * out, unsigned b);

/// Fused unpack + delta1 decode (much faster than separate unpack + delta)
const unsigned char * bitunpackd1_32Scalar(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);

/// 128v32 bitpacking: 4-lane interleaved horizontal packing for 128 elements
/// Each output lane k contains bits from values: v[k], v[k+4], v[k+8], ...
unsigned char * bitpack128v32Scalar(const uint32_t * in, unsigned char * out, unsigned b);
const unsigned char * bitunpack128v32Scalar(const unsigned char * in, uint32_t * out, unsigned b);

/// 256v32 bitpacking: 8-lane interleaved horizontal packing for 256 elements
/// Each output lane k contains bits from values: v[k], v[k+8], v[k+16], ...
unsigned char * bitpack256v32Scalar(const uint32_t * in, unsigned char * out, unsigned b);
const unsigned char * bitunpack256v32Scalar(const unsigned char * in, uint32_t * out, unsigned b);

// Variable-byte encoding constants (matching TurboPFor vlcbyte.h scheme)
//
// TurboPFor uses VB_MAX = 0xFD (not 0xFF), reserving:
//   0xFE = all-zeros marker (used in delta encoding)
//   0xFF = overflow/uncompressed escape
//
// The encoding uses _vbput(op, x, VBSIZE, VB_MAX=0xFD, VBB2=6, VBB3=5, ;)
// with thresholds computed as:
//   _vbba3(VBSIZE, 0xFD) = 0xFD - (VBSIZE/8 - 3)
//   _vbba2(VBSIZE, 0xFD, 5) = _vbba3 - 32
//   _vbo1(VBSIZE, 0xFD, 6, 5) = _vbba2 - 64
//   _vbo2 = _vbo1 + 16384
//   _vbo3 = _vbo2 + 2097152
//
// 32-bit constants (VBSIZE=32):
constexpr unsigned VBYTE_ESCAPE_UNCOMPRESSED = 0xFFu; // Escape: uncompressed data follows
constexpr unsigned VBYTE_MARKER_MAX = 0xFDu; // VB_MAX: maximum marker used by vbPut32
constexpr unsigned VBYTE_MARKER_4PLUS = 0xFCu; // _vbba3(32,0xFD) = 0xFD-(4-3) = 0xFC
constexpr unsigned VBYTE_MARKER_3BYTE = 0xDCu; // _vbba2(32,0xFD,5) = 0xFC-32 = 0xDC
constexpr unsigned VBYTE_MARKER_2BYTE = 0x9Cu; // _vbo1(32,0xFD,6,5) = 0xDC-64 = 0x9C
constexpr unsigned VBYTE_THRESHOLD_2BYTE = 156u; // _vbo1 = 0x9C = 156
constexpr unsigned VBYTE_THRESHOLD_3BYTE = 16540u; // _vbo2 = 156 + 16384 = 16540
constexpr unsigned VBYTE_THRESHOLD_4PLUS = 2113692u; // _vbo3 = 16540 + 2097152 = 2113692

// 64-bit constants (VBSIZE=64):
// Raw markers use 0xF8..0xFD for 3..8 byte raw values
constexpr unsigned VBYTE64_MARKER_RAW = 0xF8u; // _vbba3(64,0xFD) = 0xFD-(8-3) = 0xF8
constexpr unsigned VBYTE64_MARKER_3BYTE = 0xD8u; // _vbba2(64,0xFD,5) = 0xF8-32 = 0xD8
constexpr unsigned VBYTE64_MARKER_2BYTE = 0x98u; // _vbo1(64,0xFD,6,5) = 0xD8-64 = 0x98
constexpr unsigned VBYTE64_THRESHOLD_2BYTE = 152u; // _vbo1 = 0x98 = 152
constexpr unsigned VBYTE64_THRESHOLD_3BYTE = 16536u; // _vbo2 = 152 + 16384 = 16536
constexpr unsigned VBYTE64_THRESHOLD_RAW = 2113688u; // _vbo3 = 16536 + 2097152 = 2113688

/// Inline single-value variable-byte decoder (matches TurboPFor _vbget32 macro)
/// Uses likely() hints for optimal branch prediction since small values are most common.
TURBOPFOR_ALWAYS_INLINE const unsigned char * vbGet32Inline(const unsigned char * in, uint32_t & x)
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

/// Variable-byte encoding/decoding (32-bit)
unsigned char * vbEnc32(const uint32_t * in, unsigned n, unsigned char * out);
const unsigned char * vbDec32(const unsigned char * in, unsigned n, uint32_t * out);

/// Inline single-value variable-byte decoder for 64-bit values (matches TurboPFor _vbget64 macro)
///
/// Same structure as 32-bit but with shifted thresholds (VB_MAX=0xFD, VBSIZE=64):
/// - [0x00..0x97]: 1-byte encoding, value = marker
/// - [0x98..0xD7]: 2-byte encoding, value = 152 + decode(marker, data_byte)
/// - [0xD8..0xF7]: 3-byte encoding, value = 16536 + decode(marker, data16)
/// - [0xF8..0xFD]: raw encoding, (marker - 0xF8 + 3) raw bytes follow
TURBOPFOR_ALWAYS_INLINE const unsigned char * vbGet64Inline(const unsigned char * in, uint64_t & x)
{
    const unsigned marker = *in++;

    if (TURBOPFOR_LIKELY(marker < VBYTE64_MARKER_2BYTE)) // marker < 0x98 (most common)
    {
        x = marker;
        return in;
    }

    if (TURBOPFOR_LIKELY(marker < VBYTE64_MARKER_3BYTE)) // marker < 0xD8
    {
        // 2-byte encoding: reconstruct value from marker + 1 data byte
        const unsigned data_byte = *in++;
        x = ((marker - VBYTE64_MARKER_2BYTE) << 8) + data_byte + VBYTE64_THRESHOLD_2BYTE;
        return in;
    }

    if (TURBOPFOR_LIKELY(marker < VBYTE64_MARKER_RAW)) // marker < 0xF8
    {
        // 3-byte encoding: marker + 2 data bytes (little-endian)
        const unsigned low16 = loadU16Fast(in);
        x = low16 + ((marker - VBYTE64_MARKER_3BYTE) << 16) + VBYTE64_THRESHOLD_3BYTE;
        return in + 2;
    }

    // Raw encoding: marker 0xF8..0xFD = 3..8 raw bytes
    const unsigned raw_bytes = (marker - VBYTE64_MARKER_RAW) + 3u;
    // Read up to 8 bytes and mask to the appropriate width
    // Note: reading 8 bytes is safe because TurboPFor always overallocates
    x = loadU64Fast(in) & ((raw_bytes >= 8u) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << (raw_bytes * 8u)) - 1ull));
    return in + raw_bytes;
}

/// Variable-byte encoding/decoding (64-bit)
unsigned char * vbEnc64(const uint64_t * in, unsigned n, unsigned char * out);
const unsigned char * vbDec64(const unsigned char * __restrict in, unsigned n, uint64_t * __restrict out);

/// P4 bit width selection (32-bit)
unsigned p4Bits32(const uint32_t * in, unsigned n, unsigned * out_exception_bits);

/// P4 bit width selection (64-bit)
unsigned p4Bits64(const uint64_t * in, unsigned n, unsigned * out_exception_bits);

/// Write P4 header (32-bit: max base bits = 32)
void writeHeader(unsigned char *& out, unsigned b, unsigned bx);

/// Write P4 header (64-bit: max base bits = 64, with 63->64 bit quirk)
void writeHeader64(unsigned char *& out, unsigned b, unsigned bx);

/// Scalar bit packing/unpacking for 64-bit values (horizontal format)
unsigned char * bitpack64Scalar(const uint64_t * in, unsigned n, unsigned char * out, unsigned b);
const unsigned char * bitunpack64Scalar(const unsigned char * in, unsigned n, uint64_t * out, unsigned b);

/// Fused unpack + delta1 decode for 64-bit values
const unsigned char * bitunpackd1_64Scalar(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b);

/// 128v64 bitpacking: 2-lane interleaved horizontal packing for 128 elements
/// 128 bits / 64 bits = 2 lanes, 64 groups x 2 lanes = 128 elements
unsigned char * bitpack128v64Scalar(const uint64_t * in, unsigned char * out, unsigned b);
const unsigned char * bitunpack128v64Scalar(const unsigned char * in, uint64_t * out, unsigned b);

/// 256v64 bitpacking: 4-lane interleaved horizontal packing for 256 elements
/// 256 bits / 64 bits = 4 lanes, 64 groups x 4 lanes = 256 elements
unsigned char * bitpack256v64Scalar(const uint64_t * in, unsigned char * out, unsigned b);
const unsigned char * bitunpack256v64Scalar(const unsigned char * in, uint64_t * out, unsigned b);

/// Apply delta1 decoding for 256-element 64-bit blocks
void applyDelta1_256_64(uint64_t * out, unsigned n, uint64_t start);

/// Delta-1 encode: out[i] = in[i] - prev - 1, where prev starts at `start`.
/// Equivalent to TurboPFor's bitdienc32(..., mindelta=1). Used as the
/// pre-processing step for p4D1Enc* (delta-1 encoded P4 compression).
template <typename T>
TURBOPFOR_ALWAYS_INLINE void deltaEnc1(const T * __restrict in, unsigned n, T * __restrict out, T start)
{
    for (unsigned i = 0; i < n; ++i)
    {
        out[i] = in[i] - start - 1;
        start = in[i];
    }
}

} // namespace turbopfor::scalar::detail
