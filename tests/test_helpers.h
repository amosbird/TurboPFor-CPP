#pragma once

#include "turbopfor.h"
#include "scalar/p4_scalar.h"
#include "simd/p4_simd.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

extern "C" unsigned char * p4enc32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4d1dec32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * p4enc128v32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4d1dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * p4enc256v32(uint32_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4d1dec256v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);
extern "C" unsigned char * bitpack32(unsigned * in, unsigned n, unsigned char * out, unsigned b);
extern "C" unsigned char * bitunpack32(const unsigned char * in, unsigned n, uint32_t * out, unsigned b);
extern "C" unsigned char * bitd1unpack32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);

// 64-bit C reference functions
extern "C" unsigned char * p4enc64(uint64_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4dec64(unsigned char * in, unsigned n, uint64_t * out);
extern "C" unsigned char * p4d1dec64(unsigned char * in, unsigned n, uint64_t * out, uint64_t start);
extern "C" unsigned char * p4enc128v64(uint64_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * p4dec128v64(unsigned char * in, unsigned n, uint64_t * out);
extern "C" unsigned char * bitpack64(uint64_t * in, unsigned n, unsigned char * out, unsigned b);
extern "C" unsigned char * bitunpack64(const unsigned char * in, unsigned n, uint64_t * out, unsigned b);
extern "C" unsigned char * vbenc64(uint64_t * in, unsigned n, unsigned char * out);
extern "C" unsigned char * vbdec64(unsigned char * in, unsigned n, uint64_t * out);

namespace turbopfor::scalar::detail
{
unsigned char * bitpack32Scalar(const uint32_t * in, unsigned n, unsigned char * out, unsigned b);
unsigned char * bitunpack32Scalar(unsigned char * in, unsigned n, uint32_t * out, unsigned b);
unsigned char * bitunpackd1_32Scalar(unsigned char * in, unsigned n, uint32_t * out, uint32_t start, unsigned b);
unsigned char * bitpack64Scalar(const uint64_t * in, unsigned n, unsigned char * out, unsigned b);
unsigned char * bitunpack64Scalar(unsigned char * in, unsigned n, uint64_t * out, unsigned b);
unsigned char * bitunpackd1_64Scalar(unsigned char * in, unsigned n, uint64_t * out, uint64_t start, unsigned b);
unsigned char * vbEnc64(const uint64_t * in, unsigned n, unsigned char * out);
unsigned char * vbDec64(unsigned char * __restrict in, unsigned n, uint64_t * __restrict out);
}

namespace
{

unsigned pad8(unsigned bits)
{
    return (bits + 7u) / 8u;
}

unsigned popcount64(uint64_t v)
{
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_popcountll(v));
#else
    unsigned c = 0;
    while (v)
    {
        v &= v - 1u;
        ++c;
    }
    return c;
#endif
}

void maskPaddingBits(unsigned char * buf, unsigned total_bits)
{
    unsigned rem = total_bits & 7u;
    if (rem == 0u || total_bits == 0u)
        return;

    unsigned bytes = pad8(total_bits);
    unsigned char mask = static_cast<unsigned char>((1u << rem) - 1u);
    buf[bytes - 1] &= mask;
}

void fillSequential(std::vector<uint32_t> & data, uint32_t base, uint32_t step)
{
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = base + static_cast<uint32_t>(i) * step;
}

void fillRandom(std::vector<uint32_t> & data, uint32_t max_val, std::mt19937 & rng)
{
    std::uniform_int_distribution<uint32_t> dist(0u, max_val);
    for (auto & v : data)
        v = dist(rng);
}

void fillConstant(std::vector<uint32_t> & data, uint32_t value)
{
    for (auto & v : data)
        v = value;
}

void fillWithExceptions(std::vector<uint32_t> & data, uint32_t base_max, uint32_t exc_value, unsigned exc_percent, std::mt19937 & rng)
{
    std::uniform_int_distribution<uint32_t> base_dist(0u, base_max);
    std::uniform_int_distribution<unsigned> exc_dist(0u, 99u);

    for (auto & v : data)
    {
        if (exc_dist(rng) < exc_percent)
            v = exc_value;
        else
            v = base_dist(rng);
    }
}

void fillSequential64(std::vector<uint64_t> & data, uint64_t base, uint64_t step)
{
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = base + static_cast<uint64_t>(i) * step;
}

void fillRandom64(std::vector<uint64_t> & data, uint64_t max_val, std::mt19937_64 & rng)
{
    std::uniform_int_distribution<uint64_t> dist(0ull, max_val);
    for (auto & v : data)
        v = dist(rng);
}

void fillConstant64(std::vector<uint64_t> & data, uint64_t value)
{
    for (auto & v : data)
        v = value;
}

void fillWithExceptions64(std::vector<uint64_t> & data, uint64_t base_max, uint64_t exc_value, unsigned exc_percent, std::mt19937_64 & rng)
{
    std::uniform_int_distribution<uint64_t> base_dist(0ull, base_max);
    std::uniform_int_distribution<unsigned> exc_dist(0u, 99u);
    std::mt19937 rng32(static_cast<uint32_t>(rng()));

    for (auto & v : data)
    {
        if (exc_dist(rng32) < exc_percent)
            v = exc_value;
        else
            v = base_dist(rng);
    }
}

void normalizeP4Enc32(unsigned char * buf, unsigned n)
{
    if (n == 0u)
        return;

    unsigned b = buf[0];

    if ((b & 0xC0u) == 0xC0u)
        return;

    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        unsigned offset = 1u;
        if (b & 0x80u)
        {
            bx = buf[1];
            offset = 2u;
        }
        b &= 0x3Fu;

        if (bx == 0u)
        {
            maskPaddingBits(buf + offset, n * b);
            return;
        }

        if (bx <= 32u)
        {
            unsigned bitmap_bytes = pad8(n);
            unsigned xn = 0u;
            for (unsigned i = 0; i < bitmap_bytes; i += 8)
            {
                uint64_t word = 0;
                unsigned chunk = (bitmap_bytes - i) < 8 ? (bitmap_bytes - i) : 8;
                std::memcpy(&word, buf + offset + i, chunk);
                xn += popcount64(word);
            }

            unsigned char * ex_pack = buf + offset + bitmap_bytes;
            maskPaddingBits(ex_pack, xn * bx);
            unsigned ex_bytes = pad8(xn * bx);
            unsigned char * base_pack = ex_pack + ex_bytes;
            maskPaddingBits(base_pack, n * b);
            return;
        }

        return;
    }

    unsigned bx = buf[1];
    unsigned offset = 2u;
    b &= 0x3Fu;
    (void)bx;
    maskPaddingBits(buf + offset, n * b);
}

void normalizeP4Enc64(unsigned char * buf, unsigned n)
{
    if (n == 0u)
        return;

    unsigned b = buf[0];

    // Constant block: 0xC0 flag
    if ((b & 0xC0u) == 0xC0u)
        return;

    // PFOR or bitpack-only: check vbyte flag (0x40)
    if ((b & 0x40u) == 0u)
    {
        unsigned bx = 0u;
        unsigned offset = 1u;
        if (b & 0x80u)
        {
            bx = buf[1];
            offset = 2u;
        }
        b &= 0x3Fu;

        // 63→64 quirk: TurboPFor never encodes b=63, always upgrades to 64
        if (b == 63u)
            b = 64u;

        if (bx == 0u)
        {
            // Bitpack-only: mask trailing bits in the packed data
            maskPaddingBits(buf + offset, static_cast<unsigned>(static_cast<uint64_t>(n) * b));
            return;
        }

        if (bx <= 64u)
        {
            unsigned bitmap_bytes = pad8(n);
            unsigned xn = 0u;
            for (unsigned i = 0; i < bitmap_bytes; i += 8)
            {
                uint64_t word = 0;
                unsigned chunk = (bitmap_bytes - i) < 8 ? (bitmap_bytes - i) : 8;
                std::memcpy(&word, buf + offset + i, chunk);
                xn += popcount64(word);
            }

            unsigned char * ex_pack = buf + offset + bitmap_bytes;
            maskPaddingBits(ex_pack, static_cast<unsigned>(static_cast<uint64_t>(xn) * bx));
            unsigned ex_bytes = pad8(static_cast<unsigned>(static_cast<uint64_t>(xn) * bx));

            // Base values packed after exceptions
            unsigned char * base_pack = ex_pack + ex_bytes;
            maskPaddingBits(base_pack, static_cast<unsigned>(static_cast<uint64_t>(n) * b));
            return;
        }

        return;
    }

    // Vbyte exceptions: flag 0x40
    unsigned bx = buf[1];
    unsigned offset = 2u;
    b &= 0x3Fu;
    if (b == 63u)
        b = 64u;
    (void)bx;
    maskPaddingBits(buf + offset, static_cast<unsigned>(static_cast<uint64_t>(n) * b));
}

} // namespace
