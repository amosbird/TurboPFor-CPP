// Regression test for B=0 + Patching path over-advancing the input pointer
// in the SIMD bitunpack templates (AVX2 256v32 and SSE 128v32).
//
// Bug: the B=0 branch in bitunpack_{avx2,sse}_entry used to fall through to the
// generic UnpackStep::run for the Patching case. That generic step starts by
// loading a 256/128-bit word from `ip` and advancing it, but for B=0 there
// is no packed input at all — so the input pointer would advance by 32 (AVX2)
// or 16 (SSE) bytes beyond the encoded data. Any subsequent decode call would
// then misinterpret the stream.
//
// This test triggers the (b=0, bx>0) PFOR case by crafting input that
// `p4Bits32` picks as best-encoded with zero base bits plus a small bx.
// It checks two things:
//   1. The decoder returns a pointer equal to the encoder's end pointer.
//   2. Decoded output matches input.
//
// All variants exercised: p4Dec32, p4D1Dec32, p4Dec128v32, p4D1Dec128v32,
// p4Dec256v32, p4D1Dec256v32. The scalar variants serve as a sanity baseline.

#include "test_helpers.h"

#include <cstdio>
#include <cstdint>
#include <vector>

namespace
{

// Build a pattern that should make p4Enc32 pick b=0 + bx>0:
// mostly zeros with a sprinkling of small nonzero values.
// (b=0 packs no bits at all; the few nonzeros are stored as exceptions.)
void fillSparsePatchPattern(std::vector<uint32_t> & data, uint32_t exc_value, unsigned exc_stride)
{
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = ((i % exc_stride) == 0u) ? exc_value : 0u;
}

// Build a delta1 pattern that yields (b=0, bx>0) after delta1 pre-encoding:
// strictly consecutive (so deltas = 1, delta1 value = 0) with rare bumps.
void fillSparsePatchPatternD1(std::vector<uint32_t> & data, uint32_t start, uint32_t bump, unsigned bump_stride)
{
    uint32_t v = start;
    for (size_t i = 0; i < data.size(); ++i)
    {
        v += 1u;
        if ((i % bump_stride) == 0u && i != 0u)
            v += bump; // produces delta1 value = bump for this index
        data[i] = v;
    }
}

// Returns true if the header at buf[0] encodes a PFOR block with b == 0 and bx > 0.
bool headerSaysB0Patching(const unsigned char * buf)
{
    unsigned hdr = buf[0];
    if ((hdr & 0xC0u) == 0xC0u)
        return false; // constant block
    if ((hdr & 0x40u) != 0u)
        return false; // vbyte format
    if ((hdr & 0x80u) == 0u)
        return false; // no bx byte
    unsigned b = hdr & 0x7Fu;
    unsigned bx = buf[1];
    return b == 0u && bx > 0u;
}

template <typename EncFn, typename DecFn>
bool roundtripCheck(const char * name, const std::vector<uint32_t> & input, EncFn enc, DecFn dec, bool require_b0 = true)
{
    const unsigned n = static_cast<unsigned>(input.size());
    std::vector<uint32_t> in_copy = input;
    std::vector<unsigned char> buf(n * 5u + 256u, 0u);
    std::vector<uint32_t> out(n + 8u, 0xDEADBEEFu);

    unsigned char * enc_end = enc(in_copy.data(), n, buf.data());
    size_t enc_bytes = static_cast<size_t>(enc_end - buf.data());

    if (require_b0 && !headerSaysB0Patching(buf.data()))
    {
        std::printf("  [%s] header=0x%02x not (b=0,bx>0); pattern needs adjustment\n", name, buf[0]);
        return false;
    }

    const unsigned char * dec_end = dec(buf.data(), n, out.data());
    size_t dec_bytes = static_cast<size_t>(dec_end - buf.data());

    if (dec_bytes != enc_bytes)
    {
        std::printf("  [%s] FAIL: enc_bytes=%zu dec_bytes=%zu (over-advance by %zd)\n",
                    name, enc_bytes, dec_bytes, static_cast<ssize_t>(dec_bytes) - static_cast<ssize_t>(enc_bytes));
        return false;
    }

    for (unsigned i = 0; i < n; ++i)
    {
        if (out[i] != input[i])
        {
            std::printf("  [%s] FAIL: out[%u]=%u expected %u\n", name, i, out[i], input[i]);
            return false;
        }
    }

    std::printf("  [%s] OK (n=%u, bytes=%zu)\n", name, n, enc_bytes);
    return true;
}

bool runOne(unsigned n)
{
    using namespace turbopfor;

    bool all_ok = true;

    // Non-delta pattern: mostly zero with rare nonzero ⇒ b=0, bx>0.
    std::vector<uint32_t> data(n);
    // Use value=255 (8 bits) at sparse positions so encoder must choose b=0+bx=8
    // (bitmap exceptions are cheaper than packing 256 8-bit values directly when
    // only a few positions are non-zero).
    fillSparsePatchPattern(data, /*exc_value=*/255u, /*exc_stride=*/16u);

    all_ok &= roundtripCheck("p4Dec32",
                             data,
                             [](uint32_t * in, unsigned cnt, unsigned char * out) { return p4Enc32(in, cnt, out); },
                             [](const unsigned char * in, unsigned cnt, uint32_t * out) { return p4Dec32(in, cnt, out); });

    if (n == 128u)
    {
        all_ok &= roundtripCheck("p4Dec128v32",
                                 data,
                                 [](uint32_t * in, unsigned cnt, unsigned char * out) { return p4Enc128v32(in, cnt, out); },
                                 [](const unsigned char * in, unsigned cnt, uint32_t * out) { return p4Dec128v32(in, cnt, out); });
    }

    if (n == 256u)
    {
        all_ok &= roundtripCheck("p4Dec256v32",
                                 data,
                                 [](uint32_t * in, unsigned cnt, unsigned char * out) { return p4Enc256v32(in, cnt, out); },
                                 [](const unsigned char * in, unsigned cnt, uint32_t * out) { return p4Dec256v32(in, cnt, out); });
    }

    // Delta1 pattern: encoder computes tmp[i] = data[i] - data[i-1] - 1 (delta1),
    // with tmp[0] = data[0] - start - 1. To make tmp values mostly zero with sparse
    // 8-bit exceptions (so encoder picks b=0+bx=8 with bitmap exceptions, header 0x80),
    // use a small bump_stride so the count of exceptions is high enough that the
    // bitmap cost (32 bytes for 256 elements) is amortized vs the vbyte alternative.
    const uint32_t start = 0u;
    std::vector<uint32_t> d1data(n);
    fillSparsePatchPatternD1(d1data, /*start=*/start, /*bump=*/255u, /*bump_stride=*/8u);

    all_ok &= roundtripCheck("p4D1Dec32",
                             d1data,
                             [start](uint32_t * in, unsigned cnt, unsigned char * out) { return p4D1Enc32(in, cnt, out, start); },
                             [start](const unsigned char * in, unsigned cnt, uint32_t * out) { return p4D1Dec32(in, cnt, out, start); });

    if (n == 128u)
    {
        all_ok &= roundtripCheck("p4D1Dec128v32",
                                 d1data,
                                 [start](uint32_t * in, unsigned cnt, unsigned char * out) { return p4D1Enc128v32(in, cnt, out, start); },
                                 [start](const unsigned char * in, unsigned cnt, uint32_t * out) { return p4D1Dec128v32(in, cnt, out, start); });
    }

    if (n == 256u)
    {
        all_ok &= roundtripCheck("p4D1Dec256v32",
                                 d1data,
                                 [start](uint32_t * in, unsigned cnt, unsigned char * out) { return p4D1Enc256v32(in, cnt, out, start); },
                                 [start](const unsigned char * in, unsigned cnt, uint32_t * out) { return p4D1Dec256v32(in, cnt, out, start); });
    }

    return all_ok;
}

} // namespace

unsigned runOveradvanceTest()
{
    std::printf("=== Over-advance (B=0+Patching) regression test ===\n");

    unsigned failed = 0;

    std::printf("-- n=128 --\n");
    if (!runOne(128u))
        ++failed;

    std::printf("-- n=256 --\n");
    if (!runOne(256u))
        ++failed;

    // Scalar paths only (smaller sizes).
    std::printf("-- n=64 --\n");
    if (!runOne(64u))
        ++failed;

    return failed;
}
