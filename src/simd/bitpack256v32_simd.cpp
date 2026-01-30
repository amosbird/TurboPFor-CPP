#include "p4_simd.h"
#include "p4_simd_internal_256v.h"
#include <immintrin.h>

namespace turbopfor::simd::detail {
namespace {

using namespace turbopfor::simd::detail;

// --- Template Implementation ---

template <unsigned B>
struct MaskGen {
    static constexpr uint32_t value = (1u << B) - 1u;
};

template <>
struct MaskGen<32> {
    static constexpr uint32_t value = 0xFFFFFFFFu;
};

// Pack 8x32 bits horizontally
template <unsigned B, unsigned G, int CurrentStoredIdx>
struct PackStep {
    static ALWAYS_INLINE void run(const __m256i*& ip, __m256i& ov, unsigned char*& op, const __m256i& mask) {
        constexpr int TargetIdx = (G * B) / 32;      // Index of 32-bit word we are writing to (0..B-1)
        constexpr int Offset = (G * B) % 32;         // Bit offset within that word
        constexpr bool Spans = (Offset + B > 32);    // Does this value span across two words?
        
        // Load input (8 x 32-bit integers from input group G)
        // Group G contains: in[G*8], in[G*8+1], ... in[G*8+7]
        // Which corresponds to one value per lane.
        __m256i iv = _mm256_loadu_si256(ip++);
        
        // Mask the input values (only B bits)
        if (B != 32) {
            iv = _mm256_and_si256(iv, mask);
        }

        // Pack bits into accumulator ov
        if (Offset == 0) {
            // New word starts exactly at bit 0
            if (CurrentStoredIdx != TargetIdx) {
               // We moved to a new word, but previous one was fully written? 
               // Wait, logic:
               // ov holds the CURRENT 32-bit word being built.
               // If Offset == 0, it means we just finished previous word (or this is the first).
               // So ov should be reset to iv.
               ov = iv;
            } else {
               // This should not happen if logic is correct, Offset 0 usually implies new word start
               ov = iv;
            }
        } else {
            // Add to existing accumulator
            ov = _mm256_or_si256(ov, _mm256_slli_epi32(iv, Offset));
        }

        if (Spans) {
            // The current value spills over to the next 32-bit word.
            // 1. Store the current full word
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(op), ov);
            op += 32; // Advance 32 bytes (8 lanes x 4 bytes)

            // 2. Start next word with remaining bits
            constexpr int BitsInFirst = 32 - Offset;
            ov = _mm256_srli_epi32(iv, BitsInFirst);
        } else if (Offset + B == 32) {
             // Exact fit at end of word
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(op), ov);
            op += 32;
            // Next iteration will start new word (Offset=0)
        }

        constexpr int NextStoredIdx = Spans ? TargetIdx + 1 : TargetIdx;
        PackStep<B, G + 1, NextStoredIdx>::run(ip, ov, op, mask);
    }
};

// Base case
template <unsigned B, int CurrentStoredIdx>
struct PackStep<B, 32, CurrentStoredIdx> {
    static ALWAYS_INLINE void run(const __m256i*&, __m256i&, unsigned char*&, const __m256i&) {}
};

// Entry point wrappers
template <unsigned B>
ALWAYS_INLINE unsigned char* bitpack_entry(const uint32_t* in, unsigned char* out) {
    if (B == 0) return out;
    if (B == 32) {
        // Just copy
        std::memcpy(out, in, 256 * 4);
        return out + 256 * 4;
    }

    const __m256i* ip = reinterpret_cast<const __m256i*>(in);
    unsigned char* op = out;
    
    const uint32_t mask_val = MaskGen<B>::value;
    const __m256i mask = _mm256_set1_epi32(static_cast<int>(mask_val));
    
    __m256i ov = _mm256_setzero_si256(); // Accumulator

    PackStep<B, 0, -1>::run(ip, ov, op, mask);
    
    return out + (256 * B + 7) / 8;
}

} // namespace

// Dispatch Table
typedef unsigned char * (*bitpack256v32_func)(const uint32_t *, unsigned char *);
static const bitpack256v32_func bitpack_table_256v[33] = {
    bitpack_entry<0>, bitpack_entry<1>, bitpack_entry<2>, bitpack_entry<3>,
    bitpack_entry<4>, bitpack_entry<5>, bitpack_entry<6>, bitpack_entry<7>,
    bitpack_entry<8>, bitpack_entry<9>, bitpack_entry<10>, bitpack_entry<11>,
    bitpack_entry<12>, bitpack_entry<13>, bitpack_entry<14>, bitpack_entry<15>,
    bitpack_entry<16>, bitpack_entry<17>, bitpack_entry<18>, bitpack_entry<19>,
    bitpack_entry<20>, bitpack_entry<21>, bitpack_entry<22>, bitpack_entry<23>,
    bitpack_entry<24>, bitpack_entry<25>, bitpack_entry<26>, bitpack_entry<27>,
    bitpack_entry<28>, bitpack_entry<29>, bitpack_entry<30>, bitpack_entry<31>,
    bitpack_entry<32>
};

unsigned char * bitpack256v32(const uint32_t * in, unsigned char * out, unsigned b)
{
    return bitpack_table_256v[b](in, out);
}

} // namespace turbopfor::simd::detail
