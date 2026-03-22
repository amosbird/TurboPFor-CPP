# Task 5: Encode — Fused IP32 Bitpack

**Depends on**: Task 1 (baseline data needed for comparison)
**Estimated complexity**: High
**Type**: Feature

## Objective

Replace the current two-pass encode (IP32 shuffle → temp buffer → bitpack128v32) with a fused single-pass encode that interleaves IP32 shuffle with bitpacking, eliminating the temp buffer and matching TurboPFor C's fused architecture. This should close the -8 to -11% encode gap for non-power-of-2 bit widths.

## Important Information

Before coding, Read FIRST -> Load `03-tasks-00-READBEFORE.md`

## Background

### Current two-pass architecture (our code)

In `bitpack128v64()` (`src/simd/bitpack128v64_simd.cpp`, line 37):
```cpp
// Pass 1: IP32 shuffle all 128 values into uint32_t tmp[128]
alignas(16) uint32_t tmp[128];
const __m128i * ip = reinterpret_cast<const __m128i *>(in);
for (unsigned i = 0; i < 128; i += 4)
{
    __m128i lo = _mm_loadu_si128(ip++);
    __m128i hi = _mm_loadu_si128(ip++);
    __m128i lo_shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 0, 3, 1));
    __m128i hi_shuf = _mm_shuffle_epi32(hi, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i result = _mm_or_si128(lo_shuf, hi_shuf);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(tmp + i), result);
}
// Pass 2: bitpack the temp buffer
bitpack128v32(tmp, out, b);
```

Then `bitpack128v32()` (`src/simd/bitpack128v32_simd.cpp`, line 313) dispatches:
- b=1,2,4,8,16: fully-unrolled specialized functions (fast)
- b=32: memcpy (fast)
- all other b: `bitpack128v32_general()` — runtime shifts with 4x unrolled loop

The two-pass architecture is why encode is -8 to -11% for non-power-of-2 b: the temp buffer causes extra L1 cache traffic (128 stores + 128 loads = 256 cache accesses for the 512 bytes), and the `bitpack128v32_general()` uses runtime variable shifts.

### TurboPFor C's fused architecture

TurboPFor C's `bitpack128v64` (in `bitpack.c`, line 417) works as follows:
1. Redefines `IP32(_ip_, i, iv)` as inline SIMD shuffle:
   ```c
   #define IP32(_ip_, i, iv) _mm_or_si128(
       _mm_shuffle_epi32(_mm_loadu_si128(_ip_++), _MM_SHUFFLE(2,0,3,1)),
       _mm_shuffle_epi32(_mm_loadu_si128(_ip_++), _MM_SHUFFLE(3,1,2,0)))
   ```
2. Then `#include "bitpack_.h"` generates `BITPACK128V32` — a switch(b) dispatching to 33 per-bitwidth macros. Each macro calls `IP32(ip, g, iv)` inline per group, which directly loads from the uint64_t input and shuffles, bypassing any temp buffer.
3. The per-bitwidth macros have **compile-time shift constants** — e.g., for b=3, groups shift by 0,3,6,9,12,...,29 (all immediate constants).

### Why TurboPFor C is faster for non-power-of-2 b

Two reasons:
1. **No temp buffer**: IP32 shuffle is inlined per group — zero extra cache traffic.
2. **Compile-time shifts**: Each `BITPACK128V32_N` macro has hardcoded `_mm_slli_epi32(iv, constant)`. Our `bitpack128v32_general()` uses runtime `_mm_slli_epi32(iv, shift)` where `shift` is a variable — this compiles to `psrld %xmm, %xmm` (variable shift from XMM register) which is slower than `psrld $imm, %xmm` (immediate shift).

For power-of-2 b (1,2,4,8,16,32), our specialized functions already have compile-time constants and simple patterns, so the gap is smaller (-1 to -3%).

## Files to Modify/Create

- `src/simd/bitpack128v64_simd.cpp` — Replace `bitpack128v64()` with fused version using `STO64_SWITCH` + template dispatch
- `src/simd/bitunpack_sse_templates.h` — (possibly) Add `PackStepSSE_IP32<B>` template for the fused pack inner loop, OR create a new header

## Detailed Steps

1. Update `PROGRESS.md` to mark this task as In Progress.

2. **Create a fused bitpack template** — a template function `bitpack128v64_fused<B>()` that combines IP32 shuffle with bitpacking in a single pass. The template makes B a compile-time constant, enabling immediate-constant shifts.

   The key insight: the same periodic-unroll math applies to encode. For bit width B, the shift-offset pattern repeats with period P = 32/gcd(B,32). However, for encode we can start simpler — a fully-unrolled approach per B, since the encode function is called less frequently than decode and code size pressure is less critical.

   **Approach A — Template with compile-time shift loop (simplest)**:

   ```cpp
   template <unsigned B>
   ALWAYS_INLINE void bitpack128v64_fused(const uint64_t * in, unsigned char * out)
   {
       const __m128i * ip = reinterpret_cast<const __m128i *>(in);
       __m128i * op = reinterpret_cast<__m128i *>(out);
       
       constexpr uint32_t maskVal = (B < 32) ? ((1u << B) - 1u) : 0xFFFFFFFFu;
       const __m128i mv = _mm_set1_epi32(static_cast<int>(maskVal));
       
       __m128i ov = _mm_setzero_si128();
       unsigned shift = 0;  // This is still runtime but changes predictably
       
       for (unsigned g = 0; g < 32; ++g)
       {
           // Inline IP32 shuffle: load 2 × __m128i, extract low 32 bits
           __m128i lo = _mm_loadu_si128(ip++);
           __m128i hi = _mm_loadu_si128(ip++);
           __m128i iv = _mm_or_si128(
               _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 0, 3, 1)),
               _mm_shuffle_epi32(hi, _MM_SHUFFLE(3, 1, 2, 0)));
           
           if constexpr (B < 32) iv = _mm_and_si128(iv, mv);
           
           if (shift == 0)
               ov = iv;
           else
               ov = _mm_or_si128(ov, _mm_slli_epi32(iv, static_cast<int>(shift)));
           
           shift += B;
           if (shift >= 32)
           {
               _mm_storeu_si128(op++, ov);
               shift -= 32;
               if (shift > 0)
                   ov = _mm_srli_epi32(iv, static_cast<int>(B - shift));
               else
                   ov = _mm_setzero_si128();
           }
       }
       
       if (shift > 0)
           _mm_storeu_si128(op++, ov);
   }
   ```

   **PROBLEM with Approach A**: The shifts (`_mm_slli_epi32(iv, shift)`, `_mm_srli_epi32(iv, B - shift)`) still use runtime `shift` variable. Even though B is compile-time, `shift` changes per group. The compiler MAY constant-fold if it unrolls the loop (since `shift = (g * B) % 32` is predictable), but this is not guaranteed.

   **Approach B — Periodic-unroll for encode (matches decode)**:

   Apply the same periodic-unroll technique. For bit width B with period P:
   - Unroll P groups with all shifts as compile-time constants
   - Loop over 32/P periods

   This requires creating `PackStepSSE_IP32<B, GroupInPeriod>` templates (similar to `UnpackStepSSE_STO64_D1<B, G>` for decode). Each step knows its compile-time offset and whether it spans a 32-bit boundary.

   ```cpp
   template <unsigned B, unsigned G>
   ALWAYS_INLINE void PackStepSSE_IP32(const __m128i *& ip, __m128i *& op,
                                        __m128i & ov, const __m128i & mv)
   {
       constexpr unsigned Offset = (G * B) % 32;
       constexpr unsigned End = Offset + B;
       constexpr bool Spans = (End > 32);
       
       // Inline IP32
       __m128i lo = _mm_loadu_si128(ip++);
       __m128i hi = _mm_loadu_si128(ip++);
       __m128i iv = _mm_or_si128(
           _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 0, 3, 1)),
           _mm_shuffle_epi32(hi, _MM_SHUFFLE(3, 1, 2, 0)));
       if constexpr (B < 32) iv = _mm_and_si128(iv, mv);
       
       if constexpr (Offset == 0)
           ov = iv;
       else
           ov = _mm_or_si128(ov, _mm_slli_epi32(iv, Offset));
       
       if constexpr (Spans || End == 32)
       {
           _mm_storeu_si128(op++, ov);
           if constexpr (Spans)
               ov = _mm_srli_epi32(iv, 32 - Offset);
           else
               ov = _mm_setzero_si128();
       }
   }
   ```

   Then a period body template calls `PackStepSSE_IP32<B, 0>`, `PackStepSSE_IP32<B, 1>`, ..., `PackStepSSE_IP32<B, P-1>` with compile-time constants.

   **Approach C — Just fuse IP32 into existing bitpack128v32_general (minimal change)**:

   Instead of creating new templates, modify `bitpack128v32_general()` to accept a function pointer or template parameter for loading input, so IP32 is inlined. However, this still has runtime shifts — it only eliminates the temp buffer, not the variable-shift problem.

   **Recommendation**: Start with **Approach A** (fused template with runtime shift loop). The IP32 fusion alone (eliminating 512 bytes of temp buffer traffic) may be sufficient to reach +1%. If not, upgrade to **Approach B** (periodic-unroll with compile-time shifts). Approach A is simpler to implement and verify for correctness.

3. **Add switch dispatch for fused encode** in `bitpack128v64()`:

   Replace the current two-pass code with:
   ```cpp
   if (b <= 32u)
   {
       unsigned char * pout = out + (V128_64_BLOCK_SIZE * b + 7u) / 8u;
       
       #define CALL_FUSED_PACK(B) bitpack128v64_fused<B>(in, out)
       STO64_SWITCH(b, CALL_FUSED_PACK);
       #undef CALL_FUSED_PACK
       
       return pout;
   }
   ```

   Note: the `STO64_SWITCH` macro is `#undef`'d at the end of the file currently. Either move the undef, redefine it, or use the same pattern inline.

   **IMPORTANT**: The `STO64_SWITCH` macro is currently defined in `bitpack128v64_simd.cpp` and `#undef`'d at line 239. The encode function `bitpack128v64()` is at line 37, BEFORE the decode functions. So the `STO64_SWITCH` is available. However, check that the switch is not `#undef`'d too early. You may need to reorganize the file slightly.

4. **Handle b=0 and b=32 edge cases**:
   - b=0: no output needed, return `out` immediately (already handled).
   - b=32: IP32 shuffle + direct memcpy of 128 × uint32_t = 512 bytes. Specialize `bitpack128v64_fused<32>` to just do IP32 + store (no masking, no shifting, no bit-spanning).

5. **Build and test correctness**:
   ```bash
   cmake --build build --target ab_test binary_compat_test vbyte64_test -j$(nproc)
   ./build/binary_compat_test && ./build/vbyte64_test
   ```
   **CRITICAL**: `binary_compat_test` verifies our encode produces bit-identical output to TurboPFor C. If it fails, the fused encode has a bug. Debug by comparing output byte-by-byte for a specific b value.

6. **Benchmark encode**:
   ```bash
   ./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 1-32
   ```
   The `--simd128v64` flag measures both encode and non-delta decode. Focus on the encode column.

7. **Evaluate results**:
   - If all b=1..32 encode shows ≥+1%: success, commit and move on.
   - If power-of-2 b (1,2,4,8,16,32) regressed: the fused approach may have introduced overhead for cases where the specialized functions were already optimal. Consider keeping the specialized paths for these b values and only using fused for other b.
   - If non-power-of-2 b improved but not to +1%: the runtime-shift issue is the bottleneck. Upgrade to Approach B (periodic-unroll with compile-time shifts).

8. **If Approach A is insufficient, implement Approach B** (periodic-unroll for encode):

   This mirrors the decode periodic-unroll architecture:
   - Create `PackPeriodStepSSE_IP32<B>` that unrolls one period of P groups
   - Create `bitpack128v64_fused_periodic<B>()` that loops over 32/P periods
   - Use a hybrid selector similar to decode: fully-unrolled for P ≤ 2 or P == 32, periodic for others

   **Code size is less of a concern for encode** than for decode, because encode is called once per block while decode is called in hot loops. So a fully-unrolled switch of 33 cases × ~1KB each ≈ 33KB is acceptable.

9. **Also check that b>32 encode (scalar) is still fast**:
   ```bash
   ./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 33-64
   ```
   The scalar encode path is unchanged, but verify no regressions from code layout changes.

10. Update `PROGRESS.md` with results.

11. Commit: `perf(simd128v64): fuse IP32 shuffle into bitpack for single-pass encode`

## Acceptance Criteria
- [ ] Encode shows ≥+1% for every b=1..32
- [ ] Encode for b=33..64 (scalar) shows no regression
- [ ] `binary_compat_test` passes (bit-identical output)
- [ ] `vbyte64_test` passes
- [ ] Results recorded in `PROGRESS.md`

## Testing
- **Correctness**: `./build/binary_compat_test && ./build/vbyte64_test` — **This is the most critical test for this task.** The fused encode must produce identical bytes.
- **Encode performance**: `./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 1-32`
- **Scalar regression check**: `./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw-range 33-64`
- **Quick single-bw check**: `./build/ab_test --simd128v64 --iters 300000 --runs 7 --bw 18` (b=18 was typically -9%)

## Notes
- The IP32 shuffle must use the EXACT same shuffle constants as the current code and TurboPFor C: `_MM_SHUFFLE(2,0,3,1)` for the first load, `_MM_SHUFFLE(3,1,2,0)` for the second. Getting these wrong produces valid-looking but incorrect packed output.
- The `STO64_SWITCH` macro is already defined in this file for decode. Reuse it for encode dispatch, but be careful about definition scope — the macro is `#undef`'d at line 239.
- For b=0, the existing `bitpack128v32` returns `out` without writing anything. The fused template for B=0 should do the same (no-op).
- The IP32 shuffle within the bitpack loop adds 5 instructions per group (2 loads, 2 shuffles, 1 OR) compared to a plain load from tmp buffer (1 load). But it eliminates 1 store to tmp + 1 load from tmp = 2 memory ops. Net: +3 ALU ops, -2 memory ops per group. Since L1 stores/loads cost ~4-5 cycles each, this should be a clear win.
- TurboPFor C's `BITPACK128V32_N` macros use `VI32` and `IP32` macros. For the plain bitpack128v64 case (no delta), `VI32` is a no-op and `IP32` does the shuffle+load. The `IPPE`/`OPPE` macros at the end of each case are also no-ops for the non-delta case.
