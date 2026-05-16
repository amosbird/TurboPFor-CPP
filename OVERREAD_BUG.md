# TurboPFor-CPP: 64-bit Tail Block Overread Bug

## Problem

`bitunpack_b<B>` in `src/scalar/p4_scalar_bitunpack64_impl.h` (line 416-424) has a heap-buffer-overflow when decoding tail blocks (n < 32 elements) for bit-widths that use the interleaved path.

### Affected bit-widths

`uses_interleaved_path<B>()` returns true for B ∈ {33..39, 41..47, 49..55, 57..63} — i.e. B > 32 && B < 64, excluding byte-aligned widths 40, 48, 56.

### Root cause

```cpp
// line 416-424
if constexpr (uses_interleaved_path<B>())
{
    alignas(64) uint64_t tmp[32];
    unpack64_n_b<Delta1, B, 32>(in, tmp, start);  // reads ceil(32*B/8) bytes!
    std::memcpy(out, tmp, n * sizeof(uint64_t));
    return ret;
}
```

When n < 32, the actual encoded payload is `ceil(n * B / 8)` bytes, but `unpack64_n_b<B, 32>` decodes a full 32-element block, reading `ceil(32 * B / 8)` bytes from `in`.

### Overread calculation

- B=60, n=1: actual payload 8 bytes, reads 240 bytes → **232 bytes overread**
- B=60, n=3: actual payload 23 bytes, reads 240 bytes → **217 bytes overread**
- B=63, n=1: actual payload 8 bytes, reads 252 bytes → **244 bytes overread** (worst case)

### ASan report (from ClickHouse CI)

```
READ of size 8 at 0x7c7a12d3c54f thread T1747
  #0 turbopfor::scalar::detail::loadU64Fast() p4_scalar_internal.h:280
  #1 unpack_interleaved64<false, 60, 16, 16>  p4_scalar_bitunpack64_impl.h:147
  ...
  #7 p4D1Dec64PayloadExceptions             p4d1dec64.cpp:52
  #8 turbopfor::scalar::p4D1Dec64           p4d1dec64.cpp:134

0x7c7a12d3c54f is located 2 bytes after 205-byte region [0x7c7a12d3c480,0x7c7a12d3c54d)
```

## Scope analysis

### 32-bit scalar path — NOT affected

`src/scalar/p4_scalar_bitunpack_impl.h`: uses per-N template instantiation for tails (via `unpack32_tail` switch), and `load_partial` for non-8-byte-aligned last words. No overread.

### SIMD paths — NOT affected

`src/simd/`: SIMD paths process fixed-size blocks (128 or 256 elements) and delegate to scalar for variable-count tails. The scalar tail is the code being fixed here.

### `loadU64Fast` cross-word read in `unpack_interleaved64` — SEPARATE issue

In `TURBOPFOR_UNPACK64_ELEM` macro (line 119), when the last element spans two 64-bit words:
```cpp
w_cur = loadU64Fast(in + (wi_ + 1u) * 8u);
```
This can overread up to 7 bytes past the encoded payload for the **last** element. This affects both full blocks and tail blocks, but:
- For full blocks in a stream: safe in practice (next block's data follows)
- For the last full block when n is a multiple of 32: overreads up to 7 bytes
- For tail blocks: already protected by the tail fix

This is a separate, lower-priority issue. Callers should provide 7 bytes of padding after the encoded payload (ClickHouse does this with `DECODE_PADDING = 7`).

## Fix plan

### Option A: Pad input in tail path (minimal change)

Copy the tail payload into a zero-padded local buffer before decoding:

```cpp
if constexpr (uses_interleaved_path<B>())
{
    constexpr unsigned full_block_bytes = (32u * B + 7u) / 8u;
    alignas(64) unsigned char in_padded[full_block_bytes] = {};
    const unsigned actual_bytes = (n * B + 7u) / 8u;
    std::memcpy(in_padded, in, actual_bytes);

    alignas(64) uint64_t tmp[32];
    unpack64_n_b<Delta1, B, 32>(in_padded, tmp, start);
    std::memcpy(out, tmp, n * sizeof(uint64_t));
    return ret;
}
```

Stack cost: max 252 bytes (`full_block_bytes` for B=63). Acceptable.

### Option B: Generate per-N tail templates for 64-bit interleaved path

Like the 32-bit path does — instantiate `unpack64_n_b<B, N>` for N=1..31. This avoids the overread entirely and removes the extra memcpy. But it increases binary size significantly (31 templates × ~15 affected bit-widths = 465 instantiations).

This was explicitly avoided in the original code (comment: "This avoids instantiating 31 tail templates per bitwidth").

### Option C: Fix `unpack_interleaved64` to use `load_partial` for last word

Modify the `TURBOPFOR_UNPACK64_ELEM` macro to detect when the last element's `loadU64Fast` would overread and use `load_partial` instead. This fixes both the tail issue and the cross-word overread, but adds complexity and may hurt performance in the hot loop.

### Recommendation

**Option A** — minimal, correct, negligible performance impact. The tail path is cold (executed at most once per `bitunpack_b` call) and the memcpy is small.
