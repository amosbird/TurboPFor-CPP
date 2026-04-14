#pragma once

#include <cstdint>

namespace turbopfor
{

/// Encode n 32-bit integers using P4 (PFor) compression
unsigned char * p4Enc32(uint32_t * in, unsigned n, unsigned char * out);

/// Decode n 32-bit integers using P4 (PFor) decompression (no delta)
const unsigned char * p4Dec32(const unsigned char * in, unsigned n, uint32_t * out);

/// Decode n 32-bit integers using P4 (PFor) decompression with delta1
const unsigned char * p4D1Dec32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

/// Encode n 32-bit integers using P4 with 128-element SIMD blocks
unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out);

/// Decode n 32-bit integers using P4 with 128-element SIMD blocks (no delta)
const unsigned char * p4Dec128v32(const unsigned char * in, unsigned n, uint32_t * out);

/// Decode n 32-bit integers using P4 with 128-element SIMD blocks and delta1
const unsigned char * p4D1Dec128v32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

/// Encode n 32-bit integers using P4 with 256-element SIMD blocks
unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out);

/// Decode n 32-bit integers using P4 with 256-element SIMD blocks (no delta)
const unsigned char * p4Dec256v32(const unsigned char * in, unsigned n, uint32_t * out);

/// Decode n 32-bit integers using P4 with 256-element SIMD blocks and delta1
const unsigned char * p4D1Dec256v32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

/// Encode n 64-bit integers using P4 (PFor) compression
unsigned char * p4Enc64(uint64_t * in, unsigned n, unsigned char * out);

/// Decode n 64-bit integers using P4 (PFor) decompression with delta1
const unsigned char * p4D1Dec64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

/// Encode n 64-bit integers using P4 with 128-element hybrid blocks
/// Uses 128v32 SIMD format when b<=32, scalar 64-bit when b>32
unsigned char * p4Enc128v64(uint64_t * in, unsigned n, unsigned char * out);

/// Decode n 64-bit integers using P4 with 128-element hybrid blocks (non-delta)
const unsigned char * p4Dec128v64(const unsigned char * in, unsigned n, uint64_t * out);

/// Decode n 64-bit integers using P4 with 128-element hybrid blocks and delta1
const unsigned char * p4D1Dec128v64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

/// Encode n 64-bit integers using P4 with 256-element hybrid blocks
/// Uses 256v32 format when b<=32, scalar 64-bit when b>32
unsigned char * p4Enc256v64(uint64_t * in, unsigned n, unsigned char * out);

/// Decode n 64-bit integers using P4 with 256-element hybrid blocks (non-delta)
const unsigned char * p4Dec256v64(const unsigned char * in, unsigned n, uint64_t * out);

/// Decode n 64-bit integers using P4 with 256-element hybrid blocks and delta1
const unsigned char * p4D1Dec256v64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

} // namespace turbopfor
