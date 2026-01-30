#pragma once

#include <cstdint>

namespace turbopfor
{

/// Encode n 32-bit integers using P4 (PFor) compression
unsigned char * p4Enc32(uint32_t * in, uint8_t n, unsigned char * out);

/// Decode n 32-bit integers using P4 (PFor) decompression with delta1
unsigned char * p4D1Dec32(unsigned char * in, uint8_t n, uint32_t * out, uint32_t start);

/// Encode n 32-bit integers using P4 with 128-element SIMD blocks
unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out);

/// Decode n 32-bit integers using P4 with 128-element SIMD blocks and delta1
unsigned char * p4D1Dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

/// Encode n 32-bit integers using P4 with 256-element SIMD blocks
unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out);

/// Decode n 32-bit integers using P4 with 256-element SIMD blocks and delta1
unsigned char * p4D1Dec256v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

}  // namespace turbopfor
