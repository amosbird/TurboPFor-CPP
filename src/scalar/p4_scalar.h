#pragma once

#include <cstdint>

namespace turbopfor::scalar
{

unsigned char * p4Enc32(uint32_t * in, unsigned n, unsigned char * out);
const unsigned char * p4Dec32(const unsigned char * in, unsigned n, uint32_t * out);
const unsigned char * p4D1Dec32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out);
const unsigned char * p4Dec128v32(const unsigned char * in, unsigned n, uint32_t * out);
const unsigned char * p4D1Dec128v32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out);
const unsigned char * p4Dec256v32(const unsigned char * in, unsigned n, uint32_t * out);
const unsigned char * p4D1Dec256v32(const unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

unsigned char * p4Enc64(uint64_t * in, unsigned n, unsigned char * out);
const unsigned char * p4D1Dec64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

unsigned char * p4Enc128v64(uint64_t * in, unsigned n, unsigned char * out);
const unsigned char * p4Dec128v64(const unsigned char * in, unsigned n, uint64_t * out);
const unsigned char * p4D1Dec128v64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

unsigned char * p4Enc256v64(uint64_t * in, unsigned n, unsigned char * out);
const unsigned char * p4Dec256v64(const unsigned char * in, unsigned n, uint64_t * out);
const unsigned char * p4D1Dec256v64(const unsigned char * in, unsigned n, uint64_t * out, uint64_t start);

} // namespace turbopfor::scalar
