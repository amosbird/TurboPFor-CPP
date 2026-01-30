#pragma once

#include <cstdint>

namespace turbopfor::scalar
{

unsigned char * p4Enc32(uint32_t * in, unsigned n, unsigned char * out);
unsigned char * p4D1Dec32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

unsigned char * p4Enc128v32(uint32_t * in, unsigned n, unsigned char * out);
unsigned char * p4D1Dec128v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

unsigned char * p4Enc256v32(uint32_t * in, unsigned n, unsigned char * out);
unsigned char * p4D1Dec256v32(unsigned char * in, unsigned n, uint32_t * out, uint32_t start);

} // namespace turbopfor::scalar
