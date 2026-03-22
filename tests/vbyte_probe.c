// Empirical test: call vbenc32/vbdec32 with arrays large enough to avoid OVERFLOWE
// Also probe the per-value encoding by encoding many copies of the same value
#include <stdint.h>
#include <stdio.h>
#include <string.h>

unsigned char * vbenc32(uint32_t * in, unsigned n, unsigned char * out);
unsigned char * vbdec32(unsigned char * in, unsigned n, uint32_t * out);
unsigned char * vbenc64(uint64_t * in, unsigned n, unsigned char * out);
unsigned char * vbdec64(unsigned char * in, unsigned n, uint64_t * out);

static void dump_bytes(const char * label, const unsigned char * buf, int len)
{
    printf("  %s (%d bytes): ", label, len);
    for (int i = 0; i < len && i < 32; i++)
        printf("%02X ", buf[i]);
    if (len > 32)
        printf("...");
    printf("\n");
}

// Encode N copies of the same 32-bit value and see per-value byte encoding
static void probe_vb32(uint32_t val, int n)
{
    uint32_t in[256];
    unsigned char buf[2048];
    uint32_t decoded[256];

    for (int i = 0; i < n; i++)
        in[i] = val;
    memset(buf, 0xAA, sizeof(buf));

    unsigned char * end = vbenc32(in, n, buf);
    int total = (int)(end - buf);

    if (buf[0] == 0xFF)
    {
        printf("  val=%u (0x%08X): UNCOMPRESSED (%d bytes for n=%d)\n", val, val, total, n);
    }
    else
    {
        int per_value = total / n; // approximate
        printf("  val=%u (0x%08X): compressed %d bytes for n=%d (~%d per value)\n", val, val, total, n, per_value);
        dump_bytes("first bytes", buf, total < 32 ? total : 32);

        // Decode and verify
        memset(decoded, 0, sizeof(decoded));
        unsigned char * dend = vbdec32(buf, n, decoded);
        int ok = 1;
        for (int i = 0; i < n; i++)
            if (decoded[i] != val)
            {
                ok = 0;
                break;
            }
        printf("  decode %s, consumed %d bytes\n", ok ? "OK" : "FAIL", (int)(dend - buf));
    }
}

// Encode N copies of the same 64-bit value
static void probe_vb64(uint64_t val, int n)
{
    uint64_t in[256];
    unsigned char buf[4096];
    uint64_t decoded[256];

    for (int i = 0; i < n; i++)
        in[i] = val;
    memset(buf, 0xAA, sizeof(buf));

    unsigned char * end = vbenc64(in, n, buf);
    int total = (int)(end - buf);

    if (buf[0] == 0xFF)
    {
        printf("  val=%llu (0x%016llX): UNCOMPRESSED (%d bytes for n=%d)\n", (unsigned long long)val, (unsigned long long)val, total, n);
    }
    else
    {
        int per_value = total / n;
        printf(
            "  val=%llu (0x%016llX): compressed %d bytes for n=%d (~%d per value)\n",
            (unsigned long long)val,
            (unsigned long long)val,
            total,
            n,
            per_value);
        dump_bytes("first bytes", buf, total < 32 ? total : 32);

        memset(decoded, 0, sizeof(decoded));
        unsigned char * dend = vbdec64(buf, n, decoded);
        int ok = 1;
        for (int i = 0; i < n; i++)
            if (decoded[i] != val)
            {
                ok = 0;
                break;
            }
        printf("  decode %s, consumed %d bytes\n", ok ? "OK" : "FAIL", (int)(dend - buf));
    }
}

int main()
{
    int n = 128; // enough to avoid OVERFLOWE for small values

    printf("=== 32-bit vbyte, n=%d ===\n\n", n);

    printf("--- 1-byte values ---\n");
    probe_vb32(0, n);
    probe_vb32(1, n);
    probe_vb32(127, n);
    probe_vb32(155, n);
    probe_vb32(156, n);
    probe_vb32(157, n);
    probe_vb32(158, n);
    probe_vb32(159, n);

    printf("\n--- 2-byte values ---\n");
    probe_vb32(200, n);
    probe_vb32(1000, n);
    probe_vb32(16539, n);
    probe_vb32(16540, n);
    probe_vb32(16541, n);
    probe_vb32(16542, n);
    probe_vb32(16543, n);

    printf("\n--- 3-byte values ---\n");
    probe_vb32(20000, n);
    probe_vb32(100000, n);
    probe_vb32(2113691, n);
    probe_vb32(2113692, n);
    probe_vb32(2113693, n);
    probe_vb32(2113694, n);
    probe_vb32(2113695, n);

    printf("\n--- 4+ byte values ---\n");
    probe_vb32(3000000, n);
    probe_vb32(16777215, n);
    probe_vb32(16777216, n);
    probe_vb32(0xFFFFFFFF, n);

    printf("\n\n=== 64-bit vbyte, n=%d ===\n\n", n);

    printf("--- Small values ---\n");
    probe_vb64(0, n);
    probe_vb64(1, n);
    probe_vb64(127, n);
    probe_vb64(153, n);
    probe_vb64(154, n);
    probe_vb64(155, n);
    probe_vb64(158, n);

    printf("\n--- 2-byte boundary ---\n");
    probe_vb64(16537, n);
    probe_vb64(16538, n);
    probe_vb64(16539, n);

    printf("\n--- 3-byte boundary ---\n");
    probe_vb64(2113689, n);
    probe_vb64(2113690, n);
    probe_vb64(2113691, n);

    printf("\n--- 64-bit large values ---\n");
    probe_vb64(0xFFFFFFFFULL, n);
    probe_vb64(0x100000000ULL, n);
    probe_vb64(0xFFFFFFFFFFULL, n);
    probe_vb64(0xFFFFFFFFFFFFULL, n);
    probe_vb64(0xFFFFFFFFFFFFFFULL, n);
    probe_vb64(0xFFFFFFFFFFFFFFFFULL, n);

    printf("\nDone.\n");
    return 0;
}
