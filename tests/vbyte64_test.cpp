// Quick validation: C++ vbEnc64/vbDec64 vs TurboPFor C vbenc64/vbdec64
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// TurboPFor C reference
extern "C" {
unsigned char * vbenc64(uint64_t * in, unsigned n, unsigned char * out);
unsigned char * vbdec64(unsigned char * in, unsigned n, uint64_t * out);
}

// Our C++ implementation
namespace turbopfor::scalar::detail
{
unsigned char * vbEnc64(const uint64_t * in, unsigned n, unsigned char * out);
unsigned char * vbDec64(unsigned char * in, unsigned n, uint64_t * out);
}

static int test_vb64_compat(const uint64_t * values, unsigned n, const char * label)
{
    unsigned char c_buf[4096], cpp_buf[4096];
    uint64_t c_dec[256], cpp_dec[256];

    // Encode with both
    uint64_t * mut_values = const_cast<uint64_t *>(values); // C API not const-correct
    unsigned char * c_end = vbenc64(mut_values, n, c_buf);
    unsigned char * cpp_end = turbopfor::scalar::detail::vbEnc64(values, n, cpp_buf);

    int c_len = (int)(c_end - c_buf);
    int cpp_len = (int)(cpp_end - cpp_buf);

    // Compare encoded output
    if (c_len != cpp_len || memcmp(c_buf, cpp_buf, c_len) != 0)
    {
        printf("FAIL [%s] n=%u: C encoded %d bytes, C++ encoded %d bytes\n", label, n, c_len, cpp_len);
        printf("  C:   ");
        for (int i = 0; i < c_len && i < 32; i++)
            printf("%02X ", c_buf[i]);
        printf("\n");
        printf("  C++: ");
        for (int i = 0; i < cpp_len && i < 32; i++)
            printf("%02X ", cpp_buf[i]);
        printf("\n");
        return 1;
    }

    // Decode C-encoded data with C++ decoder
    memset(cpp_dec, 0, sizeof(cpp_dec));
    unsigned char * dec_end = turbopfor::scalar::detail::vbDec64(c_buf, n, cpp_dec);
    int dec_len = (int)(dec_end - c_buf);

    if (dec_len != c_len)
    {
        printf("FAIL [%s] n=%u: C++ decoder consumed %d bytes, expected %d\n", label, n, dec_len, c_len);
        return 1;
    }

    for (unsigned i = 0; i < n; i++)
    {
        if (cpp_dec[i] != values[i])
        {
            printf(
                "FAIL [%s] n=%u: value[%u] = %llu, decoded = %llu\n",
                label,
                n,
                i,
                (unsigned long long)values[i],
                (unsigned long long)cpp_dec[i]);
            return 1;
        }
    }

    // Decode C++-encoded data with C decoder
    memset(c_dec, 0, sizeof(c_dec));
    unsigned char * c_dec_end = vbdec64(cpp_buf, n, c_dec);
    int c_dec_len = (int)(c_dec_end - cpp_buf);

    if (c_dec_len != cpp_len)
    {
        printf("FAIL [%s] n=%u: C decoder consumed %d bytes from C++ encoded, expected %d\n", label, n, c_dec_len, cpp_len);
        return 1;
    }

    for (unsigned i = 0; i < n; i++)
    {
        if (c_dec[i] != values[i])
        {
            printf(
                "FAIL [%s] n=%u: C-decoded value[%u] = %llu, expected %llu\n",
                label,
                n,
                i,
                (unsigned long long)c_dec[i],
                (unsigned long long)values[i]);
            return 1;
        }
    }

    return 0;
}

int main()
{
    int failures = 0;

    // Test 1: Uniform arrays of each size class
    printf("=== vbyte64 binary compatibility ===\n\n");

    uint64_t arr[128];

    // 1-byte values
    for (int i = 0; i < 128; i++)
        arr[i] = i;
    failures += test_vb64_compat(arr, 128, "1-byte sequential");

    for (int i = 0; i < 128; i++)
        arr[i] = 151;
    failures += test_vb64_compat(arr, 128, "1-byte max (151)");

    // 2-byte values
    for (int i = 0; i < 128; i++)
        arr[i] = 152 + i;
    failures += test_vb64_compat(arr, 128, "2-byte sequential");

    for (int i = 0; i < 128; i++)
        arr[i] = 16535;
    failures += test_vb64_compat(arr, 128, "2-byte max (16535)");

    // 3-byte values
    for (int i = 0; i < 128; i++)
        arr[i] = 16536 + i;
    failures += test_vb64_compat(arr, 128, "3-byte sequential");

    for (int i = 0; i < 128; i++)
        arr[i] = 2113687;
    failures += test_vb64_compat(arr, 128, "3-byte max (2113687)");

    // Raw 3 bytes
    for (int i = 0; i < 128; i++)
        arr[i] = 2113688 + i;
    failures += test_vb64_compat(arr, 128, "raw-3 sequential");

    // Raw 4 bytes
    for (int i = 0; i < 128; i++)
        arr[i] = 0xFFFFFF + i;
    failures += test_vb64_compat(arr, 128, "raw-4 sequential");

    // Raw 5 bytes
    for (int i = 0; i < 128; i++)
        arr[i] = 0xFFFFFFFFULL + i;
    failures += test_vb64_compat(arr, 128, "raw-5 sequential");

    // Raw 6 bytes
    for (int i = 0; i < 128; i++)
        arr[i] = 0xFFFFFFFFFFULL + i;
    failures += test_vb64_compat(arr, 128, "raw-6 sequential");

    // Raw 7 bytes (will hit OVERFLOWE uncompressed)
    for (int i = 0; i < 128; i++)
        arr[i] = 0xFFFFFFFFFFFFULL + i;
    failures += test_vb64_compat(arr, 128, "raw-7 sequential");

    // Large values (will definitely hit OVERFLOWE)
    for (int i = 0; i < 128; i++)
        arr[i] = 0xFFFFFFFFFFFFFFFFULL - i;
    failures += test_vb64_compat(arr, 128, "raw-8 large");

    // Mixed values
    for (int i = 0; i < 128; i++)
    {
        switch (i % 8)
        {
            case 0:
                arr[i] = 0;
                break;
            case 1:
                arr[i] = 100;
                break;
            case 2:
                arr[i] = 500;
                break;
            case 3:
                arr[i] = 50000;
                break;
            case 4:
                arr[i] = 3000000;
                break;
            case 5:
                arr[i] = 0x12345678ULL;
                break;
            case 6:
                arr[i] = 0x123456789AULL;
                break;
            case 7:
                arr[i] = 0x123456789ABCDEFULL;
                break;
        }
    }
    failures += test_vb64_compat(arr, 128, "mixed sizes");

    // Small n
    arr[0] = 42;
    failures += test_vb64_compat(arr, 1, "n=1 small");

    arr[0] = 0xDEADBEEFCAFEULL;
    failures += test_vb64_compat(arr, 1, "n=1 large");

    printf("\n%s: %d failures\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
