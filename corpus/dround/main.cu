// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(cmd)                                                         \
    do {                                                                   \
        cudaError_t err_ = (cmd);                                          \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s\n", cudaGetErrorString(err_));        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// Double to float keeps a halfway tie on the even float and rounds a
// value past the finite maximum down to that maximum. A double NaN
// becomes a quiet float NaN with the same sign. Double to int saturates,
// and a NaN is the high bit in every rounding mode.
__global__ void convert(unsigned* u, long long* s) {
    union {
        float as_float;
        unsigned as_bits;
    } bits;
    union {
        double as_double;
        unsigned long long as_bits;
    } nan;
    double tie = 1.0 + 0x1p-24;
    double halfsub = 0x1p-150;
    double limit = 3.40282346638528859811704183484516925e+38;
    nan.as_bits = 0xfff8000000000001ull;
    bits.as_float = __double2float_rn(tie);
    u[0] = bits.as_bits;
    bits.as_float = __double2float_ru(tie);
    u[1] = bits.as_bits;
    bits.as_float = __double2float_rd(-tie);
    u[2] = bits.as_bits;
    bits.as_float = __double2float_rn(halfsub);
    u[3] = bits.as_bits;
    bits.as_float = __double2float_ru(halfsub);
    u[4] = bits.as_bits;
    bits.as_float = __double2float_rz(nextafter(limit, 1.0e308));
    u[5] = bits.as_bits;
    bits.as_float = __double2float_ru(nextafter(limit, 1.0e308));
    u[6] = bits.as_bits;
    bits.as_float = __double2float_rn(nan.as_double);
    u[7] = bits.as_bits;
    bits.as_float = __double2float_rd(-0.0);
    u[8] = bits.as_bits;
    bits.as_float = __double2float_ru(16777217.0);
    u[9] = bits.as_bits;
    s[0] = __double2int_rn(2.5);
    s[1] = __double2int_ru(2.5);
    s[2] = __double2int_rd(-2.5);
    s[3] = __double2int_rn(nan.as_double);
    s[4] = __double2uint_ru(1.1);
    s[5] = __double2ll_ru(0x1p63);
    s[6] = __double2ll_rd(-0x1p63);
    s[7] = (long long)__double2ull_rn(nan.as_double);
}

int main() {
    unsigned* u = nullptr;
    long long* s = nullptr;
    CHECK(cudaMalloc(&u, 10 * sizeof(unsigned)));
    CHECK(cudaMalloc(&s, 8 * sizeof(long long)));
    convert<<<1, 1>>>(u, s);
    CHECK(cudaDeviceSynchronize());
    unsigned hu[10] = {};
    long long hs[8] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
    std::printf("ftie=%u\n", hu[0]);
    std::printf("ftru=%u\n", hu[1]);
    std::printf("ftrd=%u\n", hu[2]);
    std::printf("fsub=%u\n", hu[3]);
    std::printf("fsubru=%u\n", hu[4]);
    std::printf("fmax=%u\n", hu[5]);
    std::printf("finf=%u\n", hu[6]);
    std::printf("fnan=%u\n", hu[7]);
    std::printf("fmzero=%u\n", hu[8]);
    std::printf("fru=%u\n", hu[9]);
    std::printf("irn=%lld\n", hs[0]);
    std::printf("iru=%lld\n", hs[1]);
    std::printf("ird=%lld\n", hs[2]);
    std::printf("inan=%lld\n", hs[3]);
    std::printf("uru=%lld\n", hs[4]);
    std::printf("llru=%lld\n", hs[5]);
    std::printf("llrd=%lld\n", hs[6]);
    std::printf("ullnan=%llu\n", (unsigned long long)hs[7]);
    cudaFree(u);
    cudaFree(s);
    return 0;
}
