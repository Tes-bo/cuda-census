// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK(cmd)                                                         \
    do {                                                                   \
        cudaError_t err_ = (cmd);                                          \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s\n", cudaGetErrorString(err_));        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// 2049 is a halfway tie at 2^11. 65520 rounds to infinity, and toward
// zero stays finite. A double just above 2^-25 is the smallest half
// subnormal; the same bits through float are zero.
__global__ void convert(unsigned* u) {
    union {
        double as_double;
        unsigned long long as_bits;
    } above, nabove, nan, nnan, mzero;
    above.as_bits = 0x3e60000000000011ull;
    nabove.as_bits = 0xbe60000000000011ull;
    nan.as_bits = 0x7ff8000000000001ull;
    nnan.as_bits = 0xfff8000000000001ull;
    mzero.as_bits = 0x8000000000000000ull;
    u[0] = __half_as_ushort(__int2half_rn(2049));
    u[1] = __half_as_ushort(__int2half_ru(2049));
    u[2] = __half_as_ushort(__int2half_rd(-2049));
    u[3] = __half_as_ushort(__int2half_rn(65520));
    u[4] = __half_as_ushort(__int2half_rz(65520));
    u[5] = __half_as_ushort(__int2half_rd(-65520));
    u[6] = __half_as_ushort(__int2half_rz(-65520));
    u[7] = __half_as_ushort(__int2half_rn(-65519));
    u[8] = __half_as_ushort(__short2half_rn((short)-32768));
    u[9] = __half_as_ushort(__ll2half_rn((long long)0x8000000000000000ull));
    u[10] = __half_as_ushort(__ll2half_rz((long long)0x8000000000000000ull));
    u[11] = __half_as_ushort(__ull2half_rz(~0ull));
    u[12] = __half_as_ushort(__ull2half_rn(~0ull));
    u[13] = __half_as_ushort(__double2half(above.as_double));
    u[14] = __half_as_ushort(__float2half_rn((float)above.as_double));
    u[15] = __half_as_ushort(__double2half(nabove.as_double));
    u[16] = __half_as_ushort(__double2half(0x1p-25));
    u[17] = __half_as_ushort(__double2half(mzero.as_double));
    u[18] = __half_as_ushort(__double2half(nan.as_double));
    u[19] = __half_as_ushort(__double2half(nnan.as_double));
}

int main() {
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&u, 20 * sizeof(unsigned)));
    convert<<<1, 1>>>(u);
    CHECK(cudaDeviceSynchronize());
    unsigned hu[20] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("i2049=%u\n", hu[0]);
    std::printf("i2049ru=%u\n", hu[1]);
    std::printf("i2049rd=%u\n", hu[2]);
    std::printf("i65520=%u\n", hu[3]);
    std::printf("i65520rz=%u\n", hu[4]);
    std::printf("n65520rd=%u\n", hu[5]);
    std::printf("n65520rz=%u\n", hu[6]);
    std::printf("n65519=%u\n", hu[7]);
    std::printf("smin=%u\n", hu[8]);
    std::printf("llmin=%u\n", hu[9]);
    std::printf("llminrz=%u\n", hu[10]);
    std::printf("ullmaxrz=%u\n", hu[11]);
    std::printf("ullmax=%u\n", hu[12]);
    std::printf("dabove=%u\n", hu[13]);
    std::printf("dvia=%u\n", hu[14]);
    std::printf("dnabove=%u\n", hu[15]);
    std::printf("dtie=%u\n", hu[16]);
    std::printf("dmzero=%u\n", hu[17]);
    std::printf("dnan=%u\n", hu[18]);
    std::printf("dnnan=%u\n", hu[19]);
    cudaFree(u);
    return 0;
}
