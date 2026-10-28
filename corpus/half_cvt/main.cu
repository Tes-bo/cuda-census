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

// Float to half keeps a halfway tie on the even half. Past 65504, toward
// zero stays at the finite maximum. A NaN becomes 0x7fff with the sign
// dropped. Widening a half NaN becomes the float 0x7fffffff.
__global__ void convert(unsigned* u) {
    union {
        float as_float;
        unsigned as_bits;
    } nan, nnan;
    float tie = 1.0f + 0x1p-11f;
    nan.as_bits = 0x7fc00000u;
    nnan.as_bits = 0xffc00000u;
    u[0] = __half_as_ushort(__float2half(tie));
    u[1] = __half_as_ushort(__float2half_ru(tie));
    u[2] = __half_as_ushort(__float2half_rd(-tie));
    u[3] = __half_as_ushort(__float2half_rn(0x1p-25f));
    u[4] = __half_as_ushort(__float2half_ru(0x1p-25f));
    u[5] = __half_as_ushort(__float2half_rz(65520.0f));
    u[6] = __half_as_ushort(__float2half_ru(65520.0f));
    u[7] = __half_as_ushort(__float2half_rn(nan.as_float));
    u[8] = __half_as_ushort(__float2half_rn(nnan.as_float));
    u[9] = __half_as_ushort(__float2half_rd(-0.0f));
    u[10] = __half_as_ushort(__float2half_rn(2049.0f));
    u[11] = __half_as_ushort(__float2half_ru(2049.0f));
    u[12] = __float_as_uint(__half2float(__ushort_as_half(0x3c00)));
    u[13] = __float_as_uint(__half2float(__ushort_as_half(0x8000)));
    u[14] = __float_as_uint(__half2float(__ushort_as_half(0x7fff)));
    u[15] = __float_as_uint(__half2float(__ushort_as_half(0x0001)));
    u[16] = __float_as_uint(__half2float(__ushort_as_half(0x7bff)));
    u[17] = __float_as_uint(__half2float(__ushort_as_half(0xfc00)));
}

int main() {
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&u, 18 * sizeof(unsigned)));
    convert<<<1, 1>>>(u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned hu[18] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("halftie=%u\n", hu[0]);
    std::printf("halfru=%u\n", hu[1]);
    std::printf("halfrd=%u\n", hu[2]);
    std::printf("halfsub=%u\n", hu[3]);
    std::printf("halfsubru=%u\n", hu[4]);
    std::printf("halfmax=%u\n", hu[5]);
    std::printf("halfinf=%u\n", hu[6]);
    std::printf("halfnan=%u\n", hu[7]);
    std::printf("halfnnan=%u\n", hu[8]);
    std::printf("halfmzero=%u\n", hu[9]);
    std::printf("halfwide=%u\n", hu[10]);
    std::printf("halfwideru=%u\n", hu[11]);
    std::printf("fone=%u\n", hu[12]);
    std::printf("fmzero=%u\n", hu[13]);
    std::printf("fnan=%u\n", hu[14]);
    std::printf("fsub=%u\n", hu[15]);
    std::printf("fmax=%u\n", hu[16]);
    std::printf("finf=%u\n", hu[17]);
    cudaFree(u);
    return 0;
}
