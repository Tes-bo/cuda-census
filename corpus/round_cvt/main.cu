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

// rn ties 2.5 to the even integer 2. ru sends that tie to 3, and rd sends
// -2.5 to -3. A magnitude past 2^31 saturates. A NaN becomes 0.
// 16777217 sits halfway between two floats, so ru takes the upper one.
// 0xFFFFFFFF is closer to 2^32 than to the previous float, so rd steps down.
__global__ void convert(int* s, unsigned* u) {
    s[0] = __float2int_rn(2.5f);
    s[1] = __float2int_ru(2.5f);
    s[2] = __float2int_rd(-2.5f);
    s[3] = __float2int_rn(1.0e20f);
    s[4] = __float2int_rn(-1.0e20f);
    union {
        float as_float;
        unsigned as_bits;
    } nan, bits;
    nan.as_bits = 0x7fc00000u;
    s[5] = __float2int_rn(nan.as_float);
    u[0] = __float2uint_rn(-1.2f);
    u[1] = __float2uint_rz(1.0e20f);
    u[2] = __float2uint_ru(0.2f);
    bits.as_float = __int2float_ru(16777217);
    u[3] = bits.as_bits;
    bits.as_float = __int2float_rd(-16777219);
    u[4] = bits.as_bits;
    bits.as_float = __int2float_rd(2147483647);
    u[5] = bits.as_bits;
    bits.as_float = __uint2float_rd(0xffffffffu);
    u[6] = bits.as_bits;
}

int main() {
    int* s = nullptr;
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&s, 6 * sizeof(int)));
    CHECK(cudaMalloc(&u, 7 * sizeof(unsigned)));
    convert<<<1, 1>>>(s, u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int hs[6] = {};
    unsigned hu[7] = {};
    CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("rn25=%d\n", hs[0]);
    std::printf("ru25=%d\n", hs[1]);
    std::printf("rdn=%d\n", hs[2]);
    std::printf("big=%d\n", hs[3]);
    std::printf("nbig=%d\n", hs[4]);
    std::printf("nan=%d\n", hs[5]);
    std::printf("uneg=%u\n", hu[0]);
    std::printf("ubig=%u\n", hu[1]);
    std::printf("uru=%u\n", hu[2]);
    std::printf("up=%u\n", hu[3]);
    std::printf("nrd=%u\n", hu[4]);
    std::printf("imaxrd=%u\n", hu[5]);
    std::printf("urd=%u\n", hu[6]);
    cudaFree(s);
    cudaFree(u);
    return 0;
}
