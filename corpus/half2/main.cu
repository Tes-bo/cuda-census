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

__device__ __half2 lanes(unsigned short lo, unsigned short hi) {
    return __halves2half2(__ushort_as_half(lo), __ushort_as_half(hi));
}

__device__ unsigned word(__half2 h) {
    union { __half2 as_h; unsigned as_bits; } in;
    in.as_h = h;
    return in.as_bits;
}

// The low lane is the low 16 bits. A NaN stays in its own lane. The fused
// residual of (1+2^-10) squared is 2^-20 in the low lane; the high lane of
// an exact 1 cancels. The first float argument is the low lane.
__global__ void convert(unsigned* u) {
    __half2 p0m0 = lanes(0x0000, 0x8000);
    __half2 m0p0 = lanes(0x8000, 0x0000);
    __half2 m0m0 = lanes(0x8000, 0x8000);
    __half2 one = lanes(0x3c00, 0x3c00);
    __half2 nanone = lanes(0x7fff, 0x3c00);
    __half2 nxt = lanes(0x3c01, 0x3c00);
    __half2 big = lanes(0x7bff, 0x3c00);
    __half2 halfone = lanes(0x3800, 0x3c00);
    __half2 twoone = lanes(0x4000, 0x3c00);
    __half2 pair = lanes(0x3c01, 0x3c02);
    __half2 left = lanes(0x0001, 0x0002);
    __half2 right = lanes(0x0003, 0x0004);
    volatile unsigned prod_bits = word(__hmul2(nxt, nxt));
    unsigned pb = prod_bits;
    __half2 prod = lanes((unsigned short)pb, (unsigned short)(pb >> 16));
    __half2 neg = lanes((unsigned short)((pb & 0xffffu) ^ 0x8000u), (unsigned short)((pb >> 16) ^ 0x8000u));
    float2 back = __half22float2(lanes(0x3c00, 0xbc00));
    float2 wide = make_float2(1.0f + 0x1p-11f, 65520.0f);
    u[0] = word(__hadd2(p0m0, m0p0));
    u[1] = word(__hadd2(m0m0, m0m0));
    u[2] = word(__hadd2_rn(p0m0, m0p0));
    u[3] = word(__hadd2(nanone, one));
    u[4] = word(__hmul2(m0p0, p0m0));
    u[5] = word(__hadd2(big, big));
    u[6] = word(__hsub2(m0p0, p0m0));
    u[7] = word(__hadd2(prod, neg));
    u[8] = word(__hfma2(nxt, nxt, neg));
    u[9] = word(__hadd2_sat(lanes(0x7fff, 0x8000), one));
    u[10] = word(__hmul2_sat(halfone, one));
    u[11] = word(__hadd2_sat(twoone, one));
    u[12] = word(__float2half2_rn(1.0f + 0x1p-11f));
    u[13] = word(__floats2half2_rn(1.0f + 0x1p-11f, 65520.0f));
    u[14] = word(__float22half2_rn(wide));
    u[15] = __float_as_uint(back.x);
    u[16] = __float_as_uint(back.y);
    u[17] = word(__low2half2(pair));
    u[18] = word(__lows2half2(left, right));
    u[19] = word(__highs2half2(left, right));
}

int main() {
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&u, 20 * sizeof(unsigned)));
    convert<<<1, 1>>>(u);
    CHECK(cudaDeviceSynchronize());
    unsigned hu[20] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("addmix=%u\n", hu[0]);
    std::printf("addmm=%u\n", hu[1]);
    std::printf("addrn=%u\n", hu[2]);
    std::printf("nanlo=%u\n", hu[3]);
    std::printf("mulsign=%u\n", hu[4]);
    std::printf("overflow=%u\n", hu[5]);
    std::printf("subm0=%u\n", hu[6]);
    std::printf("cancel=%u\n", hu[7]);
    std::printf("residual=%u\n", hu[8]);
    std::printf("satnan=%u\n", hu[9]);
    std::printf("sathalf=%u\n", hu[10]);
    std::printf("satone=%u\n", hu[11]);
    std::printf("dup=%u\n", hu[12]);
    std::printf("split=%u\n", hu[13]);
    std::printf("split2=%u\n", hu[14]);
    std::printf("flo=%u\n", hu[15]);
    std::printf("fhi=%u\n", hu[16]);
    std::printf("splatlo=%u\n", hu[17]);
    std::printf("lows=%u\n", hu[18]);
    std::printf("highs=%u\n", hu[19]);
    cudaFree(u);
    return 0;
}
