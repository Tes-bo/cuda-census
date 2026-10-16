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

__device__ __half hv(unsigned short bits) {
    volatile unsigned short held = bits;
    return __ushort_as_half(held);
}

__device__ __half2 pv(unsigned short lo, unsigned short hi) {
    return __halves2half2(hv(lo), hv(hi));
}

__device__ unsigned word(__half2 h) {
    union { __half2 as_h; unsigned as_bits; } in;
    in.as_h = h;
    return in.as_bits;
}

// Ordered inequality is false for a NaN. A true half2 lane is 1.0, and a
// true mask lane is all ones. Max keeps +0 over -0. Negating a NaN yields
// the canonical NaN.
__global__ void convert(const unsigned short* in, unsigned* u) {
    u[0] = __heq(hv(in[0]), hv(in[1]));
    u[1] = __hne(hv(in[2]), hv(in[3]));
    u[2] = __hequ(hv(in[2]), hv(in[3]));
    u[3] = __hltu(hv(in[2]), hv(in[3]));
    u[4] = __hgtu(hv(in[3]), hv(in[2]));
    u[5] = __hisnan(hv(in[4]));
    u[6] = __half_as_ushort(__hneg(hv(in[0])));
    u[7] = __half_as_ushort(__hneg(hv(in[5])));
    u[8] = __half_as_ushort(__habs(hv(in[6])));
    u[9] = word(__heq2(pv(in[3], in[2]), pv(in[3], in[3])));
    u[10] = word(__hequ2(pv(in[2], in[3]), pv(in[3], in[4])));
    u[11] = __heq2_mask(pv(in[3], in[2]), pv(in[3], in[3]));
    u[12] = __hne2_mask(pv(in[3], in[7]), pv(in[3], in[3]));
    u[13] = __hbeq2(pv(in[3], in[2]), pv(in[3], in[3]));
    u[14] = __half_as_ushort(__hmax(hv(in[2]), hv(in[3])));
    u[15] = __half_as_ushort(__hmax(hv(in[0]), hv(in[1])));
    u[16] = __half_as_ushort(__hmin(hv(in[0]), hv(in[1])));
    u[17] = word(__hmax2(pv(in[2], in[4]), pv(in[5], in[6])));
    u[18] = word(__hmin2(pv(in[2], in[0]), pv(in[3], in[1])));
    u[19] = word(__hneg2(pv(in[0], in[2])));
}

int main() {
    unsigned short bits[8] = {0x0000, 0x8000, 0x7fff, 0x3c00, 0xffff, 0x7e01, 0xfe01, 0x4000};
    unsigned short* in = nullptr;
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&in, sizeof(bits)));
    CHECK(cudaMalloc(&u, 20 * sizeof(unsigned)));
    CHECK(cudaMemcpy(in, bits, sizeof(bits), cudaMemcpyHostToDevice));
    convert<<<1, 1>>>(in, u);
    CHECK(cudaDeviceSynchronize());
    unsigned hu[20] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("eqpm=%u\n", hu[0]);
    std::printf("nenan=%u\n", hu[1]);
    std::printf("equnan=%u\n", hu[2]);
    std::printf("ltunan=%u\n", hu[3]);
    std::printf("gtunan=%u\n", hu[4]);
    std::printf("isnan=%u\n", hu[5]);
    std::printf("neg0=%u\n", hu[6]);
    std::printf("negnan=%u\n", hu[7]);
    std::printf("absnn=%u\n", hu[8]);
    std::printf("eq2=%u\n", hu[9]);
    std::printf("equ2=%u\n", hu[10]);
    std::printf("eqmask=%u\n", hu[11]);
    std::printf("nemask=%u\n", hu[12]);
    std::printf("beqone=%u\n", hu[13]);
    std::printf("maxnum=%u\n", hu[14]);
    std::printf("max0=%u\n", hu[15]);
    std::printf("min0=%u\n", hu[16]);
    std::printf("maxboth=%u\n", hu[17]);
    std::printf("minzero=%u\n", hu[18]);
    std::printf("neg2=%u\n", hu[19]);
    cudaFree(in);
    cudaFree(u);
    return 0;
}
