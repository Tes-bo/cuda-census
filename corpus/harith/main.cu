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

// +0 plus -0 is +0. A NaN is 0x7fff. The fused square residual is 2^-20;
// the separate product plus its negation cancels. Saturate clamps to [0, 1].
__global__ void convert(unsigned* u) {
    __half p0 = __ushort_as_half(0x0000);
    __half m0 = __ushort_as_half(0x8000);
    __half one = __ushort_as_half(0x3c00);
    __half nxt = __ushort_as_half(0x3c01);
    __half big = __ushort_as_half(0x7bff);
    __half nan = __ushort_as_half(0x7fff);
    __half nnan = __ushort_as_half(0xffff);
    __half pinf = __ushort_as_half(0x7c00);
    __half ninf = __ushort_as_half(0xfc00);
    volatile __half prod = __hmul(nxt, nxt);
    u[0] = __half_as_ushort(__hadd(p0, m0));
    u[1] = __half_as_ushort(__hadd(m0, m0));
    u[2] = __half_as_ushort(__hadd_rn(p0, m0));
    u[3] = __half_as_ushort(__hmul(m0, p0));
    u[4] = __half_as_ushort(__hmul_rn(m0, m0));
    u[5] = __half_as_ushort(__hadd(nan, one));
    u[6] = __half_as_ushort(__hadd(nnan, one));
    u[7] = __half_as_ushort(__hadd(pinf, ninf));
    u[8] = __half_as_ushort(__hmul(pinf, p0));
    u[9] = __half_as_ushort(__hadd(big, big));
    u[10] = __half_as_ushort(__hsub(m0, p0));
    u[11] = __half_as_ushort(prod);
    u[12] = __half_as_ushort(__hadd(prod, __hneg(prod)));
    u[13] = __half_as_ushort(__hfma(nxt, nxt, __hneg(prod)));
    u[14] = __half_as_ushort(__hadd_sat(__ushort_as_half(0x4000), one));
    u[15] = __half_as_ushort(__hadd_sat(__ushort_as_half(0xbc00), one));
    u[16] = __half_as_ushort(__hadd_sat(nan, one));
    u[17] = __half_as_ushort(__hadd_sat(m0, m0));
    u[18] = __half_as_ushort(__hadd_sat(pinf, p0));
    u[19] = __half_as_ushort(__hfma_sat(nxt, nxt, __ushort_as_half(0xbc00)));
}

int main() {
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&u, 20 * sizeof(unsigned)));
    convert<<<1, 1>>>(u);
    CHECK(cudaDeviceSynchronize());
    unsigned hu[20] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("zpn=%u\n", hu[0]);
    std::printf("znn=%u\n", hu[1]);
    std::printf("zrn=%u\n", hu[2]);
    std::printf("mzero=%u\n", hu[3]);
    std::printf("mpos=%u\n", hu[4]);
    std::printf("nan=%u\n", hu[5]);
    std::printf("nnan=%u\n", hu[6]);
    std::printf("infsub=%u\n", hu[7]);
    std::printf("inf0=%u\n", hu[8]);
    std::printf("overflow=%u\n", hu[9]);
    std::printf("subm0=%u\n", hu[10]);
    std::printf("prod=%u\n", hu[11]);
    std::printf("cancel=%u\n", hu[12]);
    std::printf("residual=%u\n", hu[13]);
    std::printf("sat3=%u\n", hu[14]);
    std::printf("satneg=%u\n", hu[15]);
    std::printf("satnan=%u\n", hu[16]);
    std::printf("satm0=%u\n", hu[17]);
    std::printf("satinf=%u\n", hu[18]);
    std::printf("fmasat=%u\n", hu[19]);
    cudaFree(u);
    return 0;
}
