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

// Halfwords and bytes, added and subtracted per lane. Plain calls wrap.
// ss saturates as signed. us saturates as unsigned.
__global__ void lanes(unsigned* out) {
    unsigned a2 = 0x7FFF0001u;
    unsigned b2 = 0x0001FFFFu;
    unsigned s2 = 0x80000000u;
    unsigned t2 = 0x00010001u;
    unsigned a4 = 0x7F0180FFu;
    unsigned b4 = 0x01010101u;
    unsigned s4 = 0x80018000u;
    unsigned t4 = 0x01010101u;
    out[0] = __vadd2(a2, b2);
    out[1] = __vaddss2(a2, b2);
    out[2] = __vaddus2(a2, b2);
    out[3] = __vsub2(s2, t2);
    out[4] = __vsubss2(s2, t2);
    out[5] = __vsubus2(s2, t2);
    out[6] = __vadd4(a4, b4);
    out[7] = __vaddss4(a4, b4);
    out[8] = __vaddus4(a4, b4);
    out[9] = __vsub4(s4, t4);
    out[10] = __vsubss4(s4, t4);
    out[11] = __vsubus4(s4, t4);
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 12 * sizeof(unsigned)));
    CHECK(cudaMemset(out, 0, 12 * sizeof(unsigned)));
    lanes<<<1, 1>>>(out);
    CHECK(cudaDeviceSynchronize());
    unsigned host[12] = {};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("add2=%u\n", host[0]);
    std::printf("addss2=%u\n", host[1]);
    std::printf("addus2=%u\n", host[2]);
    std::printf("sub2=%u\n", host[3]);
    std::printf("subss2=%u\n", host[4]);
    std::printf("subus2=%u\n", host[5]);
    std::printf("add4=%u\n", host[6]);
    std::printf("addss4=%u\n", host[7]);
    std::printf("addus4=%u\n", host[8]);
    std::printf("sub4=%u\n", host[9]);
    std::printf("subss4=%u\n", host[10]);
    std::printf("subus4=%u\n", host[11]);
    cudaFree(out);
    return 0;
}
