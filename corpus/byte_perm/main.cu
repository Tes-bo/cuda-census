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

// Selectors use three bits from each low nibble. 0x3210 returns x.
// 0x7654 returns y. Bits above bit 15 do not change that. 0x89AB keeps
// only the low three bits of each nibble, so it is the same as 0x0123.
__global__ void pick(unsigned* out) {
    unsigned x = 0xA1B2C3D4u;
    unsigned y = 0x1F2E3D4Cu;
    out[0] = __byte_perm(x, y, 0x3210u);
    out[1] = __byte_perm(x, y, 0x7654u);
    out[2] = __byte_perm(x, y, 0x32107654u);
    out[3] = __byte_perm(x, y, 0x89ABu);
    out[4] = __byte_perm(x, y, 0x0000u);
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 5 * sizeof(unsigned)));
    CHECK(cudaMemset(out, 0, 5 * sizeof(unsigned)));
    pick<<<1, 1>>>(out);
    CHECK(cudaDeviceSynchronize());
    unsigned host[5] = {};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("id=%u\n", host[0]);
    std::printf("fromy=%u\n", host[1]);
    std::printf("high=%u\n", host[2]);
    std::printf("nibble=%u\n", host[3]);
    std::printf("zero=%u\n", host[4]);
    cudaFree(out);
    return 0;
}
