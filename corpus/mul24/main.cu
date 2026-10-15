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

// The high 8 bits are ignored. Bit 23 is the sign of the signed form.
// The square of the minimum 24-bit value keeps only the low 32 bits, which
// are 0. The unsigned maximum product keeps the low 32 bits of a 48-bit result.
__global__ void mul24(int* s, unsigned* u) {
    s[0] = __mul24((int)0xFF000001, 2);
    s[1] = __mul24((int)0x00800000, 1);
    s[2] = __mul24((int)0x00FFFFFF, (int)0x00FFFFFF);
    s[3] = __mul24((int)0x00800000, (int)0x00800000);
    u[0] = __umul24(0x00FFFFFFu, 0x00FFFFFFu);
    u[1] = __umul24(0xFF800000u, 2u);
}

int main() {
    int* s = nullptr;
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&s, 4 * sizeof(int)));
    CHECK(cudaMalloc(&u, 2 * sizeof(unsigned)));
    mul24<<<1, 1>>>(s, u);
    CHECK(cudaDeviceSynchronize());
    int hs[4] = {};
    unsigned hu[2] = {};
    CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("hi=%d\n", hs[0]);
    std::printf("sign=%d\n", hs[1]);
    std::printf("neg1=%d\n", hs[2]);
    std::printf("sq=%d\n", hs[3]);
    std::printf("usq=%u\n", hu[0]);
    std::printf("uhi=%u\n", hu[1]);
    cudaFree(s);
    cudaFree(u);
    return 0;
}
