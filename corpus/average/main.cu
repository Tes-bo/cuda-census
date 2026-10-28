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

// (x + y) >> 1, and the rounded form adds one first. The sum does not
// overflow: two INT_MIN values stay negative, and 0xFFFFFFFF with 1 is 2^31.
__global__ void average(int* s, unsigned* u) {
    s[0] = __hadd(1, 2);
    s[1] = __rhadd(1, 2);
    s[2] = __hadd(-3, -2);
    s[3] = __rhadd(-3, -2);
    s[4] = __hadd((int)0x80000000, (int)0x80000000);
    u[0] = __uhadd(0xffffffffu, 1u);
}

int main() {
    int* s = nullptr;
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&s, 5 * sizeof(int)));
    CHECK(cudaMalloc(&u, sizeof(unsigned)));
    CHECK(cudaMemset(s, 0, 5 * sizeof(int)));
    CHECK(cudaMemset(u, 0, sizeof(unsigned)));
    average<<<1, 1>>>(s, u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int hs[5] = {};
    unsigned hu = 0;
    CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("h12=%d\n", hs[0]);
    std::printf("r12=%d\n", hs[1]);
    std::printf("hneg=%d\n", hs[2]);
    std::printf("rneg=%d\n", hs[3]);
    std::printf("hmin=%d\n", hs[4]);
    std::printf("uhwrap=%u\n", hu);
    cudaFree(s);
    cudaFree(u);
    return 0;
}
