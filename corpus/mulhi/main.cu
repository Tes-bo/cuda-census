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

// High half of the product. INT_MIN times itself is 2^62, so the high
// 32 bits are 2^30. A 32-bit multiply of those same bits wraps to 0.
// A small negative 64-bit product sign-extends. The unsigned maximum
// product keeps a high half of 2^64 - 2.
__global__ void mulhi(int* s, unsigned* u, long long* w, unsigned long long* z) {
    int a = (int)0x80000000;
    long long m = (long long)0x8000000000000000ull;
    s[0] = __mulhi(a, 1);
    s[1] = __mulhi(a, -1);
    s[2] = __mulhi(a, a);
    u[0] = __umulhi(0xffffffffu, 0xffffffffu);
    w[0] = __mul64hi(-3, 5);
    w[1] = __mul64hi(m, m);
    z[0] = __umul64hi(~0ull, ~0ull);
}

int main() {
    int* s = nullptr;
    unsigned* u = nullptr;
    long long* w = nullptr;
    unsigned long long* z = nullptr;
    CHECK(cudaMalloc(&s, 3 * sizeof(int)));
    CHECK(cudaMalloc(&u, sizeof(unsigned)));
    CHECK(cudaMalloc(&w, 2 * sizeof(long long)));
    CHECK(cudaMalloc(&z, sizeof(unsigned long long)));
    mulhi<<<1, 1>>>(s, u, w, z);
    CHECK(cudaDeviceSynchronize());
    int hs[3] = {};
    unsigned hu = 0;
    long long hw[2] = {};
    unsigned long long hz = 0;
    CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hw, w, sizeof(hw), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hz, z, sizeof(hz), cudaMemcpyDeviceToHost));
    std::printf("min1=%d\n", hs[0]);
    std::printf("minneg=%d\n", hs[1]);
    std::printf("minsq=%d\n", hs[2]);
    std::printf("umax=%u\n", hu);
    std::printf("neg64=%lld\n", hw[0]);
    std::printf("sq64=%lld\n", hw[1]);
    std::printf("umax64=%llu\n", hz);
    cudaFree(s);
    cudaFree(u);
    cudaFree(w);
    cudaFree(z);
    return 0;
}
