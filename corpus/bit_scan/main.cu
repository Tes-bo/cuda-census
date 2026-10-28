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

// Leading zeros of zero fill the width. The first set bit is counted
// from 1, and zero has none. Reversing 1 puts that bit in the sign place.
__global__ void bit_scan(int* s, unsigned* u, unsigned long long* z) {
    s[0] = __clz(0);
    s[1] = __clz((int)0x80000000);
    s[2] = __ffs(0);
    s[3] = __ffs((int)0x80000000);
    s[4] = __clzll(0);
    s[5] = __popc(0xffffffffu);
    u[0] = __brev(1u);
    z[0] = __brevll(1ull);
}

int main() {
    int* s = nullptr;
    unsigned* u = nullptr;
    unsigned long long* z = nullptr;
    CHECK(cudaMalloc(&s, 6 * sizeof(int)));
    CHECK(cudaMalloc(&u, sizeof(unsigned)));
    CHECK(cudaMalloc(&z, sizeof(unsigned long long)));
    bit_scan<<<1, 1>>>(s, u, z);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int hs[6] = {};
    unsigned hu = 0;
    unsigned long long hz = 0;
    CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hz, z, sizeof(hz), cudaMemcpyDeviceToHost));
    std::printf("clz0=%d\n", hs[0]);
    std::printf("clzmin=%d\n", hs[1]);
    std::printf("ffs0=%d\n", hs[2]);
    std::printf("ffsmin=%d\n", hs[3]);
    std::printf("clzll0=%d\n", hs[4]);
    std::printf("popc=%d\n", hs[5]);
    std::printf("brev1=%u\n", hu);
    std::printf("brevll=%llu\n", hz);
    cudaFree(s);
    cudaFree(u);
    cudaFree(z);
    return 0;
}
