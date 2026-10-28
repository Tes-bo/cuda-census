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

// Four-byte dot product. 0xFF is -1 signed and 255 unsigned. 0x80 is
// -128 signed and 128 unsigned.
__global__ void dot4(int* s, unsigned* u) {
    int a = 0x0101FF01;
    int b = 0x01010101;
    unsigned ua = 0x0101FF01u;
    unsigned ub = 0x01010101u;
    s[0] = __dp4a(0x01010101, 0x01010101, 0);
    s[1] = __dp4a(a, b, 0);
    s[2] = __dp4a((int)0x80808080, b, 0);
    u[0] = __dp4a(ua, ub, 0u);
    u[1] = __dp4a(0x80808080u, ub, 0u);
}

int main() {
    int* s = nullptr;
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&s, 3 * sizeof(int)));
    CHECK(cudaMalloc(&u, 2 * sizeof(unsigned)));
    CHECK(cudaMemset(s, 0, 3 * sizeof(int)));
    CHECK(cudaMemset(u, 0, 2 * sizeof(unsigned)));
    dot4<<<1, 1>>>(s, u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int hs[3] = {};
    unsigned hu[2] = {};
    CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("ones=%d\n", hs[0]);
    std::printf("s=%d\n", hs[1]);
    std::printf("s80=%d\n", hs[2]);
    std::printf("u=%u\n", hu[0]);
    std::printf("u80=%u\n", hu[1]);
    cudaFree(s);
    cudaFree(u);
    return 0;
}
