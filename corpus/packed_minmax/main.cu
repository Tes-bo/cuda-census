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

// Per-lane min and max. s compares a lane as signed. u compares it as
// unsigned. 0x8000 is -32768 on one and 32768 on the other.
__global__ void pick(unsigned* out) {
    unsigned a2 = 0x80000001u;
    unsigned b2 = 0x7FFF0000u;
    unsigned a4 = 0x8001807Fu;
    unsigned b4 = 0x7F017F00u;
    out[0] = __vmins2(a2, b2);
    out[1] = __vminu2(a2, b2);
    out[2] = __vmaxs2(a2, b2);
    out[3] = __vmaxu2(a2, b2);
    out[4] = __vmins4(a4, b4);
    out[5] = __vminu4(a4, b4);
    out[6] = __vmaxs4(a4, b4);
    out[7] = __vmaxu4(a4, b4);
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 8 * sizeof(unsigned)));
    CHECK(cudaMemset(out, 0, 8 * sizeof(unsigned)));
    pick<<<1, 1>>>(out);
    CHECK(cudaDeviceSynchronize());
    unsigned host[8] = {};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("mins2=%u\n", host[0]);
    std::printf("minu2=%u\n", host[1]);
    std::printf("maxs2=%u\n", host[2]);
    std::printf("maxu2=%u\n", host[3]);
    std::printf("mins4=%u\n", host[4]);
    std::printf("minu4=%u\n", host[5]);
    std::printf("maxs4=%u\n", host[6]);
    std::printf("maxu4=%u\n", host[7]);
    cudaFree(out);
    return 0;
}
