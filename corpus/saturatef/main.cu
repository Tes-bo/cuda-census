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

// Values outside [0, 1] clamp. A NaN and a negative zero become +0.
__global__ void clamp(unsigned* u) {
    union {
        float as_float;
        unsigned as_bits;
    } nan, bits;
    nan.as_bits = 0x7fc00000u;
    bits.as_float = __saturatef(-1.0f);
    u[0] = bits.as_bits;
    bits.as_float = __saturatef(2.0f);
    u[1] = bits.as_bits;
    bits.as_float = __saturatef(0.5f);
    u[2] = bits.as_bits;
    bits.as_float = __saturatef(-0.0f);
    u[3] = bits.as_bits;
    bits.as_float = __saturatef(nan.as_float);
    u[4] = bits.as_bits;
}

int main() {
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&u, 5 * sizeof(unsigned)));
    clamp<<<1, 1>>>(u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned hu[5] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("neg=%u\n", hu[0]);
    std::printf("two=%u\n", hu[1]);
    std::printf("half=%u\n", hu[2]);
    std::printf("mzero=%u\n", hu[3]);
    std::printf("nan=%u\n", hu[4]);
    cudaFree(u);
    return 0;
}
