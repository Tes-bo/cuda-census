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

// Sum of the per-lane absolute differences, returned as a scalar.
// A signed 0xFFFF lane is 1. The same bits unsigned are 65535.
// A 0x8000 lane contributes 32768, so two of them sum to 65536.
__global__ void packed_sad(unsigned* out) {
    unsigned a2 = 0xFFFF0001u;
    unsigned m2 = 0x80008000u;
    unsigned left = 0x00020001u;
    unsigned right = 0x00010003u;
    unsigned a4 = 0xFF0180FFu;
    out[0] = __vsads2(a2, 0u);
    out[1] = __vsadu2(a2, 0u);
    out[2] = __vsads2(m2, 0u);
    out[3] = __vsads2(left, right);
    out[4] = __vsads4(a4, 0u);
    out[5] = __vsadu4(a4, 0u);
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 6 * sizeof(unsigned)));
    packed_sad<<<1, 1>>>(out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned h[6] = {};
    CHECK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
    std::printf("s2=%u\n", h[0]);
    std::printf("u2=%u\n", h[1]);
    std::printf("smin=%u\n", h[2]);
    std::printf("pair=%u\n", h[3]);
    std::printf("s4=%u\n", h[4]);
    std::printf("u4=%u\n", h[5]);
    cudaFree(out);
    return 0;
}
