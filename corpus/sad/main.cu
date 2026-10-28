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

// Absolute difference plus the third argument. The signed form widens,
// so the absolute value of INT_MIN is 2^31 and |INT_MIN - 1| is 2^31 + 1.
// The same all-ones bits are 1 when signed and 2^32 - 1 when unsigned.
// The sum wraps modulo 2^32.
__global__ void sad(unsigned* out) {
    int a = (int)0x80000000;
    out[0] = __sad(-1, 1, 0u);
    out[1] = __sad(a, 0, 0u);
    out[2] = __sad(a, 1, 0u);
    out[3] = __sad(1, 0, 0xffffffffu);
    out[4] = __usad(0xffffffffu, 0u, 0u);
    out[5] = __usad(0u, 1u, 0xffffffffu);
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 6 * sizeof(unsigned)));
    sad<<<1, 1>>>(out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned h[6] = {};
    CHECK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
    std::printf("sneg=%u\n", h[0]);
    std::printf("smin=%u\n", h[1]);
    std::printf("smin1=%u\n", h[2]);
    std::printf("swrap=%u\n", h[3]);
    std::printf("umax=%u\n", h[4]);
    std::printf("uwrap=%u\n", h[5]);
    cudaFree(out);
    return 0;
}
