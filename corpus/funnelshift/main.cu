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

// Wrap uses shift & 31, so 32 matches 0. Clamp uses min(shift, 32), so
// left-32 returns lo and right-32 returns hi.
__global__ void shift_bits(unsigned* out) {
    unsigned lo = 0xA5A5A5A5u;
    unsigned hi = 0x3C3C3C3Cu;
    out[0] = __funnelshift_l(lo, hi, 1);
    out[1] = __funnelshift_l(lo, hi, 32);
    out[2] = __funnelshift_lc(lo, hi, 32);
    out[3] = __funnelshift_r(lo, hi, 1);
    out[4] = __funnelshift_rc(lo, hi, 32);
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 5 * sizeof(unsigned)));
    CHECK(cudaMemset(out, 0, 5 * sizeof(unsigned)));
    shift_bits<<<1, 1>>>(out);
    CHECK(cudaDeviceSynchronize());
    unsigned host[5] = {};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("l1=%u\n", host[0]);
    std::printf("l32=%u\n", host[1]);
    std::printf("lc32=%u\n", host[2]);
    std::printf("r1=%u\n", host[3]);
    std::printf("rc32=%u\n", host[4]);
    cudaFree(out);
    return 0;
}
