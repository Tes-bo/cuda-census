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

// Signed lanes round a half away from zero: 1 and 2 become 2, and
// -2 and -3 become -3. Two minimum lanes stay the minimum. The plain
// unsigned average of 0xFFFF and 1 is 0x8000, not a wrapped 0.
__global__ void packed_avg(unsigned* out) {
    out[0] = __vavgs2(0x00010001u, 0x00020002u);
    out[1] = __vavgs2(0xFFFDFFFEu, 0xFFFEFFFDu);
    out[2] = __vavgs2(0x80008000u, 0x80008000u);
    out[3] = __vavgu2(0x00010001u, 0x00020002u);
    out[4] = __vhaddu2(0x00010001u, 0x00020002u);
    out[5] = __vhaddu2(0xFFFFFFFFu, 0x00010001u);
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 6 * sizeof(unsigned)));
    packed_avg<<<1, 1>>>(out);
    CHECK(cudaDeviceSynchronize());
    unsigned h[6] = {};
    CHECK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
    std::printf("spos=%u\n", h[0]);
    std::printf("sneg=%u\n", h[1]);
    std::printf("smin=%u\n", h[2]);
    std::printf("uround=%u\n", h[3]);
    std::printf("uplain=%u\n", h[4]);
    std::printf("uhalf=%u\n", h[5]);
    cudaFree(out);
    return 0;
}
