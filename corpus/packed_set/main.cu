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

// A true halfword lane is 0x0001, not 0xFFFF. A true byte lane is 0x01.
// A lane that fails stays zero; the other lanes are not cleared.
__global__ void packed_set(unsigned* out) {
    unsigned s = 0x80000001u;
    unsigned t = 0x7FFF0000u;
    unsigned c = 0x8001807Fu;
    unsigned d = 0x7F017F00u;
    out[0] = __vseteq2(0x1234aba5u, 0x1234aba6u);
    out[1] = __vsetne2(0x1234aba5u, 0x1234aba6u);
    out[2] = __vsetgts2(s, t);
    out[3] = __vsetgtu2(s, t);
    out[4] = __vsetgts4(c, d);
    out[5] = __vsetgtu4(c, d);
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 6 * sizeof(unsigned)));
    packed_set<<<1, 1>>>(out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned h[6] = {};
    CHECK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
    std::printf("eq=%u\n", h[0]);
    std::printf("ne=%u\n", h[1]);
    std::printf("gts=%u\n", h[2]);
    std::printf("gtu=%u\n", h[3]);
    std::printf("gts4=%u\n", h[4]);
    std::printf("gtu4=%u\n", h[5]);
    cudaFree(out);
    return 0;
}
