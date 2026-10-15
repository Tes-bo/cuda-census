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

// Per-lane absolute value, negation, and absolute difference. Plain abs
// keeps the minimum lane. Saturating abs turns it into the maximum.
// Halfword saturating negation keeps 0x8000. Byte saturating negation
// turns 0x80 into 0x7F.
__global__ void magnitude(unsigned* out) {
    unsigned a2 = 0x80000002u;
    unsigned a4 = 0x80017F80u;
    unsigned x2 = 0x8000FFFFu;
    unsigned y2 = 0x00000001u;
    unsigned x4 = 0x01800080u;
    unsigned y4 = 0x00010100u;
    out[0] = __vabs2(a2);
    out[1] = __vabsss2(a2);
    out[2] = __vneg2(a2);
    out[3] = __vnegss2(a2);
    out[4] = __vabs4(a4);
    out[5] = __vabsss4(a4);
    out[6] = __vneg4(a4);
    out[7] = __vnegss4(a4);
    out[8] = __vabsdiffs2(x2, y2);
    out[9] = __vabsdiffu2(x2, y2);
    out[10] = __vabsdiffs4(x4, y4);
    out[11] = __vabsdiffu4(x4, y4);
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 12 * sizeof(unsigned)));
    CHECK(cudaMemset(out, 0, 12 * sizeof(unsigned)));
    magnitude<<<1, 1>>>(out);
    CHECK(cudaDeviceSynchronize());
    unsigned host[12] = {};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("abs2=%u\n", host[0]);
    std::printf("absss2=%u\n", host[1]);
    std::printf("neg2=%u\n", host[2]);
    std::printf("negss2=%u\n", host[3]);
    std::printf("abs4=%u\n", host[4]);
    std::printf("absss4=%u\n", host[5]);
    std::printf("neg4=%u\n", host[6]);
    std::printf("negss4=%u\n", host[7]);
    std::printf("diffs2=%u\n", host[8]);
    std::printf("diffu2=%u\n", host[9]);
    std::printf("diffs4=%u\n", host[10]);
    std::printf("diffu4=%u\n", host[11]);
    cudaFree(out);
    return 0;
}
