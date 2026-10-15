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

// The predicate is true in the first warp and false in the second.
// Each warp is uniform. A 64-lane wavefront would see a mixed predicate.
__global__ void vote(unsigned* out) {
    int pred = threadIdx.x < 32;
    unsigned ballot = __ballot_sync(0xffffffff, pred);
    int any = __any_sync(0xffffffff, pred);
    int all = __all_sync(0xffffffff, pred);
    int uni = __uni_sync(0xffffffff, pred);
    if (threadIdx.x == 0) {
        out[0] = ballot;
        out[1] = static_cast<unsigned>(any);
        out[2] = static_cast<unsigned>(all);
        out[3] = static_cast<unsigned>(uni);
    }
    if (threadIdx.x == 32) {
        out[4] = ballot;
        out[5] = static_cast<unsigned>(any);
        out[6] = static_cast<unsigned>(all);
        out[7] = static_cast<unsigned>(uni);
    }
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 8 * sizeof(unsigned)));
    CHECK(cudaMemset(out, 0, 8 * sizeof(unsigned)));
    vote<<<1, 64>>>(out);
    CHECK(cudaDeviceSynchronize());
    unsigned host[8] = {};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("lane0=%u\n", host[0]);
    std::printf("any0=%u\n", host[1]);
    std::printf("all0=%u\n", host[2]);
    std::printf("uni0=%u\n", host[3]);
    std::printf("lane32=%u\n", host[4]);
    std::printf("any32=%u\n", host[5]);
    std::printf("all32=%u\n", host[6]);
    std::printf("uni32=%u\n", host[7]);
    cudaFree(out);
    return 0;
}
