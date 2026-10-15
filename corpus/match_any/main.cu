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

// 64 threads are two CUDA warps. Keys repeat every 32 lanes, so a match
// that followed a 64-lane wavefront would see both copies.
__global__ void match_keys(unsigned* out) {
    unsigned key = threadIdx.x & 31u;
    int pred = 0;
    unsigned any = __match_any_sync(0xffffffff, key);
    unsigned all = __match_all_sync(0xffffffff, key, &pred);
    if (threadIdx.x == 0) {
        out[0] = any;
        out[1] = all;
        out[2] = static_cast<unsigned>(pred);
    }
    if (threadIdx.x == 32) {
        out[3] = any;
    }
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 4 * sizeof(unsigned)));
    CHECK(cudaMemset(out, 0, 4 * sizeof(unsigned)));
    match_keys<<<1, 64>>>(out);
    CHECK(cudaDeviceSynchronize());
    unsigned host[4] = {0, 0, 0, 0};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("lane0=%u\n", host[0]);
    std::printf("all=%u\n", host[1]);
    std::printf("pred=%u\n", host[2]);
    std::printf("lane32=%u\n", host[3]);
    cudaFree(out);
    return 0;
}
