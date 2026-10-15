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

// Each CUDA warp reduces its own 32 lanes. A 64-lane butterfly would add
// both warps into every lane.
__global__ void reduce_lanes(unsigned* out) {
    unsigned value = threadIdx.x;
    value += __shfl_xor_sync(0xffffffff, value, 16);
    value += __shfl_xor_sync(0xffffffff, value, 8);
    value += __shfl_xor_sync(0xffffffff, value, 4);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    value += __shfl_xor_sync(0xffffffff, value, 1);
    if (threadIdx.x == 0) {
        out[0] = value;
    }
    if (threadIdx.x == 32) {
        out[1] = value;
    }
}

int main() {
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&out, 2 * sizeof(unsigned)));
    CHECK(cudaMemset(out, 0, 2 * sizeof(unsigned)));
    reduce_lanes<<<1, 64>>>(out);
    CHECK(cudaDeviceSynchronize());
    unsigned host[2] = {0, 0};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("lane0=%u\n", host[0]);
    std::printf("lane32=%u\n", host[1]);
    cudaFree(out);
    return 0;
}
