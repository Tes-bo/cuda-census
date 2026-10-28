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

__global__ void read_width(int* out) {
    if (threadIdx.x == 0) {
        *out = warpSize;
    }
}

int main() {
    int* out = nullptr;
    CHECK(cudaMalloc(&out, sizeof(int)));
    CHECK(cudaMemset(out, 0, sizeof(int)));
    read_width<<<1, 32>>>(out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int width = 0;
    CHECK(cudaMemcpy(&width, out, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("width=%d\n", width);
    cudaFree(out);
    return 0;
}
