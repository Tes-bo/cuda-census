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

__global__ void count_threads(int* out) {
    __shared__ int tally;
    if (threadIdx.x == 0) {
        tally = 0;
    }
    __syncthreads();
    atomicAdd(&tally, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        *out = tally;
    }
}

int main() {
    int* out = nullptr;
    CHECK(cudaMalloc(&out, sizeof(int)));
    CHECK(cudaMemset(out, 0, sizeof(int)));
    count_threads<<<1, 128>>>(out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int total = 0;
    CHECK(cudaMemcpy(&total, out, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("total=%d\n", total);
    cudaFree(out);
    return 0;
}
