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

__global__ void exchange(int* out) {
    __shared__ int values[32];
    int lane = threadIdx.x;
    values[lane] = lane;
    __syncwarp();
    if (lane == 0) {
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            sum += values[i];
        }
        *out = sum;
    }
}

int main() {
    int* out = nullptr;
    CHECK(cudaMalloc(&out, sizeof(int)));
    CHECK(cudaMemset(out, 0, sizeof(int)));
    exchange<<<1, 32>>>(out);
    CHECK(cudaDeviceSynchronize());
    int sum = 0;
    CHECK(cudaMemcpy(&sum, out, sizeof(sum), cudaMemcpyDeviceToHost));
    std::printf("sum=%d\n", sum);
    cudaFree(out);
    return 0;
}
