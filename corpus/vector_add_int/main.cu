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

__global__ void add(const int* x, const int* y, int* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    const int n = 128;
    int host_x[128];
    int host_y[128];
    int host_z[128];
    for (int i = 0; i < n; ++i) {
        host_x[i] = i;
        host_y[i] = 1;
    }

    int* x = nullptr;
    int* y = nullptr;
    int* z = nullptr;
    CHECK(cudaMalloc(&x, n * sizeof(int)));
    CHECK(cudaMalloc(&y, n * sizeof(int)));
    CHECK(cudaMalloc(&z, n * sizeof(int)));
    CHECK(cudaMemcpy(x, host_x, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y, host_y, n * sizeof(int), cudaMemcpyHostToDevice));
    add<<<1, n>>>(x, y, z, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(host_z, z, n * sizeof(int), cudaMemcpyDeviceToHost));

    long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += host_z[i];
    }
    std::printf("sum=%ld\n", sum);

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    return 0;
}
