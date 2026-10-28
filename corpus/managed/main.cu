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

__managed__ int bumps = 5;

// The host writes managed memory directly, the device updates it, and the
// host reads it back after a synchronize, with no cudaMemcpy.
__global__ void touch(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2 + 1;
        atomicAdd(&bumps, 1);
    }
}

int main() {
    const int n = 1000;
    int* data = nullptr;
    CHECK(cudaMallocManaged(&data, n * sizeof(int)));
    for (int i = 0; i < n; ++i) {
        data[i] = i;
    }
    touch<<<(n + 127) / 128, 128>>>(data, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    int last = data[n - 1];
    for (int i = 0; i < n; ++i) {
        data[i] = -i;
    }
    touch<<<(n + 127) / 128, 128>>>(data, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    long long again = 0;
    for (int i = 0; i < n; ++i) {
        again += data[i];
    }

    int* device_only = nullptr;
    CHECK(cudaMalloc(&device_only, sizeof(int)));
    cudaPointerAttributes managed_attr = {};
    cudaPointerAttributes device_attr = {};
    CHECK(cudaPointerGetAttributes(&managed_attr, data));
    CHECK(cudaPointerGetAttributes(&device_attr, device_only));

    std::printf("sum=%lld\n", sum);
    std::printf("last=%d\n", last);
    std::printf("again=%lld\n", again);
    std::printf("bumps=%d\n", bumps);
    std::printf("ismanaged=%d\n", managed_attr.type == cudaMemoryTypeManaged ? 1 : 0);
    std::printf("isdevice=%d\n", device_attr.type == cudaMemoryTypeDevice ? 1 : 0);
    cudaFree(device_only);
    cudaFree(data);
    return 0;
}
