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

__constant__ int table[8];
__device__ unsigned counter;
__device__ int scale = 3;

// Constant and global symbols reached only through the runtime symbol API:
// copies at an offset, a symbol address, and a symbol size.
__global__ void use(int* out) {
    int t = threadIdx.x;
    out[t] = table[t] * scale;
    atomicAdd(&counter, 1u);
}

int main() {
    int host[8] = {1, -2, 4, -8, 16, -32, 64, -128};
    int patch = 1000;
    unsigned start = 10;
    int* out = nullptr;
    CHECK(cudaMalloc(&out, 8 * sizeof(int)));
    CHECK(cudaMemcpyToSymbol(table, host, sizeof(host)));
    CHECK(cudaMemcpyToSymbol(table, &patch, sizeof(int), 4 * sizeof(int)));
    CHECK(cudaMemcpyToSymbol(counter, &start, sizeof(start)));
    use<<<1, 8>>>(out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int got[8] = {};
    CHECK(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost));
    unsigned count = 0;
    CHECK(cudaMemcpyFromSymbol(&count, counter, sizeof(count)));
    int fifth = 0;
    CHECK(cudaMemcpyFromSymbol(&fifth, table, sizeof(int), 5 * sizeof(int)));
    void* scale_addr = nullptr;
    CHECK(cudaGetSymbolAddress(&scale_addr, scale));
    int scale_read = 0;
    CHECK(cudaMemcpy(&scale_read, scale_addr, sizeof(int), cudaMemcpyDeviceToHost));
    size_t table_size = 0;
    CHECK(cudaGetSymbolSize(&table_size, table));
    for (int i = 0; i < 8; ++i) std::printf("t%d=%d\n", i, got[i]);
    std::printf("counter=%u\n", count);
    std::printf("fifth=%d\n", fifth);
    std::printf("scale=%d\n", scale_read);
    std::printf("size=%zu\n", table_size);
    cudaFree(out);
    return 0;
}
