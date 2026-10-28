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

__global__ void bits_of_one(unsigned long long* out) {
    *out = __double_as_longlong(1.0);
}

int main() {
    unsigned long long* out = nullptr;
    CHECK(cudaMalloc(&out, sizeof(unsigned long long)));
    bits_of_one<<<1, 1>>>(out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned long long bits = 0;
    CHECK(cudaMemcpy(&bits, out, sizeof(bits), cudaMemcpyDeviceToHost));
    std::printf("bits=%llu\n", bits);
    cudaFree(out);
    return 0;
}
