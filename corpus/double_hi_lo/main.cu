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

// High and low 32-bit words of IEEE 1.0. Reconstructing them must be 1.0.
__global__ void split_one(int* out) {
    int hi = __double2hiint(1.0);
    int lo = __double2loint(1.0);
    double again = __hiloint2double(hi, lo);
    out[0] = hi;
    out[1] = lo;
    out[2] = again == 1.0;
}

int main() {
    int* out = nullptr;
    CHECK(cudaMalloc(&out, 3 * sizeof(int)));
    CHECK(cudaMemset(out, 0, 3 * sizeof(int)));
    split_one<<<1, 1>>>(out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int host[3] = {};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("hi=%d\n", host[0]);
    std::printf("lo=%d\n", host[1]);
    std::printf("round=%d\n", host[2]);
    cudaFree(out);
    return 0;
}
