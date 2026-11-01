// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
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

__global__ void expect_seven(const int* in) {
    assert(*in == 7);
}

// A device assert that holds, then one that fails. The failure is a sticky
// error: the synchronize reports it, cudaGetLastError returns it once and
// clears its own record, and every later runtime call still fails with it.
// The program exits 0 so the flags reach stdout.
int main() {
    int* in = nullptr;
    int seven = 7;
    int eight = 8;
    CHECK(cudaMalloc(&in, sizeof(int)));
    CHECK(cudaMemcpy(in, &seven, sizeof(int), cudaMemcpyHostToDevice));
    expect_seven<<<1, 1>>>(in);
    CHECK(cudaGetLastError());
    cudaError_t held = cudaDeviceSynchronize();

    CHECK(cudaMemcpy(in, &eight, sizeof(int), cudaMemcpyHostToDevice));
    expect_seven<<<1, 1>>>(in);
    cudaError_t launched = cudaGetLastError();
    cudaError_t failed = cudaDeviceSynchronize();
    cudaError_t last = cudaGetLastError();
    cudaError_t cleared = cudaGetLastError();
    int* more = nullptr;
    cudaError_t alloc = cudaMalloc(&more, sizeof(int));
    cudaError_t resync = cudaDeviceSynchronize();

    std::printf("held=%d\n", held == cudaSuccess ? 1 : 0);
    std::printf("launched=%d\n", launched == cudaSuccess ? 1 : 0);
    std::printf("assert=%d\n", failed == cudaErrorAssert ? 1 : 0);
    std::printf("last=%d\n", last == cudaErrorAssert ? 1 : 0);
    std::printf("cleared=%d\n", cleared == cudaSuccess ? 1 : 0);
    std::printf("alloc=%d\n", alloc == cudaErrorAssert ? 1 : 0);
    std::printf("resync=%d\n", resync == cudaErrorAssert ? 1 : 0);
    return 0;
}
