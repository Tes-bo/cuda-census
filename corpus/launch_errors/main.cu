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

__global__ void fill(int* out, int value) {
    out[threadIdx.x] = value;
}

// Launches the runtime must refuse, and how the refusal is reported. Each
// printed value is 1 when the runtime behaves as the CUDA documentation says.
int main() {
    int* out = nullptr;
    CHECK(cudaMalloc(&out, 64 * sizeof(int)));
    CHECK(cudaMemset(out, 0, 64 * sizeof(int)));
    int max_threads = 0;
    CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, 0));

    // Peek leaves a launch error in place; get returns it once and clears it.
    fill<<<1, max_threads * 2>>>(out, 1);
    cudaError_t peek1 = cudaPeekAtLastError();
    cudaError_t peek2 = cudaPeekAtLastError();
    cudaError_t got = cudaGetLastError();
    cudaError_t cleared = cudaGetLastError();

    fill<<<0, 32>>>(out, 2);
    cudaError_t zero_blocks = cudaGetLastError();
    fill<<<dim3(1, 70000), 1>>>(out, 3);
    cudaError_t grid_y = cudaGetLastError();
    fill<<<1, 32, 1u << 30>>>(out, 4);
    cudaError_t big_shared = cudaGetLastError();

    // None of the refused launches ran, and the device is still usable.
    cudaError_t sync = cudaDeviceSynchronize();
    int before[64] = {};
    CHECK(cudaMemcpy(before, out, sizeof(before), cudaMemcpyDeviceToHost));
    int untouched = 1;
    for (int i = 0; i < 64; ++i) {
        untouched &= before[i] == 0 ? 1 : 0;
    }
    fill<<<1, 64>>>(out, 5);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int after[64] = {};
    CHECK(cudaMemcpy(after, out, sizeof(after), cudaMemcpyDeviceToHost));
    int sum = 0;
    for (int i = 0; i < 64; ++i) {
        sum += after[i];
    }

    std::printf("toomany=%d\n", peek1 == cudaErrorInvalidConfiguration ? 1 : 0);
    std::printf("peekkeeps=%d\n", peek2 == peek1 ? 1 : 0);
    std::printf("gettakes=%d\n", got == peek1 ? 1 : 0);
    std::printf("cleared=%d\n", cleared == cudaSuccess ? 1 : 0);
    std::printf("zeroblocks=%d\n", zero_blocks == cudaErrorInvalidConfiguration ? 1 : 0);
    std::printf("gridy=%d\n", grid_y == cudaErrorInvalidConfiguration ? 1 : 0);
    std::printf("bigshared=%d\n", big_shared != cudaSuccess ? 1 : 0);
    std::printf("syncok=%d\n", sync == cudaSuccess ? 1 : 0);
    std::printf("untouched=%d\n", untouched);
    std::printf("sum=%d\n", sum);
    cudaFree(out);
    return 0;
}
