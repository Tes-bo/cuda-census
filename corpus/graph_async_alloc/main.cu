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

__global__ void fill_ones(int* p, int n) {
    int i = threadIdx.x;
    if (i < n) {
        p[i] = 1;
    }
}

__global__ void total(const int* p, int n, int* out) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += p[i];
    }
    *out = sum;
}

int main() {
    const int n = 32;
    int* out = nullptr;
    CHECK(cudaMalloc(&out, sizeof(int)));
    CHECK(cudaMemset(out, 0, sizeof(int)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    int* scratch = nullptr;
    CHECK(cudaMallocAsync(reinterpret_cast<void**>(&scratch), n * sizeof(int), stream));
    fill_ones<<<1, n, 0, stream>>>(scratch, n);
    total<<<1, 1, 0, stream>>>(scratch, n, out);
    CHECK(cudaFreeAsync(scratch, stream));
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));

    int sum = 0;
    CHECK(cudaMemcpy(&sum, out, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("sum=%d\n", sum);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(out);
    return 0;
}
