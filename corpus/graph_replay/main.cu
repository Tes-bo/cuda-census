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

__global__ void fill_ones(int* out, int n) {
    int i = threadIdx.x;
    if (i < n) {
        out[i] = 1;
    }
}

int main() {
    const int n = 32;
    int* out = nullptr;
    CHECK(cudaMalloc(&out, n * sizeof(int)));
    CHECK(cudaMemset(out, 0, n * sizeof(int)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    fill_ones<<<1, n, 0, stream>>>(out, n);
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGetLastError());
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));

    int host[32];
    CHECK(cudaMemcpy(host, out, n * sizeof(int), cudaMemcpyDeviceToHost));
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += host[i];
    }
    std::printf("sum=%d\n", sum);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(out);
    return 0;
}
