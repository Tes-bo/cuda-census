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

constexpr int kN = 256;

// The producer spins first, so a consumer that ignored the event would read
// the zeroed buffer.
__global__ void produce(int* buf, long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) {
    }
    buf[threadIdx.x] = threadIdx.x + 1;
}

__global__ void consume(const int* buf, int* out) {
    int sum = 0;
    for (int i = 0; i < kN; ++i) {
        sum += buf[i];
    }
    *out = sum;
}

void CUDART_CB mark(void* flag) {
    *static_cast<int*>(flag) = 42;
}

int main() {
    int* buf = nullptr;
    int* out = nullptr;
    int* pinned = nullptr;
    CHECK(cudaMalloc(&buf, kN * sizeof(int)));
    CHECK(cudaMalloc(&out, sizeof(int)));
    CHECK(cudaMallocHost(&pinned, sizeof(int)));
    *pinned = -1;

    cudaStream_t a;
    cudaStream_t b;
    cudaEvent_t ready;
    CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));

    // Launch both kernels once first. With lazy module loading, the first
    // launch of a kernel can synchronize the device and hide a missing wait.
    produce<<<1, kN, 0, a>>>(buf, 0);
    consume<<<1, 1, 0, b>>>(buf, out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemset(buf, 0, kN * sizeof(int)));
    CHECK(cudaMemset(out, 0, sizeof(int)));
    CHECK(cudaDeviceSynchronize());

    int flag = 0;
    produce<<<1, kN, 0, a>>>(buf, 20000000LL);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(ready, a));
    CHECK(cudaStreamWaitEvent(b, ready, 0));
    consume<<<1, 1, 0, b>>>(buf, out);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(pinned, out, sizeof(int), cudaMemcpyDeviceToHost, b));
    CHECK(cudaLaunchHostFunc(b, mark, &flag));
    CHECK(cudaStreamSynchronize(b));

    std::printf("sum=%d\n", *pinned);
    std::printf("hostfn=%d\n", flag);
    std::printf("eventdone=%d\n", cudaEventQuery(ready) == cudaSuccess ? 1 : 0);
    std::printf("streamdone=%d\n", cudaStreamQuery(b) == cudaSuccess ? 1 : 0);

    CHECK(cudaEventDestroy(ready));
    CHECK(cudaStreamDestroy(a));
    CHECK(cudaStreamDestroy(b));
    CHECK(cudaFreeHost(pinned));
    cudaFree(buf);
    cudaFree(out);
    return 0;
}
