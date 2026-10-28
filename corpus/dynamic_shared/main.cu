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

constexpr int kThreads = 256;
constexpr int kInts = 16384;  // 64 KiB, above the 48 KiB default, so the opt-in is needed

// One extern buffer sized at launch. Every thread fills 64 words, then each
// thread reads 64 words that other threads wrote.
__global__ void fill_and_sum(unsigned long long* out) {
    extern __shared__ int words[];
    int t = threadIdx.x;
    for (int k = 0; k < 64; ++k) {
        words[t * 64 + k] = t + k;
    }
    __syncthreads();
    unsigned long long partial = 0;
    for (int k = 0; k < 64; ++k) {
        partial += static_cast<unsigned long long>(words[(t + k * 257) % kInts]) * (k + 1);
    }
    atomicAdd(&out[0], partial);
    if (t == 0) {
        unsigned long long total = 0;
        for (int i = 0; i < kInts; ++i) {
            total += static_cast<unsigned>(words[i]);
        }
        out[1] = total;
    }
}

// The same extern buffer carved into an int region and an aligned double region.
__global__ void carve(double* out) {
    extern __shared__ __align__(16) unsigned char raw[];
    int* counts = reinterpret_cast<int*>(raw);
    double* halves = reinterpret_cast<double*>(raw + 64 * sizeof(int));
    int t = threadIdx.x;
    counts[t] = t;
    halves[t] = t * 0.5;
    __syncthreads();
    if (t == 0) {
        double sum = 0.0;
        for (int i = 0; i < 64; ++i) {
            sum += halves[i] * counts[63 - i];
        }
        out[0] = sum;
    }
}

int main() {
    const size_t bytes = kInts * sizeof(int);
    CHECK(cudaFuncSetAttribute(fill_and_sum, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    cudaFuncAttributes attr = {};
    CHECK(cudaFuncGetAttributes(&attr, fill_and_sum));
    unsigned long long* out = nullptr;
    double* dout = nullptr;
    CHECK(cudaMalloc(&out, 2 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&dout, sizeof(double)));
    CHECK(cudaMemset(out, 0, 2 * sizeof(unsigned long long)));
    fill_and_sum<<<1, kThreads, bytes>>>(out);
    CHECK(cudaGetLastError());
    carve<<<1, 64, 64 * sizeof(int) + 64 * sizeof(double)>>>(dout);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned long long host[2] = {};
    double carved = 0.0;
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&carved, dout, sizeof(carved), cudaMemcpyDeviceToHost));
    std::printf("optin=%d\n", attr.maxDynamicSharedSizeBytes);
    std::printf("total=%llu\n", host[1]);
    std::printf("weighted=%llu\n", host[0]);
    std::printf("carved=%lld\n", static_cast<long long>(carved * 2.0));
    cudaFree(out);
    cudaFree(dout);
    return 0;
}
