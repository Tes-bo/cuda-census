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

template <typename T, int N>
__device__ T accumulate(const T* p) {
    T sum = T(0);
    for (int k = 0; k < N; ++k) {
        sum += p[k];
    }
    return sum;
}

// One template kernel, instantiated for int and double. Each thread takes
// its own block from the device heap, fills it, sums it, and frees it.
template <typename T, int N>
__global__ void heap(T* out, unsigned* nulls) {
    int t = threadIdx.x;
    T* p = static_cast<T*>(malloc(N * sizeof(T)));
    if (p == nullptr) {
        atomicAdd(nulls, 1u);
        out[t] = T(0);
        return;
    }
    for (int k = 0; k < N; ++k) {
        p[k] = static_cast<T>(t + k);
    }
    out[t] = accumulate<T, N>(p);
    free(p);
}

// new[] and delete[] on the device, and a request no heap can satisfy,
// which must return a null pointer rather than trap.
__global__ void objects(int* out, unsigned* nulls) {
    int t = threadIdx.x;
    int* q = new int[16];
    if (q == nullptr) {
        atomicAdd(&nulls[0], 1u);
        out[t] = 0;
    } else {
        for (int k = 0; k < 16; ++k) {
            q[k] = t * k;
        }
        out[t] = accumulate<int, 16>(q);
        delete[] q;
    }
    void* huge = malloc(static_cast<size_t>(1) << 40);
    if (huge == nullptr) {
        atomicAdd(&nulls[1], 1u);
    } else {
        free(huge);
    }
}

int main() {
    const size_t heap_bytes = static_cast<size_t>(64) << 20;
    CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_bytes));
    size_t got_heap = 0;
    CHECK(cudaDeviceGetLimit(&got_heap, cudaLimitMallocHeapSize));
    int* ints = nullptr;
    double* doubles = nullptr;
    int* objs = nullptr;
    unsigned* nulls = nullptr;
    CHECK(cudaMalloc(&ints, 128 * sizeof(int)));
    CHECK(cudaMalloc(&doubles, 64 * sizeof(double)));
    CHECK(cudaMalloc(&objs, 32 * sizeof(int)));
    CHECK(cudaMalloc(&nulls, 3 * sizeof(unsigned)));
    CHECK(cudaMemset(nulls, 0, 3 * sizeof(unsigned)));
    heap<int, 32><<<1, 128>>>(ints, &nulls[0]);
    CHECK(cudaGetLastError());
    heap<double, 8><<<1, 64>>>(doubles, &nulls[0]);
    CHECK(cudaGetLastError());
    objects<<<1, 32>>>(objs, &nulls[1]);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int hi[128] = {};
    double hd[64] = {};
    int ho[32] = {};
    unsigned hn[3] = {};
    CHECK(cudaMemcpy(hi, ints, sizeof(hi), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hd, doubles, sizeof(hd), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ho, objs, sizeof(ho), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hn, nulls, sizeof(hn), cudaMemcpyDeviceToHost));
    long long isum = 0;
    double dsum = 0.0;
    long long osum = 0;
    for (int t = 0; t < 128; ++t) isum += hi[t];
    for (int t = 0; t < 64; ++t) dsum += hd[t];
    for (int t = 0; t < 32; ++t) osum += ho[t];
    std::printf("heapok=%d\n", got_heap >= heap_bytes ? 1 : 0);
    std::printf("isum=%lld\n", isum);
    std::printf("dsum=%lld\n", static_cast<long long>(dsum));
    std::printf("osum=%lld\n", osum);
    std::printf("i127=%d\n", hi[127]);
    std::printf("mallocnull=%u\n", hn[0]);
    std::printf("newnull=%u\n", hn[1]);
    std::printf("hugenull=%u\n", hn[2]);
    cudaFree(ints);
    cudaFree(doubles);
    cudaFree(objs);
    cudaFree(nulls);
    return 0;
}
