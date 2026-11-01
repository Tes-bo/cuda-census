// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#define CHECK(cmd)                                                         \
    do {                                                                   \
        cudaError_t err_ = (cmd);                                          \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s\n", cudaGetErrorString(err_));        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

constexpr int kN = 1000;

struct square {
    __host__ __device__ int operator()(int x) const { return x * x; }
};

__global__ void block_sums(const int* in, int* out) {
    using BlockReduce = cub::BlockReduce<int, 128>;
    __shared__ typename BlockReduce::TempStorage temp;
    int v = in[blockIdx.x * 128 + threadIdx.x];
    int s = BlockReduce(temp).Sum(v);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = s;
    }
}

// Four 32-lane reductions in one block. A 64-lane wavefront must still give
// one maximum per 32 lanes.
__global__ void warp_maxima(const int* in, int* out) {
    using WarpReduce = cub::WarpReduce<int>;
    __shared__ typename WarpReduce::TempStorage temp[4];
    int warp = threadIdx.x / 32;
    int m = WarpReduce(temp[warp]).Reduce(in[threadIdx.x], cub::Max());
    if (threadIdx.x % 32 == 0) {
        out[warp] = m;
    }
}

// Thrust and CUB from the CUDA toolkit headers, with no extra library to link.
int main() {
    thrust::host_vector<int> h(kN);
    for (int i = 0; i < kN; ++i) {
        h[i] = (i * 7919) % kN - 500;  // a permutation of -500..499
    }
    thrust::device_vector<int> d = h;
    int unsorted_sum = thrust::reduce(d.begin(), d.end());
    thrust::sort(d.begin(), d.end());
    int first = d[0];
    int middle = d[kN / 2];
    int last = d[kN - 1];
    int squares = thrust::transform_reduce(d.begin(), d.end(), square(), 0, thrust::plus<int>());
    thrust::inclusive_scan(d.begin(), d.end(), d.begin());
    int scan_half = d[kN / 2 - 1];
    int scan_end = d[kN - 1];

    thrust::device_vector<int> keys = h;
    thrust::device_vector<int> values(kN);
    for (int i = 0; i < kN; ++i) {
        values[i] = i;
    }
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin(), thrust::greater<int>());
    int top_index = values[0];

    // CUB device-wide sum, with the usual size query then run.
    int* sum_out = nullptr;
    CHECK(cudaMalloc(&sum_out, sizeof(int)));
    thrust::device_vector<int> original = h;
    const int* raw_original = thrust::raw_pointer_cast(original.data());
    size_t temp_bytes = 0;
    CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, raw_original, sum_out, kN));
    void* temp = nullptr;
    CHECK(cudaMalloc(&temp, temp_bytes));
    CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, raw_original, sum_out, kN));
    int cub_sum = 0;
    CHECK(cudaMemcpy(&cub_sum, sum_out, sizeof(int), cudaMemcpyDeviceToHost));

    int hin[1024];
    for (int i = 0; i < 1024; ++i) {
        hin[i] = i % 13;
    }
    int* in = nullptr;
    int* blocks = nullptr;
    int* maxima = nullptr;
    CHECK(cudaMalloc(&in, sizeof(hin)));
    CHECK(cudaMalloc(&blocks, 8 * sizeof(int)));
    CHECK(cudaMalloc(&maxima, 4 * sizeof(int)));
    CHECK(cudaMemcpy(in, hin, sizeof(hin), cudaMemcpyHostToDevice));
    block_sums<<<8, 128>>>(in, blocks);
    CHECK(cudaGetLastError());
    for (int i = 0; i < 128; ++i) {
        hin[i] = (i * 37) % 101;
    }
    int* win = nullptr;
    CHECK(cudaMalloc(&win, 128 * sizeof(int)));
    CHECK(cudaMemcpy(win, hin, 128 * sizeof(int), cudaMemcpyHostToDevice));
    warp_maxima<<<1, 128>>>(win, maxima);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int hb[8] = {};
    int hm[4] = {};
    CHECK(cudaMemcpy(hb, blocks, sizeof(hb), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hm, maxima, sizeof(hm), cudaMemcpyDeviceToHost));
    int block_total = 0;
    for (int b = 0; b < 8; ++b) block_total += hb[b];

    std::printf("sum=%d\n", unsorted_sum);
    std::printf("first=%d\n", first);
    std::printf("middle=%d\n", middle);
    std::printf("last=%d\n", last);
    std::printf("squares=%d\n", squares);
    std::printf("scanhalf=%d\n", scan_half);
    std::printf("scanend=%d\n", scan_end);
    std::printf("topindex=%d\n", top_index);
    std::printf("cubsum=%d\n", cub_sum);
    std::printf("block3=%d\n", hb[3]);
    std::printf("blocktotal=%d\n", block_total);
    std::printf("warp0=%d\n", hm[0]);
    std::printf("warp1=%d\n", hm[1]);
    std::printf("warp2=%d\n", hm[2]);
    std::printf("warp3=%d\n", hm[3]);
    cudaFree(sum_out);
    cudaFree(temp);
    cudaFree(in);
    cudaFree(blocks);
    cudaFree(maxima);
    cudaFree(win);
    return 0;
}
