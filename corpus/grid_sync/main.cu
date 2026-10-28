// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define CHECK(cmd)                                                         \
    do {                                                                   \
        cudaError_t err_ = (cmd);                                          \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s\n", cudaGetErrorString(err_));        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

constexpr int kBlocks = 8;
constexpr int kThreads = 128;
constexpr int kN = kBlocks * kThreads;

// The last block spins before it writes. Without a grid-wide barrier, the
// block before it would read zeros from the last block's slots.
__global__ void two_phase(int* buf, int* out, long long spin) {
    cg::grid_group grid = cg::this_grid();
    int i = static_cast<int>(grid.thread_rank());
    if (blockIdx.x == gridDim.x - 1) {
        long long start = clock64();
        while (clock64() - start < spin) {
        }
    }
    buf[i] = i + 1;
    grid.sync();
    out[i] = buf[(i + kThreads) % kN];
    if (i == 0) {
        out[kN] = static_cast<int>(grid.size());
        out[kN + 1] = grid.is_valid() ? 1 : 0;
    }
}

int main() {
    int supported = 0;
    CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, 0));
    int* buf = nullptr;
    int* out = nullptr;
    CHECK(cudaMalloc(&buf, kN * sizeof(int)));
    CHECK(cudaMalloc(&out, (kN + 2) * sizeof(int)));
    CHECK(cudaMemset(buf, 0, kN * sizeof(int)));
    CHECK(cudaMemset(out, 0, (kN + 2) * sizeof(int)));
    long long spin = 20000000LL;
    void* args[] = {&buf, &out, &spin};
    CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(two_phase), dim3(kBlocks), dim3(kThreads), args, 0, nullptr));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int host[kN + 2] = {};
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    long long sum = 0;
    int moved = 0;
    for (int i = 0; i < kN; ++i) {
        sum += host[i];
        moved += host[i] == (i + kThreads) % kN + 1 ? 1 : 0;
    }
    std::printf("cooperative=%d\n", supported);
    std::printf("sum=%lld\n", sum);
    std::printf("moved=%d\n", moved);
    std::printf("size=%d\n", host[kN]);
    std::printf("valid=%d\n", host[kN + 1]);
    cudaFree(buf);
    cudaFree(out);
    return 0;
}
