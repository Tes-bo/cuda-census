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

constexpr unsigned kFull = 0xffffffffu;
constexpr int kOps = 6;

// 40 threads: one full warp and a second warp of 8 lanes. A width below 32
// splits the warp into segments. A source outside the caller's segment
// returns the caller's own value, except that a butterfly may read an
// earlier segment.
__global__ void segments(unsigned* lanes, unsigned long long* wide, unsigned* misc) {
    int t = threadIdx.x;
    unsigned active = __activemask();
    unsigned v = static_cast<unsigned>(t * t + 1);
    if (t < 32) {
        lanes[0 * 32 + t] = __shfl_sync(kFull, v, 3, 8);
        lanes[1 * 32 + t] = __shfl_sync(kFull, v, 13, 8);
        lanes[2 * 32 + t] = __shfl_up_sync(kFull, v, 5, 16);
        lanes[3 * 32 + t] = __shfl_down_sync(kFull, v, 3, 8);
        lanes[4 * 32 + t] = __shfl_xor_sync(kFull, v, 8, 8);
        lanes[5 * 32 + t] = __shfl_up_sync(kFull, v, 1);
        unsigned long long w = (static_cast<unsigned long long>(t) << 32) | v;
        wide[t] = __shfl_down_sync(kFull, w, 16);
    } else {
        misc[2 + (t - 32)] = __shfl_sync(0xffu, v, 2);
    }
    if (t == 0) {
        misc[0] = active;
    }
    if (t == 32) {
        misc[1] = active;
    }
}

int main() {
    unsigned* lanes = nullptr;
    unsigned long long* wide = nullptr;
    unsigned* misc = nullptr;
    CHECK(cudaMalloc(&lanes, kOps * 32 * sizeof(unsigned)));
    CHECK(cudaMalloc(&wide, 32 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&misc, 10 * sizeof(unsigned)));
    segments<<<1, 40>>>(lanes, wide, misc);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned hl[kOps * 32] = {};
    unsigned long long hw[32] = {};
    unsigned hm[10] = {};
    CHECK(cudaMemcpy(hl, lanes, sizeof(hl), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hw, wide, sizeof(hw), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hm, misc, sizeof(hm), cudaMemcpyDeviceToHost));
    // Each lane's result, weighted by lane + 1, so a swap of two lanes changes the sum.
    const char* names[kOps] = {"idx8", "idx8wrap", "up16", "down8", "xor8", "up1"};
    for (int op = 0; op < kOps; ++op) {
        unsigned long long sum = 0;
        for (int lane = 0; lane < 32; ++lane) {
            sum += static_cast<unsigned long long>(hl[op * 32 + lane]) * (lane + 1);
        }
        std::printf("%s=%llu\n", names[op], sum);
    }
    unsigned long long wsum = 0;
    for (int lane = 0; lane < 32; ++lane) {
        wsum += hw[lane] * (lane + 1);
    }
    std::printf("down64=%llu\n", wsum);
    std::printf("xor8l3=%u\n", hl[4 * 32 + 3]);
    std::printf("xor8l11=%u\n", hl[4 * 32 + 11]);
    std::printf("xor8l19=%u\n", hl[4 * 32 + 19]);
    std::printf("xor8l27=%u\n", hl[4 * 32 + 27]);
    std::printf("active0=%u\n", hm[0]);
    std::printf("active32=%u\n", hm[1]);
    unsigned part = 0;
    for (int i = 2; i < 10; ++i) {
        part += hm[i];
    }
    std::printf("part=%u\n", part);
    cudaFree(lanes);
    cudaFree(wide);
    cudaFree(misc);
    return 0;
}
