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

constexpr int kW = 37;  // ints per row, so a row is not a multiple of any alignment
constexpr int kH = 9;
constexpr int kHostPitch = kW + 3;
constexpr int kBackPitch = kW + 1;

__global__ void bump(int* base, size_t pitch) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    if (x < kW) {
        int* row = reinterpret_cast<int*>(reinterpret_cast<char*>(base) + y * pitch);
        row[x] += y * 100 + x;
    }
}

// The device pitch is chosen by the runtime and differs between devices, so
// it is never printed. Every copy names its own pitch, and padding outside
// the copied width must not change.
int main() {
    static int host[kH][kHostPitch];
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kHostPitch; ++x) {
            host[y][x] = x < kW ? y * kW + x : -1;
        }
    }
    int* d = nullptr;
    size_t pitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d), &pitch, kW * sizeof(int), kH));
    CHECK(cudaMemcpy2D(d, pitch, host, kHostPitch * sizeof(int), kW * sizeof(int), kH, cudaMemcpyHostToDevice));
    // Rows 2..4, ints 5..9, every byte 0x01.
    char* corner = reinterpret_cast<char*>(d) + 2 * pitch + 5 * sizeof(int);
    CHECK(cudaMemset2D(corner, pitch, 0x01, 5 * sizeof(int), 3));
    bump<<<kH, 64>>>(d, pitch);
    CHECK(cudaGetLastError());

    // A 4x3 block from (x=10, y=1) into (x=2, y=1) of a second pitched buffer.
    int* small = nullptr;
    size_t small_pitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&small), &small_pitch, 8 * sizeof(int), 4));
    CHECK(cudaMemset2D(small, small_pitch, 0, 8 * sizeof(int), 4));
    CHECK(cudaMemcpy2D(reinterpret_cast<char*>(small) + small_pitch + 2 * sizeof(int), small_pitch,
                       reinterpret_cast<char*>(d) + pitch + 10 * sizeof(int), pitch, 4 * sizeof(int), 3,
                       cudaMemcpyDeviceToDevice));
    CHECK(cudaDeviceSynchronize());

    static int back[kH][kBackPitch];
    for (int y = 0; y < kH; ++y) {
        back[y][kW] = -7;
    }
    CHECK(cudaMemcpy2D(back, kBackPitch * sizeof(int), d, pitch, kW * sizeof(int), kH, cudaMemcpyDeviceToHost));
    int hs[4][8] = {};
    CHECK(cudaMemcpy2D(hs, 8 * sizeof(int), small, small_pitch, 8 * sizeof(int), 4, cudaMemcpyDeviceToHost));

    long long weighted = 0;
    int padding = 1;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            weighted += static_cast<long long>(back[y][x]) * (y * kW + x + 1);
        }
        padding &= back[y][kW] == -7 ? 1 : 0;
    }
    long long small_weighted = 0;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 8; ++x) {
            small_weighted += static_cast<long long>(hs[y][x]) * (y * 8 + x + 1);
        }
    }
    std::printf("pitchok=%d\n", pitch >= kW * sizeof(int) ? 1 : 0);
    std::printf("plain=%d\n", back[0][3]);
    std::printf("set=%d\n", back[3][7]);
    std::printf("edge=%d\n", back[4][10]);
    std::printf("last=%d\n", back[kH - 1][kW - 1]);
    std::printf("padding=%d\n", padding);
    std::printf("weighted=%lld\n", weighted);
    std::printf("block=%lld\n", small_weighted);
    std::printf("blockcorner=%d\n", hs[1][2]);
    cudaFree(d);
    cudaFree(small);
    return 0;
}
