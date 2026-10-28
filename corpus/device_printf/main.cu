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

// One device thread prints, in order, before the host prints. The device
// lines reach stdout at the synchronize, so they come first.
__global__ void say(int base, long long wide, unsigned top) {
    if (blockIdx.x == 1 && threadIdx.x == 7) {
        printf("dev=%d\n", base + 1);
        printf("devneg=%d\n", -base);
        printf("devll=%lld\n", wide);
        printf("devu=%u\n", top);
        printf("devtid=%d\n", static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x));
    }
}

int main() {
    say<<<2, 32>>>(41, -9223372036854775807LL - 1, 4294967295u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::printf("host=1\n");
    return 0;
}
