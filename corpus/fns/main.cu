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

__device__ unsigned um(const unsigned* p) {
    volatile unsigned held = *p;
    return held;
}

__device__ int im(const int* p) {
    volatile int held = *p;
    return held;
}

// Nth set bit. Inputs come from host memory so nvcc cannot fold INT_MIN.
__global__ void find(const unsigned* m, const unsigned* b, const int* o, unsigned* out) {
    out[0] = __fns(um(m + 0), um(b + 0), im(o + 0));
    out[1] = __fns(um(m + 1), um(b + 0), im(o + 1));
    out[2] = __fns(um(m + 2), um(b + 0), im(o + 1));
    out[3] = __fns(um(m + 2), um(b + 0), im(o + 2));
    out[4] = __fns(um(m + 2), um(b + 1), im(o + 3));
    out[5] = __fns(um(m + 3), um(b + 2), im(o + 3));
    out[6] = __fns(um(m + 3), um(b + 0), im(o + 4));
    out[7] = __fns(um(m + 3), um(b + 0), im(o + 5));
    out[8] = __fns(um(m + 4), um(b + 0), im(o + 2));
    out[9] = __fns(um(m + 5), um(b + 3), im(o + 3));
    out[10] = __fns(um(m + 5), um(b + 0), im(o + 6));
    out[11] = __fns(um(m + 5), um(b + 4), im(o + 7));
    out[12] = __fns(um(m + 5), um(b + 4), im(o + 8));
    out[13] = __fns(um(m + 5), um(b + 4), im(o + 9));
    out[14] = __fns(um(m + 0), um(b + 3), im(o + 9));
}

int main() {
    unsigned masks[6] = {0u, 1u, 2u, 0xfu, 0x80000000u, 0xffffffffu};
    unsigned bases[5] = {0u, 4u, 2u, 32u, 31u};
    int offsets[10] = {1, 0, 1, -1, 4, 5, 32, -32, -33, (int)0x80000000};
    unsigned* m = nullptr;
    unsigned* b = nullptr;
    int* o = nullptr;
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&m, sizeof(masks)));
    CHECK(cudaMalloc(&b, sizeof(bases)));
    CHECK(cudaMalloc(&o, sizeof(offsets)));
    CHECK(cudaMalloc(&out, 15 * sizeof(unsigned)));
    CHECK(cudaMemcpy(m, masks, sizeof(masks), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b, bases, sizeof(bases), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(o, offsets, sizeof(offsets), cudaMemcpyHostToDevice));
    find<<<1, 1>>>(m, b, o, out);
    CHECK(cudaDeviceSynchronize());
    unsigned got[15] = {};
    CHECK(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost));
    const char* names[15] = {"none", "zbit", "clear", "up", "down", "self", "nth", "past",
                             "msb", "bhi", "top", "bot", "over", "imin", "imin0"};
    for (int i = 0; i < 15; ++i) std::printf("%s=%u\n", names[i], got[i]);
    return 0;
}
