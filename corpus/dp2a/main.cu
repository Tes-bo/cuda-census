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

__device__ int iv(const int* p) {
    volatile int held = *p;
    return held;
}

__device__ unsigned uv(const unsigned* p) {
    volatile unsigned held = *p;
    return held;
}

// Two-byte dot product. Inputs come from host memory. lo reads the low
// two bytes of the second argument. hi reads the high two bytes.
__global__ void dot2(const int* s, const unsigned* u, int* out) {
    int a = iv(s + 0);
    int b = iv(s + 1);
    int bh = iv(s + 2);
    int one = iv(s + 3);
    int ff = iv(s + 4);
    int pair = iv(s + 5);
    int ones = iv(s + 6);
    int top = iv(s + 7);
    out[0] = __dp2a_lo(a, b, 0);
    out[1] = (int)__dp2a_lo(uv(u + 0), uv(u + 1), 0u);
    out[2] = __dp2a_hi(a, bh, 0);
    out[3] = (int)__dp2a_hi(uv(u + 0), uv(u + 2), 0u);
    out[4] = __dp2a_lo(a, bh, 0);
    out[5] = __dp2a_lo(one, ff, 0);
    out[6] = (int)__dp2a_lo(uv(u + 3), uv(u + 4), 0u);
    out[7] = __dp2a_hi(pair, ff, 0);
    out[8] = __dp2a_lo(pair, ones, top);
}

int main() {
    int sbits[8] = {0x00018000, 0x00008080, (int)0x80800000u, 1, (int)0xff, 0x00010001, 0x00000101, 0x7fffffff};
    unsigned ubits[5] = {0x00018000u, 0x00008080u, 0x80800000u, 1u, 0xffu};
    int* s = nullptr;
    unsigned* u = nullptr;
    int* out = nullptr;
    CHECK(cudaMalloc(&s, sizeof(sbits)));
    CHECK(cudaMalloc(&u, sizeof(ubits)));
    CHECK(cudaMalloc(&out, 9 * sizeof(int)));
    CHECK(cudaMemcpy(s, sbits, sizeof(sbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(u, ubits, sizeof(ubits), cudaMemcpyHostToDevice));
    dot2<<<1, 1>>>(s, u, out);
    CHECK(cudaDeviceSynchronize());
    int got[9] = {};
    CHECK(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost));
    const char* names[9] = {"slo", "ulo", "shi", "uhi", "zlo", "sff", "uff", "hiff", "wrap"};
    for (int i = 0; i < 9; ++i) std::printf("%s=%d\n", names[i], got[i]);
    return 0;
}
