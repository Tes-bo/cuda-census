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

__device__ float fv(const unsigned* p) {
    volatile unsigned held = *p;
    union { unsigned as_bits; float as_float; } v;
    v.as_bits = held;
    return v.as_float;
}

__device__ unsigned fb(float x) {
    union { float as_float; unsigned as_bits; } v;
    v.as_float = x;
    return v.as_bits;
}

// Scaled hypotenuse. Inputs come from host memory so nvcc cannot fold an
// infinity, a NaN, or a square that overflows.
__global__ void lengths(const unsigned* f, unsigned* out) {
    float inf = fv(f + 0);
    float nan = fv(f + 1);
    float m0 = fv(f + 2);
    float big = fv(f + 3);
    float m1 = fv(f + 4);
    float zero = fv(f + 5);
    float one = fv(f + 6);
    out[0] = fb(hypotf(inf, nan));
    out[1] = fb(hypotf(nan, nan));
    out[2] = fb(hypotf(m0, m0));
    out[3] = fb(hypotf(big, big));
    out[4] = fb(sqrtf(big * big + big * big));
    out[5] = fb(hypotf(m1, zero));
    out[6] = fb(hypotf(one, one));
    out[7] = fb(norm3df(inf, nan, nan));
    out[8] = fb(norm4df(big, big, big, big));
    out[9] = fb(norm3df(one, one, one));
}

int main() {
    unsigned fbits[7] = {0x7f800000u, 0x7fc00001u, 0x80000000u, 0x5f800000u,
                         0xbf800000u, 0x00000000u, 0x3f800000u};
    unsigned* f = nullptr;
    unsigned* out = nullptr;
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&out, 10 * sizeof(unsigned)));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    lengths<<<1, 1>>>(f, out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned got[10] = {};
    CHECK(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost));
    const char* names[10] = {"infnan", "nann", "zz", "big", "naive", "m1", "one", "n3", "n4", "n3one"};
    for (int i = 0; i < 10; ++i) std::printf("%s=%u\n", names[i], got[i]);
    return 0;
}
