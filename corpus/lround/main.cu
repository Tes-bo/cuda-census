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

__device__ double dv(const unsigned long long* p) {
    volatile unsigned long long held = *p;
    union { unsigned long long as_bits; double as_double; } v;
    v.as_bits = held;
    return v.as_double;
}

// Integer-returning rounding. Inputs come from host memory so nvcc cannot
// fold a halfway value, a NaN, or an infinity.
__global__ void round_to_int(const unsigned* f, const unsigned long long* d, long long* out) {
    out[0] = lrintf(fv(f + 0));
    out[1] = lroundf(fv(f + 0));
    out[2] = lroundf(fv(f + 1));
    out[3] = lrintf(fv(f + 2));
    out[4] = lroundf(fv(f + 2));
    out[5] = lroundf(fv(f + 3));
    out[6] = lrintf(fv(f + 4));
    out[7] = lrintf(fv(f + 5));
    out[8] = lrintf(fv(f + 6));
    out[9] = lrintf(fv(f + 7));
    out[10] = lrintf(fv(f + 8));
    out[11] = llrintf(fv(f + 2));
    out[12] = llroundf(fv(f + 2));
    out[13] = lrint(dv(d + 0));
    out[14] = lround(dv(d + 0));
    out[15] = lrint(dv(d + 1));
    out[16] = lrint(dv(d + 2));
    out[17] = lrint(dv(d + 3));
    out[18] = lrint(dv(d + 4));
    out[19] = llrint(dv(d + 2));
    out[20] = llround(dv(d + 3));
}

int main() {
    unsigned fbits[9] = {0x3f000000u, 0xbf000000u, 0x40200000u, 0xc0200000u, 0x4f000000u,
                         0x5effffffu, 0x7fc00001u, 0x7f800000u, 0xdf000000u};
    unsigned long long dbits[5] = {0x4320000000000001ull, 0x432fffffffffffffull, 0x43e0000000000000ull,
                                   0xc3e0000000000000ull, 0x7ff8000000000001ull};
    unsigned* f = nullptr;
    unsigned long long* d = nullptr;
    long long* out = nullptr;
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&d, sizeof(dbits)));
    CHECK(cudaMalloc(&out, 21 * sizeof(long long)));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, dbits, sizeof(dbits), cudaMemcpyHostToDevice));
    round_to_int<<<1, 1>>>(f, d, out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    long long got[21] = {};
    CHECK(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost));
    const char* names[21] = {"rhalf", "ohalf", "nhalf", "r25",  "o25",  "n25", "p31",
                             "fhi",   "fnan",  "finf",  "n63",   "ll25", "llo25", "dlr",
                             "dlo",   "d52",   "d63",   "dn63",  "dnan", "ll63", "lldn"};
    for (int i = 0; i < 21; ++i) std::printf("%s=%lld\n", names[i], got[i]);
    return 0;
}
