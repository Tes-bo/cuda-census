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

__device__ long long ls(const long long* p) {
    volatile long long held = *p;
    return held;
}

__device__ unsigned long long lu(const unsigned long long* p) {
    volatile unsigned long long held = *p;
    return held;
}

__device__ int li(const int* p) {
    volatile int held = *p;
    return held;
}

__device__ unsigned lui(const unsigned* p) {
    volatile unsigned held = *p;
    return held;
}

__device__ unsigned long long dbits(double x) {
    union { double as_double; unsigned long long as_bits; } v;
    v.as_double = x;
    return v.as_bits;
}

// Integer to double. Values come from host memory so nvcc cannot fold a tie.
__global__ void widen(const long long* s, const unsigned long long* u, const int* si,
                      const unsigned* ui, unsigned long long* o) {
    long long tie = ls(s + 0);
    long long tie3 = ls(s + 1);
    long long llmax = ls(s + 2);
    long long llmin = ls(s + 3);
    long long ntie = ls(s + 4);
    long long ntie3 = ls(s + 5);
    unsigned long long half = lu(u + 0);
    unsigned long long ullmax = lu(u + 1);
    int imin = li(si + 0);
    int imax = li(si + 1);
    unsigned umax = lui(ui);
    o[0] = dbits(__ll2double_rn(tie));
    o[1] = dbits(__ll2double_ru(tie));
    o[2] = dbits(__ll2double_rn(tie3));
    o[3] = dbits(__ll2double_rz(tie3));
    o[4] = dbits(__ll2double_rn(llmax));
    o[5] = dbits(__ll2double_rz(llmax));
    o[6] = dbits(__ll2double_rn(llmin));
    o[7] = dbits(__ll2double_rn(ntie));
    o[8] = dbits(__ll2double_rd(ntie));
    o[9] = dbits(__ll2double_rn(ntie3));
    o[10] = dbits(__ll2double_ru(ntie3));
    o[11] = dbits(__ull2double_rn(half));
    o[12] = dbits(__ull2double_ru(half));
    o[13] = dbits(__ull2double_rn(ullmax));
    o[14] = dbits(__ull2double_rz(ullmax));
    o[15] = dbits(__int2double_rn(imin));
    o[16] = dbits(__int2double_rn(imax));
    o[17] = dbits(__uint2double_rn(umax));
}

int main() {
    long long sbits[6] = {9007199254740993ll, 9007199254740995ll, 9223372036854775807ll,
                          (long long)0x8000000000000000ull, -9007199254740993ll, -9007199254740995ll};
    unsigned long long ubits[2] = {4611686018427388416ull, 0xffffffffffffffffull};
    int ibits[2] = {(int)0x80000000, 2147483647};
    unsigned uibits[1] = {0xffffffffu};
    long long* s = nullptr;
    unsigned long long* u = nullptr;
    int* si = nullptr;
    unsigned* ui = nullptr;
    unsigned long long* o = nullptr;
    CHECK(cudaMalloc(&s, sizeof(sbits)));
    CHECK(cudaMalloc(&u, sizeof(ubits)));
    CHECK(cudaMalloc(&si, sizeof(ibits)));
    CHECK(cudaMalloc(&ui, sizeof(uibits)));
    CHECK(cudaMalloc(&o, 18 * sizeof(unsigned long long)));
    CHECK(cudaMemcpy(s, sbits, sizeof(sbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(u, ubits, sizeof(ubits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(si, ibits, sizeof(ibits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ui, uibits, sizeof(uibits), cudaMemcpyHostToDevice));
    widen<<<1, 1>>>(s, u, si, ui, o);
    CHECK(cudaDeviceSynchronize());
    unsigned long long h[18] = {};
    CHECK(cudaMemcpy(h, o, sizeof(h), cudaMemcpyDeviceToHost));
    const char* names[18] = {"tie",    "tieru", "tie3",   "tie3rz", "llmax", "llmaxrz",
                             "llmin",  "ntie",  "ntierd", "ntie3",  "ntie3ru", "halfrn",
                             "halfru", "ullrn", "ullrz",  "imin",   "imax",    "umax"};
    for (int i = 0; i < 18; ++i) std::printf("%s=%llu\n", names[i], h[i]);
    return 0;
}
