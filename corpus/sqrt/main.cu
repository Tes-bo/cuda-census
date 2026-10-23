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

__device__ unsigned fb(float x) {
    union { float as_float; unsigned as_bits; } v;
    v.as_float = x;
    return v.as_bits;
}

__device__ unsigned long long dbits(double x) {
    union { double as_double; unsigned long long as_bits; } v;
    v.as_double = x;
    return v.as_bits;
}

// Round-to-nearest square root. Inputs come from host memory so nvcc
// cannot fold a negative zero or a NaN.
__global__ void roots(const unsigned* f, const unsigned long long* d, unsigned* fo, unsigned long long* dout) {
    float p0 = fv(f + 0);
    float m0 = fv(f + 1);
    float neg = fv(f + 2);
    float nan = fv(f + 3);
    float nnan = fv(f + 4);
    float sig = fv(f + 5);
    float ninf = fv(f + 6);
    float pinf = fv(f + 7);
    float sub = fv(f + 8);
    float four = fv(f + 9);
    fo[0] = fb(sqrtf(p0));
    fo[1] = fb(sqrtf(m0));
    fo[2] = fb(sqrtf(neg));
    fo[3] = fb(sqrtf(nan));
    fo[4] = fb(sqrtf(nnan));
    fo[5] = fb(sqrtf(sig));
    fo[6] = fb(sqrtf(ninf));
    fo[7] = fb(sqrtf(pinf));
    fo[8] = fb(sqrtf(sub));
    fo[9] = fb(sqrtf(four));
    double dp0 = dv(d + 0);
    double dm0 = dv(d + 1);
    double dneg = dv(d + 2);
    double dnan = dv(d + 3);
    double ndnan = dv(d + 4);
    double dsig = dv(d + 5);
    double ndsig = dv(d + 6);
    double dninf = dv(d + 7);
    double dpinf = dv(d + 8);
    double dfour = dv(d + 9);
    dout[0] = dbits(sqrt(dp0));
    dout[1] = dbits(sqrt(dm0));
    dout[2] = dbits(sqrt(dneg));
    dout[3] = dbits(sqrt(dnan));
    dout[4] = dbits(sqrt(ndnan));
    dout[5] = dbits(sqrt(dsig));
    dout[6] = dbits(sqrt(ndsig));
    dout[7] = dbits(sqrt(dninf));
    dout[8] = dbits(sqrt(dpinf));
    dout[9] = dbits(sqrt(dfour));
}

int main() {
    unsigned fbits[10] = {0u, 0x80000000u, 0xbf800000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u, 0xff800000u,
                          0x7f800000u, 1u, 0x40800000u};
    unsigned long long dbits_in[10] = {0ull, 0x8000000000000000ull, 0xbff0000000000000ull, 0x7ff8000000000001ull,
                                       0xfff8000000000001ull, 0x7ff0000000000001ull, 0xfff0000000000001ull,
                                       0xfff0000000000000ull, 0x7ff0000000000000ull, 0x4010000000000000ull};
    unsigned* f = nullptr;
    unsigned long long* d = nullptr;
    unsigned* fo = nullptr;
    unsigned long long* dout = nullptr;
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&d, sizeof(dbits_in)));
    CHECK(cudaMalloc(&fo, sizeof(fbits)));
    CHECK(cudaMalloc(&dout, sizeof(dbits_in)));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, dbits_in, sizeof(dbits_in), cudaMemcpyHostToDevice));
    roots<<<1, 1>>>(f, d, fo, dout);
    CHECK(cudaDeviceSynchronize());
    unsigned fg[10] = {};
    unsigned long long dg[10] = {};
    CHECK(cudaMemcpy(fg, fo, sizeof(fg), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dg, dout, sizeof(dg), cudaMemcpyDeviceToHost));
    const char* fn[10] = {"fp0", "fm0", "fneg", "fnan", "fnnan", "fsig", "fninf", "fpinf", "fsub", "ffour"};
    const char* dn[10] = {"dp0", "dm0", "dneg", "dnan", "ndnan", "dsig", "ndsig", "dninf", "dpinf", "dfour"};
    for (int i = 0; i < 10; ++i) std::printf("%s=%u\n", fn[i], fg[i]);
    for (int i = 0; i < 10; ++i) std::printf("%s=%llu\n", dn[i], dg[i]);
    return 0;
}
