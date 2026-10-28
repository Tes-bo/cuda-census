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

// Round-to-nearest fused multiply-add. Inputs come from host memory so nvcc
// cannot fold a signed zero, a NaN, or the extra product bit. The rounded
// product uses __fmul_rn and __dmul_rn so no compiler may contract it into
// the following add.
__global__ void fused(const unsigned* f, const unsigned long long* d, unsigned* fo, unsigned long long* dout) {
    float a = fv(f + 0);
    float one = fv(f + 1);
    float p0 = fv(f + 2);
    float m0 = fv(f + 3);
    float nan = fv(f + 4);
    float nnan = fv(f + 5);
    float sig = fv(f + 6);
    float inf = fv(f + 7);
    float ninf = fv(f + 8);
    float prod = __fmul_rn(a, a);
    fo[0] = fb(fmaf(a, a, -prod));
    fo[1] = fb(prod + (-prod));
    fo[2] = fb(fma(a, a, -prod));
    fo[3] = fb(fmaf(m0, p0, m0));
    fo[4] = fb(fmaf(m0, m0, m0));
    fo[5] = fb(fmaf(nan, one, one));
    fo[6] = fb(fmaf(one, nnan, one));
    fo[7] = fb(fmaf(one, one, sig));
    fo[8] = fb(fmaf(inf, p0, one));
    fo[9] = fb(fmaf(inf, one, ninf));
    fo[10] = fb(fmaf(inf, one, inf));
    double da = dv(d + 0);
    double done = dv(d + 1);
    double dm0 = dv(d + 2);
    double dnan = dv(d + 3);
    double ndnan = dv(d + 4);
    double dsig = dv(d + 5);
    double ndsig = dv(d + 6);
    double dinf = dv(d + 7);
    double dninf = dv(d + 8);
    double zp = dv(d + 9);
    double dprod = __dmul_rn(da, da);
    dout[0] = dbits(fma(da, da, -dprod));
    dout[1] = dbits(dprod + (-dprod));
    dout[2] = dbits(fma(dnan, done, done));
    dout[3] = dbits(fma(done, ndnan, done));
    dout[4] = dbits(fma(done, done, dsig));
    dout[5] = dbits(fma(done, ndsig, done));
    dout[6] = dbits(fma(dinf, zp, done));
    dout[7] = dbits(fma(dinf, done, dninf));
    dout[8] = dbits(fma(dm0, dm0, dm0));
}

int main() {
    unsigned fbits[9] = {0x3f800001u, 0x3f800000u, 0u, 0x80000000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u,
                         0x7f800000u, 0xff800000u};
    unsigned long long dbits_in[10] = {0x3ff0000000000001ull, 0x3ff0000000000000ull, 0x8000000000000000ull,
                                       0x7ff8000000000001ull, 0xfff8000000000001ull, 0x7ff0000000000001ull,
                                       0xfff0000000000001ull, 0x7ff0000000000000ull, 0xfff0000000000000ull, 0ull};
    unsigned* f = nullptr;
    unsigned long long* d = nullptr;
    unsigned* fo = nullptr;
    unsigned long long* dout = nullptr;
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&d, sizeof(dbits_in)));
    CHECK(cudaMalloc(&fo, 11 * sizeof(unsigned)));
    CHECK(cudaMalloc(&dout, 9 * sizeof(unsigned long long)));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, dbits_in, sizeof(dbits_in), cudaMemcpyHostToDevice));
    fused<<<1, 1>>>(f, d, fo, dout);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned fg[11] = {};
    unsigned long long dg[9] = {};
    CHECK(cudaMemcpy(fg, fo, sizeof(fg), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dg, dout, sizeof(dg), cudaMemcpyDeviceToHost));
    const char* fn[11] = {"gap", "split", "fgap", "zm0", "zz", "nan", "nnan", "sig", "inf0", "infninf", "infinf"};
    const char* dn[9] = {"dgap", "dsplit", "dnan", "ndnan", "dsig", "ndsig", "dinf0", "dinfcanc", "dm0"};
    for (int i = 0; i < 11; ++i) std::printf("%s=%u\n", fn[i], fg[i]);
    for (int i = 0; i < 9; ++i) std::printf("%s=%llu\n", dn[i], dg[i]);
    return 0;
}
