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

__device__ int iv(const int* p) {
    volatile int held = *p;
    return held;
}

__device__ long lv(const long long* p) {
    volatile long long held = *p;
    return (long)held;
}

__device__ unsigned fb(float x) {
    union { float as_float; unsigned as_bits; } v;
    v.as_float = x;
    return v.as_bits;
}

__device__ unsigned long long db(double x) {
    union { double as_double; unsigned long long as_bits; } v;
    v.as_double = x;
    return v.as_bits;
}

// Exponent as a float, and a scale by a 32-bit or 64-bit power of two.
// Inputs come from host memory so nvcc cannot fold a zero, a NaN, or a
// long exponent.
__global__ void powers(const unsigned* f, const unsigned long long* d, const int* e, const long long* el,
                       unsigned* fout, unsigned long long* dout) {
    float zero = fv(f + 0);
    float mzero = fv(f + 1);
    float inf = fv(f + 2);
    float ninf = fv(f + 3);
    float nan = fv(f + 4);
    float sub = fv(f + 5);
    float one = fv(f + 6);
    float half = fv(f + 7);
    float mx = fv(f + 8);
    int four = iv(e + 0);
    int down = iv(e + 1);
    int three = iv(e + 2);
    int neg5 = iv(e + 3);
    long huge = lv(el + 0);
    long tiny = lv(el + 1);
    long up = lv(el + 2);
    fout[0] = fb(logbf(zero));
    fout[1] = fb(logbf(mzero));
    fout[2] = fb(logbf(inf));
    fout[3] = fb(logbf(ninf));
    fout[4] = fb(logbf(nan));
    fout[5] = fb(logbf(sub));
    fout[6] = fb(logbf(one));
    fout[7] = fb(logbf(half));
    fout[8] = fb(logbf(mx));
    fout[9] = fb(scalbnf(mzero, four));
    fout[10] = fb(scalbnf(half, down));
    fout[11] = fb(scalbnf(nan, three));
    fout[12] = fb(scalbnf(inf, neg5));
    fout[13] = fb(scalblnf(one, huge));
    fout[14] = fb(scalblnf(one, tiny));
    fout[15] = fb(scalblnf(sub, up));

    double dzero = dv(d + 0);
    double dninf = dv(d + 1);
    double dsig = dv(d + 2);
    double dsub = dv(d + 3);
    double done = dv(d + 4);
    dout[0] = db(logb(dzero));
    dout[1] = db(logb(-dzero));
    dout[2] = db(logb(dninf));
    dout[3] = db(logb(-dninf));
    dout[4] = db(logb(dsig));
    dout[5] = db(logb(dsub));
    dout[6] = db(scalbln(done, huge));
    dout[7] = db(scalbln(done, tiny));
    dout[8] = db(scalbn(dsig, three));
}

int main() {
    unsigned fbits[9] = {0u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00001u, 1u, 0x3f800000u,
                         0x3f000000u, 0x7f7fffffu};
    unsigned long long dbits[5] = {0ull, 0xfff0000000000000ull, 0x7ff0000000000001ull, 1ull,
                                   0x3ff0000000000000ull};
    int ebits[4] = {4, -126, 3, -5};
    long long lbits[3] = {2147483648ll, (-9223372036854775807ll - 1), 200ll};
    unsigned* f = nullptr;
    unsigned long long* d = nullptr;
    int* e = nullptr;
    long long* el = nullptr;
    unsigned* fout = nullptr;
    unsigned long long* dout = nullptr;
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&d, sizeof(dbits)));
    CHECK(cudaMalloc(&e, sizeof(ebits)));
    CHECK(cudaMalloc(&el, sizeof(lbits)));
    CHECK(cudaMalloc(&fout, 16 * sizeof(unsigned)));
    CHECK(cudaMalloc(&dout, 9 * sizeof(unsigned long long)));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, dbits, sizeof(dbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(e, ebits, sizeof(ebits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(el, lbits, sizeof(lbits), cudaMemcpyHostToDevice));
    powers<<<1, 1>>>(f, d, e, el, fout, dout);
    CHECK(cudaDeviceSynchronize());
    unsigned hf[16] = {};
    unsigned long long hd[9] = {};
    CHECK(cudaMemcpy(hf, fout, sizeof(hf), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hd, dout, sizeof(hd), cudaMemcpyDeviceToHost));
    const char* fn[16] = {"z", "mz", "pinf", "ninf", "nan", "sub", "one", "half", "mx",
                          "sm0", "slow", "snan", "sinf", "shuge", "stiny", "ssub"};
    for (int i = 0; i < 16; ++i) std::printf("%s=%u\n", fn[i], hf[i]);
    const char* dn[9] = {"dz", "dmz", "dninf", "dpinf", "dsig", "dsub", "dhuge", "dtiny", "dsnan"};
    for (int i = 0; i < 9; ++i) std::printf("%s=%llu\n", dn[i], hd[i]);
    return 0;
}
