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

__device__ unsigned long long db(double x) {
    union { double as_double; unsigned long long as_bits; } v;
    v.as_double = x;
    return v.as_bits;
}

// Split, scale, and remainder. Inputs come from host memory so nvcc cannot
// fold a halfway power, a NaN, or an infinity.
__global__ void split(const unsigned* f, const unsigned long long* d, unsigned long long* bits,
                      int* expo) {
    float sub = fv(f + 0);
    float zero = fv(f + 1);
    float inf = fv(f + 2);
    float one = fv(f + 3);
    float eps = fv(f + 4);
    float seven = fv(f + 5);
    float two = fv(f + 6);
    float ninf = fv(f + 7);
    float pinf = fv(f + 8);
    float minf = fv(f + 9);
    int fexp = 0;
    bits[0] = fb(frexpf(sub, &fexp));
    expo[0] = fexp;
    expo[1] = ilogbf(zero);
    expo[2] = ilogbf(inf);
    bits[1] = fb(ldexpf(one, -150));
    bits[2] = fb(ldexpf(eps, -150));
    bits[3] = fb(ldexpf(one, 128));
    float ip = 0.0f;
    bits[4] = fb(modff(ninf, &ip));
    bits[5] = fb(fmodf(seven, two));
    bits[6] = fb(remainderf(seven, two));
    bits[7] = fb(fmodf(one, zero));
    bits[8] = fb(fdimf(pinf, pinf));
    bits[9] = fb(fdimf(pinf, minf));

    double dsub = dv(d + 0);
    double done = dv(d + 1);
    double dzero = dv(d + 2);
    double dseven = dv(d + 3);
    double dtwo = dv(d + 4);
    double dnan = dv(d + 5);
    double dnan5 = dv(d + 6);
    int dexp = 0;
    bits[10] = db(frexp(dsub, &dexp));
    expo[3] = dexp;
    bits[11] = db(fmod(done, dzero));
    bits[12] = db(remainder(dseven, dtwo));
    bits[13] = db(fmod(dnan, done));
    bits[14] = db(fmod(done, dnan5));
}

int main() {
    unsigned fbits[10] = {0x00000001u, 0x00000000u, 0x7f800000u, 0x3f800000u, 0x3f800001u,
                          0x40e00000u, 0x40000000u, 0xff800000u, 0x7f800000u, 0xff800000u};
    unsigned long long dbits[7] = {0x0000000000000001ull, 0x3ff0000000000000ull, 0x0000000000000000ull,
                                   0x401c000000000000ull, 0x4000000000000000ull, 0x7ff8000000000001ull,
                                   0x7ff8000000000005ull};
    unsigned* f = nullptr;
    unsigned long long* d = nullptr;
    unsigned long long* bits = nullptr;
    int* expo = nullptr;
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&d, sizeof(dbits)));
    CHECK(cudaMalloc(&bits, 15 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&expo, 4 * sizeof(int)));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, dbits, sizeof(dbits), cudaMemcpyHostToDevice));
    split<<<1, 1>>>(f, d, bits, expo);
    CHECK(cudaDeviceSynchronize());
    unsigned long long hb[15] = {};
    int he[4] = {};
    CHECK(cudaMemcpy(hb, bits, sizeof(hb), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(he, expo, sizeof(he), cudaMemcpyDeviceToHost));
    const char* bnames[15] = {"fsig", "llo", "lup",  "lhi",   "mfrac", "f7",   "r7",  "fnan",
                              "fd0",  "fdinf", "dsig", "dinv", "d7",    "dnan", "dynan"};
    const char* enames[4] = {"fexp", "ilog0", "ilogi", "dexp"};
    for (int i = 0; i < 15; ++i) std::printf("%s=%llu\n", bnames[i], hb[i]);
    for (int i = 0; i < 4; ++i) std::printf("%s=%d\n", enames[i], he[i]);
    return 0;
}
