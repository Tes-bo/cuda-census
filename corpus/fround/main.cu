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

// floor, ceil, trunc, rint, nearbyint, and round. Inputs come from host
// memory so nvcc cannot fold a halfway value or a NaN.
__global__ void round_to_int(const unsigned* f, const unsigned long long* d,
                             unsigned long long* o) {
    float neg0 = fv(f + 0);
    float half = fv(f + 1);
    float nhalf = fv(f + 2);
    float two = fv(f + 3);
    float ntwo = fv(f + 4);
    float nan = fv(f + 5);
    float nsub = fv(f + 6);
    double dhalf = dv(d + 0);
    double dnhalf = dv(d + 1);
    double dtwo = dv(d + 2);
    double dnan = dv(d + 3);
    double dnsub = dv(d + 4);
    o[0] = fb(floorf(neg0));
    o[1] = fb(rintf(half));
    o[2] = fb(nearbyintf(half));
    o[3] = fb(roundf(half));
    o[4] = fb(rintf(two));
    o[5] = fb(roundf(two));
    o[6] = fb(rintf(ntwo));
    o[7] = fb(roundf(ntwo));
    o[8] = fb(floorf(nan));
    o[9] = fb(floorf(nsub));
    o[10] = fb(truncf(nsub));
    o[11] = fb(ceilf(nsub));
    o[12] = db(rint(dhalf));
    o[13] = db(round(dhalf));
    o[14] = db(round(dnhalf));
    o[15] = db(rint(dtwo));
    o[16] = db(nearbyint(dtwo));
    o[17] = db(round(dtwo));
    o[18] = db(floor(dnan));
    o[19] = db(floor(dnsub));
    o[20] = db(trunc(dnsub));
}

int main() {
    unsigned fbits[7] = {0x80000000u, 0x3f000000u, 0xbf000000u, 0x40200000u,
                         0xc0200000u, 0x7fc00001u, 0x80000001u};
    unsigned long long dbits[5] = {0x3fe0000000000000ull, 0xbfe0000000000000ull,
                                   0x4004000000000000ull, 0x7ff0000000000001ull,
                                   0x8000000000000001ull};
    unsigned* f = nullptr;
    unsigned long long* d = nullptr;
    unsigned long long* o = nullptr;
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&d, sizeof(dbits)));
    CHECK(cudaMalloc(&o, 21 * sizeof(unsigned long long)));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, dbits, sizeof(dbits), cudaMemcpyHostToDevice));
    round_to_int<<<1, 1>>>(f, d, o);
    CHECK(cudaDeviceSynchronize());
    unsigned long long h[21] = {};
    CHECK(cudaMemcpy(h, o, sizeof(h), cudaMemcpyDeviceToHost));
    const char* names[21] = {"fz",    "rhalf", "nhalf", "ohalf", "r25",   "o25",  "rn25",
                             "on25",  "fnan",  "fsub",  "tsub",  "csub",  "drhalf", "dohalf",
                             "dn05",  "dr25",  "n25",   "do25",  "dnan",  "dnsub", "dtnsub"};
    for (int i = 0; i < 21; ++i) std::printf("%s=%llu\n", names[i], h[i]);
    return 0;
}
