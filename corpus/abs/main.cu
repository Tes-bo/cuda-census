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

__device__ long long lv(const long long* p) {
    volatile long long held = *p;
    return held;
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

// Absolute value. Inputs come from host memory so nvcc cannot fold INT_MIN,
// LLONG_MIN, a NaN, or a negative zero.
__global__ void magnitudes(const int* n, const long long* w, const unsigned* f,
                           const unsigned long long* d, int* io, long long* lo, unsigned* fo,
                           unsigned long long* dlo) {
    int zero = iv(n + 0);
    int imin = iv(n + 1);
    int ineg = iv(n + 2);
    long lmin = (long)lv(w + 0);
    long long llmin = lv(w + 1);
    io[0] = abs(zero);
    io[1] = abs(imin);
    io[2] = abs(ineg);
    lo[0] = abs(lmin);
    lo[1] = abs(llmin);
    lo[2] = labs(lmin);
    lo[3] = llabs(llmin);

    float m0 = fv(f + 0);
    float nan = fv(f + 1);
    float nnan = fv(f + 2);
    float sig = fv(f + 3);
    float nsig = fv(f + 4);
    float ninf = fv(f + 5);
    float nsub = fv(f + 6);
    float mone = fv(f + 7);
    fo[0] = fb(fabsf(m0));
    fo[1] = fb(fabsf(nan));
    fo[2] = fb(fabsf(nnan));
    fo[3] = fb(fabsf(sig));
    fo[4] = fb(fabsf(nsig));
    fo[5] = fb(fabsf(ninf));
    fo[6] = fb(fabsf(nsub));
    fo[7] = fb(abs(mone));

    double dm0 = dv(d + 0);
    double dnan = dv(d + 1);
    double ndnan = dv(d + 2);
    double dsig = dv(d + 3);
    double ndsig = dv(d + 4);
    double dninf = dv(d + 5);
    double dsub = dv(d + 6);
    dlo[0] = db(fabs(dm0));
    dlo[1] = db(fabs(dnan));
    dlo[2] = db(fabs(ndnan));
    dlo[3] = db(fabs(dsig));
    dlo[4] = db(fabs(ndsig));
    dlo[5] = db(fabs(dninf));
    dlo[6] = db(fabs(dsub));
    dlo[7] = db(abs(ndnan));
}

int main() {
    int nbits[3] = {0, -2147483647 - 1, -3};
    long long wbits[2] = {-2147483648ll, -9223372036854775807ll - 1};
    unsigned fbits[8] = {0x80000000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u, 0xff800001u, 0xff800000u,
                         0x80000001u, 0xbf800000u};
    unsigned long long dbits[7] = {0x8000000000000000ull, 0x7ff8000000000001ull, 0xfff8000000000001ull,
                                   0x7ff0000000000001ull, 0xfff0000000000001ull, 0xfff0000000000000ull, 1ull};
    int* n = nullptr;
    long long* w = nullptr;
    unsigned* f = nullptr;
    unsigned long long* d = nullptr;
    int* io = nullptr;
    long long* lo = nullptr;
    unsigned* fo = nullptr;
    unsigned long long* dlo = nullptr;
    CHECK(cudaMalloc(&n, sizeof(nbits)));
    CHECK(cudaMalloc(&w, sizeof(wbits)));
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&d, sizeof(dbits)));
    CHECK(cudaMalloc(&io, 3 * sizeof(int)));
    CHECK(cudaMalloc(&lo, 4 * sizeof(long long)));
    CHECK(cudaMalloc(&fo, 8 * sizeof(unsigned)));
    CHECK(cudaMalloc(&dlo, 8 * sizeof(unsigned long long)));
    CHECK(cudaMemcpy(n, nbits, sizeof(nbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(w, wbits, sizeof(wbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, dbits, sizeof(dbits), cudaMemcpyHostToDevice));
    magnitudes<<<1, 1>>>(n, w, f, d, io, lo, fo, dlo);
    CHECK(cudaDeviceSynchronize());
    int hi[3] = {};
    long long hl[4] = {};
    unsigned hf[8] = {};
    unsigned long long hd[8] = {};
    CHECK(cudaMemcpy(hi, io, sizeof(hi), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hl, lo, sizeof(hl), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hf, fo, sizeof(hf), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hd, dlo, sizeof(hd), cudaMemcpyDeviceToHost));
    const char* inames[3] = {"a0", "amin", "aneg"};
    for (int i = 0; i < 3; ++i) std::printf("%s=%d\n", inames[i], hi[i]);
    const char* lnames[4] = {"along", "absll", "labsmin", "llmin"};
    for (int i = 0; i < 4; ++i) std::printf("%s=%lld\n", lnames[i], hl[i]);
    const char* fnames[8] = {"fm0", "fnan", "fnnan", "fsig", "fnsig", "fninf", "fsub", "am1"};
    for (int i = 0; i < 8; ++i) std::printf("%s=%u\n", fnames[i], hf[i]);
    const char* dnames[8] = {"dm0", "dnan", "ndnan", "dsig", "ndsig", "dninf", "dsub", "adnan"};
    for (int i = 0; i < 8; ++i) std::printf("%s=%llu\n", dnames[i], hd[i]);
    return 0;
}
