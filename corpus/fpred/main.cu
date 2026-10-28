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

// Sign and classification. Inputs come from host memory so nvcc cannot
// fold a negative zero, a signaling NaN, or an infinity.
__global__ void classify(const unsigned* f, const unsigned long long* d, int* out) {
    float p0 = fv(f + 0);
    float m0 = fv(f + 1);
    float nan = fv(f + 2);
    float nnan = fv(f + 3);
    float sig = fv(f + 4);
    float inf = fv(f + 5);
    float ninf = fv(f + 6);
    float sub = fv(f + 7);
    out[0] = signbit(p0);
    out[1] = signbit(m0);
    out[2] = signbit(nnan);
    out[3] = signbit(nan);
    out[4] = signbit(ninf);
    out[5] = isnan(nan);
    out[6] = isnan(sig);
    out[7] = isnan(inf);
    out[8] = isinf(inf);
    out[9] = isinf(ninf);
    out[10] = isinf(nan);
    out[11] = isfinite(sub);
    out[12] = isfinite(inf);
    out[13] = isfinite(nan);
    out[14] = isfinite(m0);

    double dm0 = dv(d + 0);
    double ndnan = dv(d + 1);
    double dsig = dv(d + 2);
    double ndsig = dv(d + 3);
    double dninf = dv(d + 4);
    double dsub = dv(d + 5);
    out[15] = signbit(dm0);
    out[16] = signbit(ndnan);
    out[17] = isnan(dsig);
    out[18] = isnan(ndsig);
    out[19] = isinf(dninf);
    out[20] = isinf(dsig);
    out[21] = isfinite(dsub);
    out[22] = isfinite(ndnan);
}

int main() {
    unsigned fbits[8] = {0u, 0x80000000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u, 0x7f800000u, 0xff800000u, 1u};
    unsigned long long dbits[6] = {0x8000000000000000ull, 0xfff8000000000001ull, 0x7ff0000000000001ull,
                                   0xfff0000000000001ull, 0xfff0000000000000ull, 1ull};
    unsigned* f = nullptr;
    unsigned long long* d = nullptr;
    int* out = nullptr;
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&d, sizeof(dbits)));
    CHECK(cudaMalloc(&out, 23 * sizeof(int)));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, dbits, sizeof(dbits), cudaMemcpyHostToDevice));
    classify<<<1, 1>>>(f, d, out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int got[23] = {};
    CHECK(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost));
    const char* names[23] = {"sb0", "sbm", "sbn", "sbp", "sbi", "nnan", "nsig", "ninf", "iinf", "ininf",
                             "inan", "fsub", "finf", "fnan", "fm0", "dsbm", "dsbn", "dnsig", "dnnsig",
                             "dinf", "dnan", "dfsub", "dfnan"};
    for (int i = 0; i < 23; ++i) std::printf("%s=%d\n", names[i], got[i]);
    return 0;
}
