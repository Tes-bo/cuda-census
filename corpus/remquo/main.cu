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

// Remainder and a 3-bit quotient. Inputs come from host memory so nvcc
// cannot fold a tie, a NaN, or a quotient that does not fit in 3 bits.
__global__ void quotients(const unsigned* f, const unsigned long long* d, unsigned* fout,
                          int* qout, unsigned long long* dout, int* dq) {
    float seven = fv(f + 0);
    float two = fv(f + 1);
    float fifteen = fv(f + 2);
    float sixteen = fv(f + 3);
    float one = fv(f + 4);
    float odd = fv(f + 5);
    float nseven = fv(f + 6);
    float nan = fv(f + 7);
    float inf = fv(f + 8);
    float zero = fv(f + 9);
    float n16 = fv(f + 10);
    int q = 0;
    fout[0] = fb(remquof(seven, two, &q));
    qout[0] = q;
    fout[1] = fb(remquof(fifteen, two, &q));
    qout[1] = q;
    fout[2] = fb(remquof(sixteen, one, &q));
    qout[2] = q;
    fout[3] = fb(remquof(fifteen, one, &q));
    qout[3] = q;
    fout[4] = fb(remquof(odd, two, &q));
    qout[4] = q;
    fout[5] = fb(remquof(nseven, two, &q));
    qout[5] = q;
    fout[6] = fb(remquof(nan, two, &q));
    qout[6] = q;
    fout[7] = fb(remquof(inf, two, &q));
    qout[7] = q;
    fout[8] = fb(remquof(seven, zero, &q));
    qout[8] = q;
    fout[9] = fb(remquof(n16, one, &q));
    qout[9] = q;

    double dseven = dv(d + 0);
    double dtwo = dv(d + 1);
    double dnan = dv(d + 2);
    double done = dv(d + 3);
    double dzero = dv(d + 4);
    double dfifteen = dv(d + 5);
    double dsixteen = dv(d + 6);
    dout[0] = db(remquo(dseven, dtwo, &q));
    dq[0] = q;
    dout[1] = db(remquo(dnan, dtwo, &q));
    dq[1] = q;
    dout[2] = db(remquo(done, dzero, &q));
    dq[2] = q;
    dout[3] = db(remquo(dfifteen, done, &q));
    dq[3] = q;
    dout[4] = db(remquo(dsixteen, done, &q));
    dq[4] = q;
}

int main() {
    unsigned fbits[11] = {0x40e00000u, 0x40000000u, 0x41700000u, 0x41800000u, 0x3f800000u,
                          0x4b000001u, 0xc0e00000u, 0x7fc00001u, 0x7f800000u, 0x00000000u,
                          0xc1800000u};
    unsigned long long dbits[7] = {0x401c000000000000ull, 0x4000000000000000ull, 0x7ff0000000000001ull,
                                   0x3ff0000000000000ull, 0x0000000000000000ull, 0x402e000000000000ull,
                                   0x4030000000000000ull};
    unsigned* f = nullptr;
    unsigned long long* d = nullptr;
    unsigned* fout = nullptr;
    int* qout = nullptr;
    unsigned long long* dout = nullptr;
    int* dq = nullptr;
    CHECK(cudaMalloc(&f, sizeof(fbits)));
    CHECK(cudaMalloc(&d, sizeof(dbits)));
    CHECK(cudaMalloc(&fout, 10 * sizeof(unsigned)));
    CHECK(cudaMalloc(&qout, 10 * sizeof(int)));
    CHECK(cudaMalloc(&dout, 5 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&dq, 5 * sizeof(int)));
    CHECK(cudaMemcpy(f, fbits, sizeof(fbits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, dbits, sizeof(dbits), cudaMemcpyHostToDevice));
    quotients<<<1, 1>>>(f, d, fout, qout, dout, dq);
    CHECK(cudaDeviceSynchronize());
    unsigned hf[10] = {};
    int hq[10] = {};
    unsigned long long hd[5] = {};
    int hdq[5] = {};
    CHECK(cudaMemcpy(hf, fout, sizeof(hf), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hq, qout, sizeof(hq), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hd, dout, sizeof(hd), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hdq, dq, sizeof(hdq), cudaMemcpyDeviceToHost));
    const char* fn[10] = {"r7", "r15", "r16", "r15one", "rodd", "rn", "rnan", "rinf", "r0", "rm0"};
    const char* qn[10] = {"q7", "q15", "q16", "q15one", "qodd", "qn", "qnan", "qinf", "q0", "qm0"};
    for (int i = 0; i < 10; ++i) {
        std::printf("%s=%u\n", fn[i], hf[i]);
        std::printf("%s=%d\n", qn[i], hq[i]);
    }
    const char* dn[5] = {"d7", "dnan", "dinv", "d15", "d16"};
    const char* dqn[5] = {"dq7", "dqnan", "dqinv", "dq15", "dq16"};
    for (int i = 0; i < 5; ++i) {
        std::printf("%s=%llu\n", dn[i], hd[i]);
        std::printf("%s=%d\n", dqn[i], hdq[i]);
    }
    return 0;
}
