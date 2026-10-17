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

__device__ double dv(unsigned long long bits) {
    volatile unsigned long long held = bits;
    union { double as_double; unsigned long long as_bits; } v;
    v.as_bits = held;
    return v.as_double;
}

__device__ unsigned long long dbits(double x) {
    union { double as_double; unsigned long long as_bits; } v;
    v.as_double = x;
    return v.as_bits;
}

// Directed double division, reciprocal, and square root. NaN payloads are
// kept and quieted, unlike the float forms. Subnormal quotients round by
// mode, and a quotient that overflows toward zero stops at the finite maximum.
__global__ void round_ddiv(const unsigned long long* in, unsigned long long* u) {
    double p0 = dv(in[0]);
    double m0 = dv(in[1]);
    double nan1 = dv(in[2]);
    double nan2 = dv(in[3]);
    double one = dv(in[4]);
    double three = dv(in[5]);
    double inf = dv(in[6]);
    double sub = dv(in[7]);
    double big = dv(in[8]);
    double two = dv(in[9]);
    double neg = dv(in[10]);
    u[0] = dbits(__ddiv_rn(p0, m0));
    u[1] = dbits(__ddiv_rn(one, p0));
    u[2] = dbits(__ddiv_rn(one, m0));
    u[3] = dbits(__ddiv_rn(nan1, nan2));
    u[4] = dbits(__ddiv_rn(nan1, one));
    u[5] = dbits(__ddiv_rn(one, three));
    u[6] = dbits(__ddiv_rz(one, three));
    u[7] = dbits(__ddiv_ru(one, three));
    u[8] = dbits(__ddiv_rn(sub, big));
    u[9] = dbits(__ddiv_ru(sub, big));
    u[10] = dbits(__ddiv_rd(big, sub));
    u[11] = dbits(__ddiv_rz(big, sub));
    u[12] = dbits(__dsqrt_rn(m0));
    u[13] = dbits(__dsqrt_rn(neg));
    u[14] = dbits(__dsqrt_rn(two));
    u[15] = dbits(__dsqrt_ru(two));
    u[16] = dbits(__dsqrt_rd(two));
    u[17] = dbits(__dsqrt_rn(sub));
    u[18] = dbits(__dsqrt_ru(sub));
    u[19] = dbits(__dsqrt_rn(nan1));
    u[20] = dbits(__drcp_rn(three));
    u[21] = dbits(__drcp_ru(three));
    u[22] = dbits(__drcp_rn(p0));
    u[23] = dbits(__drcp_rn(sub));
    u[24] = dbits(__drcp_rz(sub));
    u[25] = dbits(__drcp_rn(nan1));
    u[26] = dbits(__dsqrt_rn(inf));
    u[27] = dbits(__ddiv_rn(sub, two));
    u[28] = dbits(__ddiv_ru(sub, two));
}

int main() {
    unsigned long long bits[11] = {
        0x0000000000000000ull, 0x8000000000000000ull, 0x7ff0000000000001ull,
        0x7ff8000000000005ull, 0x3ff0000000000000ull, 0x4008000000000000ull,
        0x7ff0000000000000ull, 0x0000000000000001ull, 0x7fefffffffffffffull,
        0x4000000000000000ull, 0xc000000000000000ull};
    unsigned long long* in = nullptr;
    unsigned long long* u = nullptr;
    CHECK(cudaMalloc(&in, sizeof(bits)));
    CHECK(cudaMalloc(&u, 29 * sizeof(unsigned long long)));
    CHECK(cudaMemcpy(in, bits, sizeof(bits), cudaMemcpyHostToDevice));
    round_ddiv<<<1, 1>>>(in, u);
    CHECK(cudaDeviceSynchronize());
    unsigned long long hu[29] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    const char* names[29] = {
        "dnan0", "dinf", "dninf", "dnan2", "dnan1q", "d3rn", "d3rz", "d3ru",
        "dunrn", "dunru", "dovrd", "dovrz", "dsm0", "dsneg", "ds2rn", "ds2ru",
        "ds2rd", "dssub", "dssubru", "dsnan", "r3rn", "r3ru", "r0", "rsub",
        "rsubrz", "rnan", "sinf", "tiern", "tieru"};
    for (int i = 0; i < 29; ++i) std::printf("%s=%llu\n", names[i], hu[i]);
    return 0;
}
