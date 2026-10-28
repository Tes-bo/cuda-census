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

// Directed double add, sub, mul, and fma. The second NaN operand wins, and
// a signaling NaN is quieted. inf*0 is one negative quiet NaN. Fused
// multiply-add of (1+2^-52) squared keeps the 2^-104 the rounded product loses.
__global__ void round_double(const unsigned long long* in, unsigned long long* u) {
    double p0 = dv(in[0]);
    double m0 = dv(in[1]);
    double nan1 = dv(in[2]);
    double nan2 = dv(in[3]);
    double nnan = dv(in[4]);
    double big = dv(in[5]);
    double nxt = dv(in[6]);
    double tie = dv(in[7]);
    double inf = dv(in[8]);
    double one = dv(in[9]);
    u[0] = dbits(__dadd_rn(p0, m0));
    u[1] = dbits(__dadd_rd(p0, m0));
    u[2] = dbits(__dadd_rd(p0, p0));
    u[3] = dbits(__dadd_rn(big, big));
    u[4] = dbits(__dadd_rz(big, big));
    u[5] = dbits(__dadd_rd(-big, -big));
    u[6] = dbits(__dadd_rz(-big, -big));
    u[7] = dbits(__dadd_rn(nan1, nan2));
    u[8] = dbits(__dsub_rn(one, nnan));
    u[9] = dbits(__dmul_rn(inf, p0));
    u[10] = dbits(__dmul_rn(m0, m0));
    u[11] = dbits(__dadd_rn(one, tie));
    u[12] = dbits(__dadd_ru(one, tie));
    volatile unsigned long long prod = dbits(__dmul_rn(nxt, nxt));
    u[13] = prod;
    u[14] = dbits(__dadd_rn(dv(prod), -dv(prod)));
    u[15] = dbits(__fma_rn(nxt, nxt, -dv(prod)));
    u[16] = dbits(__fma_rz(big, big, one));
    u[17] = dbits(__dsub_rn(m0, p0));
    u[18] = dbits(__dsub_rd(one, tie));
}

int main() {
    unsigned long long bits[10] = {
        0x0000000000000000ull, 0x8000000000000000ull, 0x7ff0000000000001ull,
        0x7ff8000000000002ull, 0xfff8000000000005ull, 0x7fefffffffffffffull,
        0x3ff0000000000001ull, 0x3ca0000000000000ull, 0x7ff0000000000000ull,
        0x3ff0000000000000ull};
    unsigned long long* in = nullptr;
    unsigned long long* u = nullptr;
    CHECK(cudaMalloc(&in, sizeof(bits)));
    CHECK(cudaMalloc(&u, 19 * sizeof(unsigned long long)));
    CHECK(cudaMemcpy(in, bits, sizeof(bits), cudaMemcpyHostToDevice));
    round_double<<<1, 1>>>(in, u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned long long hu[19] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("zrn=%llu\n", hu[0]);
    std::printf("zrd=%llu\n", hu[1]);
    std::printf("zsame=%llu\n", hu[2]);
    std::printf("addinf=%llu\n", hu[3]);
    std::printf("addmax=%llu\n", hu[4]);
    std::printf("nrd=%llu\n", hu[5]);
    std::printf("nrz=%llu\n", hu[6]);
    std::printf("nan2=%llu\n", hu[7]);
    std::printf("nansub=%llu\n", hu[8]);
    std::printf("mulnan=%llu\n", hu[9]);
    std::printf("mulmm=%llu\n", hu[10]);
    std::printf("tiern=%llu\n", hu[11]);
    std::printf("tieru=%llu\n", hu[12]);
    std::printf("prod=%llu\n", hu[13]);
    std::printf("cancel=%llu\n", hu[14]);
    std::printf("residual=%llu\n", hu[15]);
    std::printf("fmamax=%llu\n", hu[16]);
    std::printf("subm0=%llu\n", hu[17]);
    std::printf("subtie=%llu\n", hu[18]);
    return 0;
}
