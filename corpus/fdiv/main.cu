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

__device__ float fv(unsigned bits) {
    volatile unsigned held = bits;
    return __uint_as_float(held);
}

// Directed float division, reciprocal, square root, and reciprocal square
// root. 0/0 is a canonical NaN. sqrt(-0) stays -0. Reciprocal square root of
// a tiny normal is not the reciprocal of the rounded square root.
__global__ void round_div(const unsigned* in, unsigned* u) {
    float p0 = fv(in[0]);
    float m0 = fv(in[1]);
    float one = fv(in[2]);
    float inf = fv(in[3]);
    float two = fv(in[4]);
    float three = fv(in[5]);
    float sub = fv(in[6]);
    float big = fv(in[7]);
    float gap = fv(in[8]);
    float neg = fv(in[9]);
    u[0] = __float_as_uint(__fdiv_rn(p0, m0));
    u[1] = __float_as_uint(__fdiv_rn(one, p0));
    u[2] = __float_as_uint(__fdiv_rn(one, m0));
    u[3] = __float_as_uint(__fdiv_rn(m0, one));
    u[4] = __float_as_uint(__fdiv_rn(one, inf));
    u[5] = __float_as_uint(__fsqrt_rn(m0));
    u[6] = __float_as_uint(__fsqrt_rn(neg));
    u[7] = __float_as_uint(__fsqrt_rn(two));
    u[8] = __float_as_uint(__fsqrt_ru(two));
    u[9] = __float_as_uint(__fdiv_rn(one, three));
    u[10] = __float_as_uint(__fdiv_rz(one, three));
    u[11] = __float_as_uint(__fdiv_rn(sub, big));
    u[12] = __float_as_uint(__fdiv_ru(sub, big));
    u[13] = __float_as_uint(__fdiv_rd(big, sub));
    u[14] = __float_as_uint(__frsqrt_rn(gap));
    u[15] = __float_as_uint(__frcp_rn(__fsqrt_rn(gap)));
    u[16] = __float_as_uint(__frsqrt_rn(p0));
    u[17] = __float_as_uint(__frsqrt_rn(m0));
    u[18] = __float_as_uint(__fsqrt_rn(sub));
    u[19] = __float_as_uint(__fsqrt_ru(sub));
}

int main() {
    unsigned bits[10] = {0x00000000u, 0x80000000u, 0x3f800000u, 0x7f800000u, 0x40000000u,
                         0x40400000u, 0x00000001u, 0x7f7fffffu, 0x187fffffu, 0xc0000000u};
    unsigned* in = nullptr;
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&in, sizeof(bits)));
    CHECK(cudaMalloc(&u, 20 * sizeof(unsigned)));
    CHECK(cudaMemcpy(in, bits, sizeof(bits), cudaMemcpyHostToDevice));
    round_div<<<1, 1>>>(in, u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned hu[20] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("divnan=%u\n", hu[0]);
    std::printf("divinf=%u\n", hu[1]);
    std::printf("divninf=%u\n", hu[2]);
    std::printf("mzero=%u\n", hu[3]);
    std::printf("zeroinf=%u\n", hu[4]);
    std::printf("sqrtm0=%u\n", hu[5]);
    std::printf("sqrtneg=%u\n", hu[6]);
    std::printf("sq2=%u\n", hu[7]);
    std::printf("sq2ru=%u\n", hu[8]);
    std::printf("d3=%u\n", hu[9]);
    std::printf("d3rz=%u\n", hu[10]);
    std::printf("underrn=%u\n", hu[11]);
    std::printf("underru=%u\n", hu[12]);
    std::printf("ovrd=%u\n", hu[13]);
    std::printf("rsqrt=%u\n", hu[14]);
    std::printf("split=%u\n", hu[15]);
    std::printf("rs0=%u\n", hu[16]);
    std::printf("rsm0=%u\n", hu[17]);
    std::printf("sqsub=%u\n", hu[18]);
    std::printf("sqsubru=%u\n", hu[19]);
    return 0;
}
