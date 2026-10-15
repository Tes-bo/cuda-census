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

// Directed float add, sub, mul, and fma. Plain + and * round ties to even.
// A tiny addend below one ulp still moves the round-up result. Overflow
// toward zero stays at the finite maximum. Opposite-sign zeros are -0 only
// when rounding toward -inf. A NaN operand becomes one canonical quiet NaN.
__global__ void round_float(unsigned* u) {
    union {
        float as_float;
        unsigned as_bits;
    } tiny, tie, big, nan, wide, bits;
    tiny.as_bits = 0x21800000u;
    tie.as_bits = 0x33800000u;
    big.as_bits = 0x7f000000u;
    nan.as_bits = 0x7fc00000u;
    wide.as_bits = 0x3f800001u;
    bits.as_float = __fadd_rn(1.0f, tiny.as_float);
    u[0] = bits.as_bits;
    bits.as_float = __fadd_ru(1.0f, tiny.as_float);
    u[1] = bits.as_bits;
    bits.as_float = __fadd_rd(-1.0f, -tiny.as_float);
    u[2] = bits.as_bits;
    bits.as_float = __fadd_rn(1.0f, tie.as_float);
    u[3] = bits.as_bits;
    bits.as_float = __fadd_ru(1.0f, tie.as_float);
    u[4] = bits.as_bits;
    bits.as_float = __fadd_rn(big.as_float, big.as_float);
    u[5] = bits.as_bits;
    bits.as_float = __fadd_rz(big.as_float, big.as_float);
    u[6] = bits.as_bits;
    bits.as_float = __fadd_rd(0.0f, -0.0f);
    u[7] = bits.as_bits;
    bits.as_float = __fadd_rd(0.0f, 0.0f);
    u[8] = bits.as_bits;
    bits.as_float = __fadd_rn(nan.as_float, 1.0f);
    u[9] = bits.as_bits;
    bits.as_float = __fsub_rd(1.0f, tiny.as_float);
    u[10] = bits.as_bits;
    bits.as_float = __fmul_rz(big.as_float, big.as_float);
    u[11] = bits.as_bits;
    bits.as_float = __fmul_rd(big.as_float, -big.as_float);
    u[12] = bits.as_bits;
    bits.as_float = __fmul_rn(-0.0f, 0.0f);
    u[13] = bits.as_bits;
    bits.as_float = __fmaf_ru(1.0f, tiny.as_float, 1.0f);
    u[14] = bits.as_bits;
    bits.as_float = __fmaf_rn(wide.as_float, wide.as_float, __uint_as_float(0xbf800002u));
    u[15] = bits.as_bits;
    bits.as_float = __fmaf_rn(0.0f, __uint_as_float(0x7f800000u), 1.0f);
    u[16] = bits.as_bits;
}

int main() {
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&u, 17 * sizeof(unsigned)));
    round_float<<<1, 1>>>(u);
    CHECK(cudaDeviceSynchronize());
    unsigned hu[17] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    std::printf("addtiny=%u\n", hu[0]);
    std::printf("addru=%u\n", hu[1]);
    std::printf("addrdn=%u\n", hu[2]);
    std::printf("addtie=%u\n", hu[3]);
    std::printf("addtieru=%u\n", hu[4]);
    std::printf("addinf=%u\n", hu[5]);
    std::printf("addmax=%u\n", hu[6]);
    std::printf("zrd=%u\n", hu[7]);
    std::printf("zsame=%u\n", hu[8]);
    std::printf("nan=%u\n", hu[9]);
    std::printf("subrd=%u\n", hu[10]);
    std::printf("mulrz=%u\n", hu[11]);
    std::printf("mulrd=%u\n", hu[12]);
    std::printf("mzero=%u\n", hu[13]);
    std::printf("fmaru=%u\n", hu[14]);
    std::printf("fmacancel=%u\n", hu[15]);
    std::printf("fmanan=%u\n", hu[16]);
    cudaFree(u);
    return 0;
}
