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

__device__ unsigned ld(const unsigned* p) {
    volatile unsigned held = *p;
    return held;
}

// One integer PTX instruction each. Inputs come from host memory so nvcc
// cannot fold them, and the logic-op immediate stays in the instruction.
__global__ void ptx_ops(const unsigned* in, const unsigned* ctl, unsigned* u) {
    unsigned a = ld(in + 0);
    unsigned b = ld(in + 1);
    unsigned c = ld(in + 2);
    unsigned ones = ld(in + 3);
    unsigned one = ld(in + 4);
    unsigned two = ld(in + 5);
    unsigned four = ld(in + 6);
    unsigned word = ld(in + 7);
    unsigned src = ld(in + 8);
    unsigned base = ld(in + 9);
    unsigned high = ld(in + 10);
    unsigned pa = ld(in + 11);
    unsigned pb = ld(in + 12);
    unsigned pc = ld(in + 13);
    unsigned base2 = ld(in + 14);
    unsigned zero = ld(in + 15);
    unsigned p0 = ld(ctl + 0);
    unsigned p1 = ld(ctl + 1);
    unsigned p4 = ld(ctl + 2);
    unsigned p8 = ld(ctl + 3);
    unsigned p24 = ld(ctl + 4);
    unsigned p28 = ld(ctl + 5);
    unsigned p32 = ld(ctl + 6);
    unsigned p101 = ld(ctl + 7);
    unsigned perm = ld(ctl + 8);
    unsigned sign = ld(ctl + 9);
    unsigned highsel = ld(ctl + 10);
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(u[0]) : "r"(a), "r"(b), "r"(c));
    asm("lop3.b32 %0, %1, %2, %3, 0xF0;" : "=r"(u[1]) : "r"(ones), "r"(b), "r"(c));
    asm("lop3.b32 %0, %1, %2, %3, 0xCC;" : "=r"(u[2]) : "r"(a), "r"(ones), "r"(c));
    asm("lop3.b32 %0, %1, %2, %3, 0xAA;" : "=r"(u[3]) : "r"(a), "r"(b), "r"(ones));
    asm("lop3.b32 %0, %1, %2, %3, 0x80;" : "=r"(u[4]) : "r"(one), "r"(two), "r"(four));
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u[5]) : "r"(word), "r"(p4), "r"(p8));
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u[6]) : "r"(word), "r"(p28), "r"(p8));
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u[7]) : "r"(word), "r"(p0), "r"(p0));
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u[8]) : "r"(ones), "r"(p32), "r"(p1));
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(u[9]) : "r"(src), "r"(p0), "r"(p101));
    asm("bfe.s32 %0, %1, %2, %3;" : "=r"(u[10]) : "r"(word), "r"(p24), "r"(p8));
    asm("bfe.s32 %0, %1, %2, %3;" : "=r"(u[11]) : "r"(one), "r"(p0), "r"(p1));
    asm("bfe.s32 %0, %1, %2, %3;" : "=r"(u[12]) : "r"(high), "r"(p32), "r"(p1));
    asm("bfe.s32 %0, %1, %2, %3;" : "=r"(u[13]) : "r"(high), "r"(p32), "r"(p0));
    asm("bfe.s32 %0, %1, %2, %3;" : "=r"(u[14]) : "r"(zero), "r"(p32), "r"(p1));
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(u[15]) : "r"(src), "r"(base), "r"(p8), "r"(p8));
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(u[16]) : "r"(src), "r"(base), "r"(p28), "r"(p8));
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(u[17]) : "r"(src), "r"(base2), "r"(p32), "r"(p4));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u[18]) : "r"(pa), "r"(pb), "r"(perm));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u[19]) : "r"(pa), "r"(pc), "r"(sign));
    u[20] = __byte_perm(pa, pc, sign);
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u[21]) : "r"(pa), "r"(pc), "r"(highsel));
}

int main() {
    unsigned bits[16] = {0x12345678u, 0x9abcdef0u, 0x0f0f0f0fu, 0xffffffffu, 1u, 2u, 4u,
                         0x89abcdefu, 0xffu,        0x11111111u, 0x80000000u, 0x04030201u,
                         0x08070605u, 0x80ff7f01u,  0x12345678u, 0u};
    unsigned ctrl[11] = {0u, 1u, 4u, 8u, 24u, 28u, 32u, 0x101u, 0x76543210u, 0x8u, 0x10000u};
    unsigned* in = nullptr;
    unsigned* ctl = nullptr;
    unsigned* u = nullptr;
    CHECK(cudaMalloc(&in, sizeof(bits)));
    CHECK(cudaMalloc(&ctl, sizeof(ctrl)));
    CHECK(cudaMalloc(&u, 22 * sizeof(unsigned)));
    CHECK(cudaMemcpy(in, bits, sizeof(bits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ctl, ctrl, sizeof(ctrl), cudaMemcpyHostToDevice));
    ptx_ops<<<1, 1>>>(in, ctl, u);
    CHECK(cudaDeviceSynchronize());
    unsigned hu[22] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    const char* names[22] = {"lop",  "lopa", "lopb", "lopc", "lopz", "u4",   "u28",
                             "uz",   "upos", "ulen", "s24",  "sbit", "spos", "sz",
                             "s0",   "ins",  "ins28","insp", "perm", "psign","pbyte",
                             "phigh"};
    for (int i = 0; i < 22; ++i) std::printf("%s=%u\n", names[i], hu[i]);
    return 0;
}
