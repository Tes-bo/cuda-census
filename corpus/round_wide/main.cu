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

// A NaN widened to 64 bits is the high bit. The same NaN through
// __float2int_rn is 0, and __double2int_rz yields 0x80000000.
// (2^53)+1 is not an exact float. ru is one ulp above rn.
// A double past 2^64 saturates an unsigned 64-bit conversion.
__global__ void convert(long long* s, unsigned long long* u, unsigned* bits) {
    union {
        float as_float;
        unsigned as_bits;
    } nan, wide;
    nan.as_bits = 0x7fc00000u;
    s[0] = __float2ll_rn(nan.as_float);
    s[1] = __float2ll_rn(1.0e20f);
    u[0] = __float2ull_rn(nan.as_float);
    wide.as_float = __ll2float_rn((1LL << 53) + 1);
    bits[0] = wide.as_bits;
    wide.as_float = __ll2float_ru((1LL << 53) + 1);
    bits[1] = wide.as_bits;
    wide.as_float = __ull2float_rd(0xffffffffffffffffull);
    bits[2] = wide.as_bits;
    union {
        double as_double;
        unsigned long long as_bits;
    } dnan;
    dnan.as_bits = 0x7ff8000000000000ull;
    s[2] = __double2int_rz(dnan.as_double);
    s[3] = __double2int_rz(1.9);
    u[1] = __double2uint_rz(dnan.as_double);
    u[2] = __double2ull_rz(1.0e20);
}

int main() {
    long long* s = nullptr;
    unsigned long long* u = nullptr;
    unsigned* bits = nullptr;
    CHECK(cudaMalloc(&s, 4 * sizeof(long long)));
    CHECK(cudaMalloc(&u, 3 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&bits, 3 * sizeof(unsigned)));
    convert<<<1, 1>>>(s, u, bits);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    long long hs[4] = {};
    unsigned long long hu[3] = {};
    unsigned hb[3] = {};
    CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hb, bits, sizeof(hb), cudaMemcpyDeviceToHost));
    std::printf("llnan=%lld\n", hs[0]);
    std::printf("llbig=%lld\n", hs[1]);
    std::printf("ullnan=%llu\n", hu[0]);
    std::printf("llrn=%u\n", hb[0]);
    std::printf("llru=%u\n", hb[1]);
    std::printf("ullrd=%u\n", hb[2]);
    std::printf("dnan=%lld\n", hs[2]);
    std::printf("dtrunc=%lld\n", hs[3]);
    std::printf("dun=%llu\n", hu[1]);
    std::printf("dull=%llu\n", hu[2]);
    cudaFree(s);
    cudaFree(u);
    cudaFree(bits);
    return 0;
}
