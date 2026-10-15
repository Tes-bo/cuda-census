// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK(cmd)                                                         \
    do {                                                                   \
        cudaError_t err_ = (cmd);                                          \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s\n", cudaGetErrorString(err_));        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// 1.5 rounds to 2, or to 1 toward zero. 32768 fits in 32 bits and
// saturates a short. A NaN is 0 in 32 bits and the high bit in 64 bits.
__global__ void convert(long long* o, unsigned long long* q) {
    __half h15 = __ushort_as_half(0x3e00);
    __half h05 = __ushort_as_half(0x3800);
    __half h32 = __ushort_as_half(0x7800);
    __half hbig = __ushort_as_half(0x7bff);
    __half hneg = __ushort_as_half(0xfbff);
    __half pinf = __ushort_as_half(0x7c00);
    __half ninf = __ushort_as_half(0xfc00);
    __half nan = __ushort_as_half(0x7fff);
    __half nnan = __ushort_as_half(0xffff);
    o[0] = __half2int_rn(h15);
    o[1] = __half2int_rz(h15);
    o[2] = __half2int_ru(h15);
    o[3] = __half2int_rd(h15);
    o[4] = __half2int_rn(h05);
    o[5] = __half2int_ru(h05);
    o[6] = __half2int_rn(h32);
    o[7] = __half2short_rn(h32);
    o[8] = __half2short_rn(hbig);
    o[9] = __half2ushort_rn(hbig);
    o[10] = __half2short_rn(hneg);
    o[11] = __half2uint_rn(hneg);
    o[12] = __half2int_rn(pinf);
    o[13] = __half2uint_rn(pinf);
    o[14] = __half2short_rn(pinf);
    o[15] = __half2ushort_rn(pinf);
    o[16] = __half2int_rn(ninf);
    o[17] = __half2short_rn(ninf);
    o[18] = __half2int_rn(nan);
    o[19] = __half2short_rn(nan);
    o[20] = __half2int_rn(nnan);
    q[0] = (unsigned long long)__half2ll_rn(nan);
    q[1] = __half2ull_rn(nan);
    q[2] = (unsigned long long)__half2ll_rn(nnan);
}

int main() {
    long long* o = nullptr;
    unsigned long long* q = nullptr;
    CHECK(cudaMalloc(&o, 21 * sizeof(*o)));
    CHECK(cudaMalloc(&q, 3 * sizeof(*q)));
    convert<<<1, 1>>>(o, q);
    CHECK(cudaDeviceSynchronize());
    long long ho[21] = {};
    unsigned long long hq[3] = {};
    CHECK(cudaMemcpy(ho, o, sizeof(ho), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hq, q, sizeof(hq), cudaMemcpyDeviceToHost));
    std::printf("i15rn=%lld\n", ho[0]);
    std::printf("i15rz=%lld\n", ho[1]);
    std::printf("i15ru=%lld\n", ho[2]);
    std::printf("i15rd=%lld\n", ho[3]);
    std::printf("i05rn=%lld\n", ho[4]);
    std::printf("i05ru=%lld\n", ho[5]);
    std::printf("i32=%lld\n", ho[6]);
    std::printf("s32=%lld\n", ho[7]);
    std::printf("sbig=%lld\n", ho[8]);
    std::printf("usbig=%lld\n", ho[9]);
    std::printf("sneg=%lld\n", ho[10]);
    std::printf("uneg=%lld\n", ho[11]);
    std::printf("iinf=%lld\n", ho[12]);
    std::printf("uinf=%lld\n", ho[13]);
    std::printf("sinf=%lld\n", ho[14]);
    std::printf("usinf=%lld\n", ho[15]);
    std::printf("ninf=%lld\n", ho[16]);
    std::printf("sninf=%lld\n", ho[17]);
    std::printf("nan=%lld\n", ho[18]);
    std::printf("snan=%lld\n", ho[19]);
    std::printf("nnan=%lld\n", ho[20]);
    std::printf("llnan=%llu\n", hq[0]);
    std::printf("ullnan=%llu\n", hq[1]);
    std::printf("llnnan=%llu\n", hq[2]);
    cudaFree(o);
    cudaFree(q);
    return 0;
}
