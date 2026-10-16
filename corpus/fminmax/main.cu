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

// fmin, fmax, nextafter, and copysign. Bits come from host memory so nvcc
// cannot fold a NaN. Float NaNs become one canonical quiet NaN. Two double
// NaNs keep the second operand, quieted. copysign keeps the payload.
__global__ void edges(const unsigned* in, const unsigned long long* din, unsigned* u,
                      unsigned long long* d) {
    float p0 = fv(in[0]);
    float m0 = fv(in[1]);
    float one = fv(in[2]);
    float pinf = fv(in[3]);
    float ninf = fv(in[4]);
    float nan1 = fv(in[5]);
    float nan2 = fv(in[6]);
    float big = fv(in[7]);
    float normal = fv(in[8]);
    float sig = fv(in[9]);
    u[0] = __float_as_uint(fminf(p0, m0));
    u[1] = __float_as_uint(fmaxf(p0, m0));
    u[2] = __float_as_uint(fminf(p0, p0));
    u[3] = __float_as_uint(fminf(m0, m0));
    u[4] = __float_as_uint(fmaxf(m0, m0));
    u[5] = __float_as_uint(fminf(nan1, nan2));
    u[6] = __float_as_uint(fmaxf(nan1, nan2));
    u[7] = __float_as_uint(fminf(nan1, one));
    u[8] = __float_as_uint(fminf(pinf, ninf));
    u[9] = __float_as_uint(fmaxf(pinf, ninf));
    u[10] = __float_as_uint(nextafterf(p0, pinf));
    u[11] = __float_as_uint(nextafterf(p0, m0));
    u[12] = __float_as_uint(nextafterf(m0, p0));
    u[13] = __float_as_uint(nextafterf(pinf, one));
    u[14] = __float_as_uint(nextafterf(normal, p0));
    u[15] = __float_as_uint(nextafterf(nan1, one));
    u[16] = __float_as_uint(nextafterf(big, pinf));
    u[17] = __float_as_uint(copysignf(nan1, m0));
    u[18] = __float_as_uint(copysignf(sig, m0));

    double dp0 = dv(din[0]);
    double dm0 = dv(din[1]);
    double done = dv(din[2]);
    double dpinf = dv(din[3]);
    double dnan1 = dv(din[4]);
    double dnan5 = dv(din[5]);
    double dnormal = dv(din[6]);
    d[0] = dbits(fmin(dp0, dm0));
    d[1] = dbits(fmax(dp0, dm0));
    d[2] = dbits(fmin(dnan1, done));
    d[3] = dbits(fmin(dnan1, dnan5));
    d[4] = dbits(fmax(dnan1, dnan5));
    d[5] = dbits(nextafter(dnan1, done));
    d[6] = dbits(copysign(dnan1, dm0));
    d[7] = dbits(nextafter(dp0, dpinf));
    d[8] = dbits(nextafter(dnormal, dp0));
}

int main() {
    unsigned bits[10] = {0x00000000u, 0x80000000u, 0x3f800000u, 0x7f800000u, 0xff800000u,
                         0x7fc00001u, 0xffc00005u, 0x7f7fffffu, 0x00800000u, 0x7f800001u};
    unsigned long long dbits_in[7] = {0x0000000000000000ull, 0x8000000000000000ull,
                                      0x3ff0000000000000ull, 0x7ff0000000000000ull,
                                      0x7ff8000000000001ull, 0x7ff8000000000005ull,
                                      0x0010000000000000ull};
    unsigned* in = nullptr;
    unsigned long long* din = nullptr;
    unsigned* u = nullptr;
    unsigned long long* d = nullptr;
    CHECK(cudaMalloc(&in, sizeof(bits)));
    CHECK(cudaMalloc(&din, sizeof(dbits_in)));
    CHECK(cudaMalloc(&u, 19 * sizeof(unsigned)));
    CHECK(cudaMalloc(&d, 9 * sizeof(unsigned long long)));
    CHECK(cudaMemcpy(in, bits, sizeof(bits), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(din, dbits_in, sizeof(dbits_in), cudaMemcpyHostToDevice));
    edges<<<1, 1>>>(in, din, u, d);
    CHECK(cudaDeviceSynchronize());
    unsigned hu[19] = {};
    unsigned long long hd[9] = {};
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hd, d, sizeof(hd), cudaMemcpyDeviceToHost));
    std::printf("minzm=%u\n", hu[0]);
    std::printf("maxzm=%u\n", hu[1]);
    std::printf("minpp=%u\n", hu[2]);
    std::printf("minmm=%u\n", hu[3]);
    std::printf("maxmm=%u\n", hu[4]);
    std::printf("minnan=%u\n", hu[5]);
    std::printf("maxnan=%u\n", hu[6]);
    std::printf("minnum=%u\n", hu[7]);
    std::printf("mininf=%u\n", hu[8]);
    std::printf("maxinf=%u\n", hu[9]);
    std::printf("nxt0=%u\n", hu[10]);
    std::printf("nxtzm=%u\n", hu[11]);
    std::printf("nxtmz=%u\n", hu[12]);
    std::printf("nxtinf=%u\n", hu[13]);
    std::printf("nxtsub=%u\n", hu[14]);
    std::printf("nxtnan=%u\n", hu[15]);
    std::printf("nxtmax=%u\n", hu[16]);
    std::printf("copynan=%u\n", hu[17]);
    std::printf("copysig=%u\n", hu[18]);
    std::printf("dminzm=%llu\n", hd[0]);
    std::printf("dmaxzm=%llu\n", hd[1]);
    std::printf("dminnum=%llu\n", hd[2]);
    std::printf("dminnan=%llu\n", hd[3]);
    std::printf("dmaxnan=%llu\n", hd[4]);
    std::printf("dnxtone=%llu\n", hd[5]);
    std::printf("dcopynan=%llu\n", hd[6]);
    std::printf("dnxt0=%llu\n", hd[7]);
    std::printf("dnxtsub=%llu\n", hd[8]);
    return 0;
}
