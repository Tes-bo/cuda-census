// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(cmd)                                                         \
    do {                                                                   \
        cudaError_t err_ = (cmd);                                          \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s\n", cudaGetErrorString(err_));        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// 256 threads in two blocks hit the same global words. Every printed value
// is independent of the order in which the threads arrive.
__global__ void contend(int* s, unsigned* u, unsigned long long* w, float* f, double* d) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int old = atomicAdd(&s[0], 1);
    atomicAdd(&s[4], old);                          // the old values are 0..255 in some order
    atomicSub(&s[1], 3);
    atomicMax(&s[2], t - 300);
    atomicMin(&s[3], t - 300);
    atomicMax(&u[0], static_cast<unsigned>(t - 300)); // unsigned: -300 is near 2^32
    atomicMin(&u[1], 0x80000000u + t);              // unsigned: above the initial 5
    atomicInc(&u[2], 9u);                           // counts 0..9, then wraps to 0
    atomicDec(&u[3], 9u);                           // 0 wraps to 9, then counts down
    atomicOr(&u[4], 1u << (t & 31));
    atomicAnd(&u[5], ~(1u << (t % 31)));
    atomicXor(&u[6], 1u << (t % 9));                // bits 0..3 flip 29 times, bits 4..8 flip 28
    unsigned prev = atomicExch(&u[7], 7u);
    atomicAdd(&u[8], prev == 7u ? 1u : 0u);         // every thread but the first sees 7
    unsigned seen = atomicCAS(&u[9], 0u, static_cast<unsigned>(t + 1));
    atomicAdd(&u[10], seen == 0u ? 1u : 0u);        // exactly one compare-and-swap wins
    atomicAdd(&w[0], 1ull);                         // wraps past 2^64
    atomicMax(&w[1], static_cast<unsigned long long>(t) << 40);
    atomicMin(&w[2], (1ull << 63) | static_cast<unsigned long long>(t));
    atomicAdd(&f[0], 0.5f);                         // every partial sum is exact
    atomicAdd(&d[0], 0.25);
}

int main() {
    int hs[5] = {0, 0, -1000, 1000, 0};
    unsigned hu[11] = {0u, 5u, 0u, 0u, 0u, 0xffffffffu, 0x5au, 0u, 0u, 0u, 0u};
    unsigned long long hw[3] = {0xfffffffffffffff0ull, 0ull, ~0ull};
    float hf = 0.0f;
    double hd = 0.0;
    int* s = nullptr;
    unsigned* u = nullptr;
    unsigned long long* w = nullptr;
    float* f = nullptr;
    double* d = nullptr;
    CHECK(cudaMalloc(&s, sizeof(hs)));
    CHECK(cudaMalloc(&u, sizeof(hu)));
    CHECK(cudaMalloc(&w, sizeof(hw)));
    CHECK(cudaMalloc(&f, sizeof(hf)));
    CHECK(cudaMalloc(&d, sizeof(hd)));
    CHECK(cudaMemcpy(s, hs, sizeof(hs), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(u, hu, sizeof(hu), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(w, hw, sizeof(hw), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(f, &hf, sizeof(hf), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d, &hd, sizeof(hd), cudaMemcpyHostToDevice));
    contend<<<2, 128>>>(s, u, w, f, d);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hs, s, sizeof(hs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hu, u, sizeof(hu), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hw, w, sizeof(hw), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hf, f, sizeof(hf), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost));
    unsigned fbits = 0;
    unsigned long long dbits = 0;
    std::memcpy(&fbits, &hf, sizeof(fbits));
    std::memcpy(&dbits, &hd, sizeof(dbits));
    std::printf("count=%d\n", hs[0]);
    std::printf("oldsum=%d\n", hs[4]);
    std::printf("sub=%d\n", hs[1]);
    std::printf("smax=%d\n", hs[2]);
    std::printf("smin=%d\n", hs[3]);
    std::printf("umax=%u\n", hu[0]);
    std::printf("umin=%u\n", hu[1]);
    std::printf("inc=%u\n", hu[2]);
    std::printf("dec=%u\n", hu[3]);
    std::printf("or=%u\n", hu[4]);
    std::printf("and=%u\n", hu[5]);
    std::printf("xor=%u\n", hu[6]);
    std::printf("exch=%u\n", hu[7]);
    std::printf("exchold=%u\n", hu[8]);
    std::printf("casset=%u\n", hu[9] >= 1u && hu[9] <= 256u ? 1u : 0u);
    std::printf("caswins=%u\n", hu[10]);
    std::printf("u64wrap=%llu\n", hw[0]);
    std::printf("u64max=%llu\n", hw[1]);
    std::printf("u64min=%llu\n", hw[2]);
    std::printf("fadd=%u\n", fbits);
    std::printf("dadd=%llu\n", dbits);
    cudaFree(s);
    cudaFree(u);
    cudaFree(w);
    cudaFree(f);
    cudaFree(d);
    return 0;
}
