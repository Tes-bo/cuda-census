// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK(cmd)                                                         \
    do {                                                                   \
        cudaError_t err_ = (cmd);                                          \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s\n", cudaGetErrorString(err_));        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// One warp multiplies two 16x16 half matrices into a float accumulator that
// starts at 1. Every element is a small integer, so every product and sum is
// exact and the result does not depend on the order of accumulation.
__global__ void tile(const __half* a, const __half* b, float* c, float* ct) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 1.0f);
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c, fc, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(ct, fc, 16, wmma::mem_col_major);
}

int main() {
    __half ha[256];
    __half hb[256];
    for (int i = 0; i < 16; ++i) {
        for (int k = 0; k < 16; ++k) {
            ha[i * 16 + k] = __float2half(static_cast<float>((i + 2 * k) % 7 - 3));
        }
    }
    for (int j = 0; j < 16; ++j) {
        for (int k = 0; k < 16; ++k) {
            hb[j * 16 + k] = __float2half(static_cast<float>((3 * k + j) % 5 - 2));  // column j, row k
        }
    }
    __half* a = nullptr;
    __half* b = nullptr;
    float* c = nullptr;
    float* ct = nullptr;
    CHECK(cudaMalloc(&a, sizeof(ha)));
    CHECK(cudaMalloc(&b, sizeof(hb)));
    CHECK(cudaMalloc(&c, 256 * sizeof(float)));
    CHECK(cudaMalloc(&ct, 256 * sizeof(float)));
    CHECK(cudaMemcpy(a, ha, sizeof(ha), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b, hb, sizeof(hb), cudaMemcpyHostToDevice));
    tile<<<1, 32>>>(a, b, c, ct);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float hc[256] = {};
    float hct[256] = {};
    CHECK(cudaMemcpy(hc, c, sizeof(hc), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hct, ct, sizeof(hct), cudaMemcpyDeviceToHost));
    long long weighted = 0;
    long long trace = 0;
    int transposed = 0;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            long long v = static_cast<long long>(hc[i * 16 + j]);
            weighted += v * (i * 16 + j + 1);
            transposed += hct[j * 16 + i] == hc[i * 16 + j] ? 1 : 0;
        }
        trace += static_cast<long long>(hc[i * 17]);
    }
    std::printf("c00=%lld\n", static_cast<long long>(hc[0]));
    std::printf("c0f=%lld\n", static_cast<long long>(hc[15]));
    std::printf("cf0=%lld\n", static_cast<long long>(hc[240]));
    std::printf("cff=%lld\n", static_cast<long long>(hc[255]));
    std::printf("trace=%lld\n", trace);
    std::printf("weighted=%lld\n", weighted);
    std::printf("colmajor=%d\n", transposed);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(ct);
    return 0;
}
