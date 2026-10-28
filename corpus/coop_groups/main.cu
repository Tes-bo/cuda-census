// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define CHECK(cmd)                                                         \
    do {                                                                   \
        cudaError_t err_ = (cmd);                                          \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s\n", cudaGetErrorString(err_));        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// 64 threads: two 32-thread tiles and four 16-thread tiles. A tile that
// followed a 64-lane wavefront would sum across tile boundaries.
__global__ void groups(int* out) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    cg::thread_block_tile<16> half = cg::tiled_partition<16>(block);
    int t = static_cast<int>(block.thread_rank());

    int v = t;
    for (int offset = 8; offset > 0; offset /= 2) {
        v += half.shfl_down(v, offset);
    }
    if (half.thread_rank() == 0) {
        out[t / 16] = v;
    }
    int wsum = cg::reduce(warp, t, cg::plus<int>());
    if (warp.thread_rank() == 0) {
        out[4 + t / 32] = wsum;
    }
    unsigned ballot = warp.ballot(t % 2 == 0);
    if (t == 32) {
        out[6] = static_cast<int>(ballot);
    }
    if (t == 48) {
        out[7] = static_cast<int>(warp.size());
        out[8] = static_cast<int>(half.size());
        out[9] = static_cast<int>(block.size());
        out[10] = static_cast<int>(half.meta_group_rank());
        out[11] = static_cast<int>(half.meta_group_size());
    }
    // Lanes 0, 3, ..., 30 of the first warp take the branch together.
    if (t < 32 && t % 3 == 0) {
        cg::coalesced_group active = cg::coalesced_threads();
        int csum = cg::reduce(active, t, cg::plus<int>());
        if (t == 30) {
            out[12] = static_cast<int>(active.size());
            out[13] = static_cast<int>(active.thread_rank());
            out[14] = csum;
        }
    }
    block.sync();
}

int main() {
    int* out = nullptr;
    CHECK(cudaMalloc(&out, 15 * sizeof(int)));
    CHECK(cudaMemset(out, 0, 15 * sizeof(int)));
    groups<<<1, 64>>>(out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int h[15] = {};
    CHECK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
    std::printf("tile0=%d\n", h[0]);
    std::printf("tile1=%d\n", h[1]);
    std::printf("tile2=%d\n", h[2]);
    std::printf("tile3=%d\n", h[3]);
    std::printf("warp0=%d\n", h[4]);
    std::printf("warp1=%d\n", h[5]);
    std::printf("ballot=%u\n", static_cast<unsigned>(h[6]));
    std::printf("wsize=%d\n", h[7]);
    std::printf("hsize=%d\n", h[8]);
    std::printf("bsize=%d\n", h[9]);
    std::printf("hrank=%d\n", h[10]);
    std::printf("hcount=%d\n", h[11]);
    std::printf("csize=%d\n", h[12]);
    std::printf("crank=%d\n", h[13]);
    std::printf("csum=%d\n", h[14]);
    cudaFree(out);
    return 0;
}
