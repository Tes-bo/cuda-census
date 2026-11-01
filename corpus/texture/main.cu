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

constexpr int kTexels = 8;
constexpr int kSide = 4;

__device__ unsigned fb(float x) {
    union { float as_float; unsigned as_bits; } v;
    v.as_float = x;
    return v.as_bits;
}

// Eight float texels 10, 20, ..., 80. Every linear sample uses a weight of
// 0, 1/4, 1/2, or 3/4, which the filter hardware represents exactly, so the
// expected values are exact.
__global__ void sample(cudaTextureObject_t point, cudaTextureObject_t border, cudaTextureObject_t linear,
                       cudaTextureObject_t wrap, cudaTextureObject_t mirror, cudaTextureObject_t fetch,
                       unsigned* f, int* i) {
    f[0] = fb(tex1D<float>(point, 2.5f));
    f[1] = fb(tex1D<float>(point, 2.99f));
    f[2] = fb(tex1D<float>(point, -3.0f));
    f[3] = fb(tex1D<float>(point, 100.0f));
    f[4] = fb(tex1D<float>(border, -1.0f));
    f[5] = fb(tex1D<float>(border, 8.5f));
    f[6] = fb(tex1D<float>(border, 7.5f));
    f[7] = fb(tex1D<float>(linear, 2.75f));
    f[8] = fb(tex1D<float>(linear, 1.0f));
    f[9] = fb(tex1D<float>(linear, 0.25f));
    f[10] = fb(tex1D<float>(linear, 8.25f));
    f[11] = fb(tex1D<float>(wrap, 1.3125f));
    f[12] = fb(tex1D<float>(wrap, -0.0625f));
    f[13] = fb(tex1D<float>(mirror, 1.3125f));
    f[14] = fb(tex1D<float>(mirror, -0.3125f));
    i[0] = tex1Dfetch<int>(fetch, 0);
    i[1] = tex1Dfetch<int>(fetch, 5);
}

// A surface written by one kernel and read by the next. Within one kernel a
// surface read after a surface write is not defined.
__global__ void paint(cudaSurfaceObject_t s) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    surf2Dwrite(y * 10 + x, s, x * static_cast<int>(sizeof(int)), y);
}

__global__ void probe(cudaSurfaceObject_t s, int* out) {
    surf2Dread(&out[0], s, 3 * static_cast<int>(sizeof(int)), 2);
    surf2Dread(&out[1], s, 7 * static_cast<int>(sizeof(int)), 1, cudaBoundaryModeZero);
    surf2Dread(&out[2], s, 7 * static_cast<int>(sizeof(int)), 1, cudaBoundaryModeClamp);
    surf2Dread(&out[3], s, 1 * static_cast<int>(sizeof(int)), -2, cudaBoundaryModeClamp);
}

cudaTextureObject_t make(cudaArray_t array, cudaTextureAddressMode mode, cudaTextureFilterMode filter, int normalized,
                         cudaError_t* err) {
    cudaResourceDesc res;
    std::memset(&res, 0, sizeof(res));
    res.resType = cudaResourceTypeArray;
    res.res.array.array = array;
    cudaTextureDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.addressMode[0] = mode;
    desc.filterMode = filter;
    desc.readMode = cudaReadModeElementType;
    desc.normalizedCoords = normalized;
    cudaTextureObject_t tex = 0;
    *err = cudaCreateTextureObject(&tex, &res, &desc, nullptr);
    return tex;
}

int main() {
    float texels[kTexels];
    for (int k = 0; k < kTexels; ++k) {
        texels[k] = 10.0f * (k + 1);
    }
    cudaChannelFormatDesc fdesc = cudaCreateChannelDesc<float>();
    cudaArray_t array = nullptr;
    CHECK(cudaMallocArray(&array, &fdesc, kTexels));
    CHECK(cudaMemcpy2DToArray(array, 0, 0, texels, sizeof(texels), sizeof(texels), 1, cudaMemcpyHostToDevice));
    cudaError_t e = cudaSuccess;
    cudaTextureObject_t point = make(array, cudaAddressModeClamp, cudaFilterModePoint, 0, &e);
    CHECK(e);
    cudaTextureObject_t border = make(array, cudaAddressModeBorder, cudaFilterModePoint, 0, &e);
    CHECK(e);
    cudaTextureObject_t linear = make(array, cudaAddressModeClamp, cudaFilterModeLinear, 0, &e);
    CHECK(e);
    cudaTextureObject_t wrap = make(array, cudaAddressModeWrap, cudaFilterModePoint, 1, &e);
    CHECK(e);
    cudaTextureObject_t mirror = make(array, cudaAddressModeMirror, cudaFilterModePoint, 1, &e);
    CHECK(e);

    int words[16];
    for (int k = 0; k < 16; ++k) {
        words[k] = k * k - 3;
    }
    int* linear_words = nullptr;
    CHECK(cudaMalloc(&linear_words, sizeof(words)));
    CHECK(cudaMemcpy(linear_words, words, sizeof(words), cudaMemcpyHostToDevice));
    cudaResourceDesc lres;
    std::memset(&lres, 0, sizeof(lres));
    lres.resType = cudaResourceTypeLinear;
    lres.res.linear.devPtr = linear_words;
    lres.res.linear.desc = cudaCreateChannelDesc<int>();
    lres.res.linear.sizeInBytes = sizeof(words);
    cudaTextureDesc ldesc;
    std::memset(&ldesc, 0, sizeof(ldesc));
    ldesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t fetch = 0;
    CHECK(cudaCreateTextureObject(&fetch, &lres, &ldesc, nullptr));

    unsigned* f = nullptr;
    int* i = nullptr;
    CHECK(cudaMalloc(&f, 15 * sizeof(unsigned)));
    CHECK(cudaMalloc(&i, 2 * sizeof(int)));
    sample<<<1, 1>>>(point, border, linear, wrap, mirror, fetch, f, i);
    CHECK(cudaGetLastError());

    cudaChannelFormatDesc idesc = cudaCreateChannelDesc<int>();
    cudaArray_t grid = nullptr;
    CHECK(cudaMallocArray(&grid, &idesc, kSide, kSide, cudaArraySurfaceLoadStore));
    cudaResourceDesc sres;
    std::memset(&sres, 0, sizeof(sres));
    sres.resType = cudaResourceTypeArray;
    sres.res.array.array = grid;
    cudaSurfaceObject_t surface = 0;
    CHECK(cudaCreateSurfaceObject(&surface, &sres));
    paint<<<1, dim3(kSide, kSide)>>>(surface);
    CHECK(cudaGetLastError());
    int* reads = nullptr;
    CHECK(cudaMalloc(&reads, 4 * sizeof(int)));
    probe<<<1, 1>>>(surface, reads);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    unsigned hf[15] = {};
    int hi[2] = {};
    int hr[4] = {};
    int painted[kSide][kSide] = {};
    CHECK(cudaMemcpy(hf, f, sizeof(hf), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hi, i, sizeof(hi), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hr, reads, sizeof(hr), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy2DFromArray(painted, kSide * sizeof(int), grid, 0, 0, kSide * sizeof(int), kSide,
                                cudaMemcpyDeviceToHost));
    long long weighted = 0;
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            weighted += static_cast<long long>(painted[y][x]) * (y * kSide + x + 1);
        }
    }
    const char* fnames[15] = {"pmid", "pnear", "plow", "phigh", "blow", "bhigh", "bedge", "lquarter",
                              "lhalf", "llow", "lhigh", "wrap", "wrapneg", "mirror", "mirrorneg"};
    for (int k = 0; k < 15; ++k) {
        std::printf("%s=%u\n", fnames[k], hf[k]);
    }
    std::printf("fetch0=%d\n", hi[0]);
    std::printf("fetch5=%d\n", hi[1]);
    std::printf("sread=%d\n", hr[0]);
    std::printf("szero=%d\n", hr[1]);
    std::printf("sclamp=%d\n", hr[2]);
    std::printf("sclampy=%d\n", hr[3]);
    std::printf("painted=%lld\n", weighted);

    cudaDestroySurfaceObject(surface);
    cudaDestroyTextureObject(point);
    cudaDestroyTextureObject(border);
    cudaDestroyTextureObject(linear);
    cudaDestroyTextureObject(wrap);
    cudaDestroyTextureObject(mirror);
    cudaDestroyTextureObject(fetch);
    cudaFreeArray(array);
    cudaFreeArray(grid);
    cudaFree(linear_words);
    cudaFree(f);
    cudaFree(i);
    cudaFree(reads);
    return 0;
}
