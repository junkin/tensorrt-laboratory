#include "test_jarvis.h"

#include <trtlab/cuda/common.h>

#include "pipeline/scatter_features.h"

TEST_F(TestJarvis, ScatterFeatures)
{
    // src: (N, frames, features)
    // dst: Nx (frames, features)

    std::uint32_t batch_size = 256;
    std::uint32_t frames     = 251;
    std::uint32_t features   = 64;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // src - this holds the data that will be scatter
    CuMatrix<float> src(stream);
    src.Resize(batch_size * frames, features, kUndefined, kStrideEqualNumCols);

    // initialize contiguous data source
    std::vector<float> h_src;
    h_src.resize(batch_size * frames * features);

    std::size_t idx = 0;
    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < frames; j++)
        {
            for (int k = 0; k < features; k++)
            {
                h_src[idx++] = 1000. * i + (static_cast<float>(j) + static_cast<float>(k) / 100.);
            }
        }
    }
    CHECK_CUDA(cudaMemcpyAsync(src.Data(), h_src.data(), batch_size * frames * features * sizeof(float), cudaMemcpyHostToDevice, stream));

    // dst
    std::vector<CuMatrix<float>> dst;
    std::vector<float*>          ptrs;

    dst.reserve(batch_size);
    ptrs.reserve(batch_size);

    std::vector<std::vector<float>> vals;
    vals.resize(batch_size);

    for (int i = 0; i < batch_size; i++)
    {
        dst.emplace_back(stream);
        dst[i].Resize(frames, features, kUndefined, kStrideEqualNumCols);
        ptrs.push_back(dst[i].Data());
        dst[i].SetZero();

        vals[i].resize(features * frames);
        std::fill(vals[i].begin(), vals[i].end(), 0);
    }

    // the list of device pointers for which data will be scattered
    void* d_ptrs = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptrs, batch_size * sizeof(float*)));
    CHECK_CUDA(cudaMemcpyAsync(d_ptrs, static_cast<void*>(ptrs.data()), batch_size * sizeof(float*), cudaMemcpyHostToDevice, stream));

    scatter_features(static_cast<float**>(d_ptrs), src, batch_size, frames, features, stream);

    for (int i = 0; i < batch_size; i++)
    {
        CHECK_CUDA(cudaMemcpyAsync(vals[i].data(), dst[i].Data(), features * frames * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        for (int j = 0; j < frames; j++)
        {
            for (int k = 0; k < features; k++)
            {
                float val = 1000.0 * i + (static_cast<float>(j) + static_cast<float>(k) / 100.);
                ASSERT_FLOAT_EQ(vals[i][j * features + k], val) << "i: " << i << "; j: " << j << "; k: " << k;
            }
        }
    }
}