

#include "test_jarvis.h"

#include <trtlab/cuda/common.h>

#include "pipeline/scatter_features.h"
#include "pipeline/gather_normalized_features.h"

static float feat_val(int i, int j, int k)
{
    return 10. * i + (static_cast<float>(j) + static_cast<float>(k) / 100.);
}

TEST_F(TestJarvis, GatherNormalizedFeaturesWithGlobalStats)
{
    // src: feats (N, frames, features)
    // intermediate: N x (frames, features)
    // state: feats (N, features, frames) normalized

    std::uint32_t batch_size = 256;
    std::uint32_t frames     = 251;
    std::uint32_t features   = 64;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // feats - this holds the data that will be scatter
    CuMatrix<float> feats(stream);
    feats.Resize(batch_size * frames, features, kUndefined, kStrideEqualNumCols);

    // initialize contiguous data source
    std::vector<float> h_feats;
    h_feats.resize(batch_size * frames * features);

    std::size_t idx = 0;
    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < frames; j++)
        {
            for (int k = 0; k < features; k++)
            {
                h_feats[idx++] = feat_val(i,j,k);
            }
        }
    }
    CHECK_CUDA(
        cudaMemcpyAsync(feats.Data(), h_feats.data(), batch_size * frames * features * sizeof(float), cudaMemcpyHostToDevice, stream));

    // intermediate
    std::vector<CuMatrix<float>> state;
    std::vector<float*>          ptrs;

    state.reserve(batch_size);
    ptrs.reserve(batch_size);

    for (int i = 0; i < batch_size; i++)
    {
        state.emplace_back(stream);
        state[i].Resize(frames, features, kUndefined, kStrideEqualNumCols);
        state[i].SetZero();
        ptrs.push_back(state[i].Data());
    }

    // the list of device pointers for which data will be scattered
    void* d_ptrs = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptrs, batch_size * sizeof(float*)));
    CHECK_CUDA(cudaMemcpyAsync(d_ptrs, static_cast<void*>(ptrs.data()), batch_size * sizeof(float*), cudaMemcpyHostToDevice, stream));

    // write features to their individual state machines
    scatter_features(static_cast<float**>(d_ptrs), feats, batch_size, frames, features, stream);

    // gather without nomalization (use_global_stats = true, mean = 0.0, stddev = 1.0)
    gather_normalized_features(feats, static_cast<float**>(d_ptrs), batch_size, frames, features, stream, true, 0.0, 1.0);

    CHECK_CUDA(
        cudaMemcpyAsync(h_feats.data(), feats.Data(), batch_size * frames * features * sizeof(float), cudaMemcpyDeviceToHost, stream));

    for (int i = 0; i < batch_size; i++)
    {
        for (int k = 0; k < features; k++)
        {
            for (int j = 0; j < frames; j++)
            {
                std::size_t offset = i * (features * frames) + k * frames + j;
                float       val    = feat_val(i,j,k);
                ASSERT_FLOAT_EQ(h_feats[offset], val) << "i: " << i << "; j: " << j << "; k: " << k;
            }
        }
    }
}

TEST_F(TestJarvis, GatherNormalizedFeaturesComputeStats)
{
    // src: feats (N, frames, features)
    // intermediate: N x (frames, features)
    // state: feats (N, features, frames) normalized

    std::uint32_t batch_size = 256;
    std::uint32_t frames     = 251;
    std::uint32_t features   = 64;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // feats - this holds the data that will be scatter
    CuMatrix<float> feats(stream);
    feats.Resize(batch_size * frames, features, kUndefined, kStrideEqualNumCols);

    // initialize contiguous data source
    std::vector<float> h_feats;
    h_feats.resize(batch_size * frames * features);

    std::size_t idx = 0;
    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < frames; j++)
        {
            for (int k = 0; k < features; k++)
            {
                h_feats[idx++] = feat_val(i,j,k);
            }
        }
    }

    CHECK_CUDA(
        cudaMemcpyAsync(feats.Data(), h_feats.data(), batch_size * frames * features * sizeof(float), cudaMemcpyHostToDevice, stream));

    // intermediate
    std::vector<CuMatrix<float>> state;
    std::vector<float*>          ptrs;

    state.reserve(batch_size);
    ptrs.reserve(batch_size);

    for (int i = 0; i < batch_size; i++)
    {
        state.emplace_back(stream);
        state[i].Resize(frames, features, kUndefined, kStrideEqualNumCols);
        state[i].SetZero();
        ptrs.push_back(state[i].Data());
    }

    // the list of device pointers for which data will be scattered
    void* d_ptrs = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptrs, batch_size * sizeof(float*)));
    CHECK_CUDA(cudaMemcpyAsync(d_ptrs, static_cast<void*>(ptrs.data()), batch_size * sizeof(float*), cudaMemcpyHostToDevice, stream));

    // write features to their individual state machines
    scatter_features(static_cast<float**>(d_ptrs), feats, batch_size, frames, features, stream);

    // gather without nomalization (use_global_stats = true, mean = 0.0, stddev = 1.0)
    gather_normalized_features(feats, static_cast<float**>(d_ptrs), batch_size, frames, features, stream, false);

    CHECK_CUDA(
        cudaMemcpyAsync(h_feats.data(), feats.Data(), batch_size * frames * features * sizeof(float), cudaMemcpyDeviceToHost, stream));

    for (int i = 0; i < batch_size; i++)
    {
        for (int k = 0; k < features; k++)
        {
            float mean   = 0.0;
            float stddev = 0.0;

            for (int j = 0; j < frames; j++)
            {
                mean += feat_val(i,j,k);
            }
            mean /= frames;

            for (int j = 0; j < frames; j++)
            {
                float feat = feat_val(i,j,k);
                stddev += (feat - mean) * (feat - mean);
            }
            stddev = std::sqrt(stddev / frames);

            for (int j = 0; j < frames; j++)
            {
                float       feat   = feat_val(i,j,k);
                float       val    = (feat - mean) / stddev;
                std::size_t offset = i * (features * frames) + k * frames + j;
                // ASSERT_FLOAT_EQ(h_feats[offset], val) << "i: " << i << "; j: " << j << "; k: " << k;
                // the monotonically increasing feat_val function starts to reach the limits
                // of floating point precision as the batch size grows.
                // possibly update the formual with some periodicity to avoid unbounded growth
                // regardless, if you wish tigher precision to host side calcuated normalization statistics
                // you may have to adjust your cuda compiler settings
                ASSERT_NEAR(h_feats[offset], val, 1e-5) << "i: " << i << "; j: " << j << "; k: " << k;
            }
        }
    }
}