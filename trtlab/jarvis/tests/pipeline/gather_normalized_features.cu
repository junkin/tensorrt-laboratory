#include "gather_normalized_features.h"

#include <glog/logging.h>

__global__ void gather_normalized_features_kernel(float* __restrict__ features, float const* const* ctx_feature_buffers, uint32_t nframes,
                                                  uint32_t features_per_frame, bool use_global_stats, float mean, float stddev)
{
    // src: N x (nframes, features_per_frame) row-major
    // dst: (N, features_per_frame, nframes) row-major

    int tidx      = threadIdx.x;
    int batch_num = blockIdx.x;

    float const* ctx_features   = ctx_feature_buffers[batch_num];
    float*       batch_features = features + batch_num * nframes * features_per_frame;

    for (int feat_idx = tidx; feat_idx < features_per_frame; feat_idx += blockDim.x)
    {
        if (!use_global_stats)
        {
            for (int time_frame = 0; time_frame < nframes; ++time_frame)
            {
                mean += ctx_features[time_frame * features_per_frame + feat_idx];
            }
            mean /= nframes;

            for (int time_frame = 0; time_frame < nframes; ++time_frame)
            {
                float feat = ctx_features[time_frame * features_per_frame + feat_idx];
                stddev += (feat - mean) * (feat - mean);
            }
            stddev = sqrt(stddev / nframes);
        }

        for (int time_frame = 0; time_frame < nframes; ++time_frame)
        {
            uint32_t offset                                 = time_frame * features_per_frame + feat_idx;
            float    normalized_feat                        = (ctx_features[offset] - mean) / stddev;
            batch_features[feat_idx * nframes + time_frame] = normalized_feat;
        }
    }
}

void gather_normalized_features(CuMatrix<float>& normalized_features, float const* const* ctx_features, uint32_t batch_size,
                                uint32_t nframes, uint32_t features_per_frame, cudaStream_t stream, bool use_global_stats,
                                const float mean, const float stddev)
{
    CHECK_EQ(features_per_frame, 64);
    constexpr int threads_per_block = 64;
    gather_normalized_features_kernel<<<batch_size, threads_per_block, 0, stream>>>(normalized_features.Data(), ctx_features, nframes,
                                                                                    features_per_frame, use_global_stats, mean, stddev);
}
