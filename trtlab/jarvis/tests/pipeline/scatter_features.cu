#include "scatter_features.h"

#include <glog/logging.h>

__global__ void scatter_features_kernel(float** __restrict__ feature_buffers_by_context, const float* __restrict__ features,
                                        uint32_t frames_per_batch, uint32_t features_per_frame)
{
    int tidx      = threadIdx.x;
    int frame     = blockIdx.x;
    int batch_num = blockIdx.y;

    float*       ctx_features   = feature_buffers_by_context[batch_num];
    const float* batch_features = features + batch_num * frames_per_batch * features_per_frame;

    int offset = frame * features_per_frame + tidx;

    ctx_features[offset] = batch_features[offset];
}

void scatter_features(float** context_new_feature_buffers, const CuMatrix<float>& new_features, uint32_t batch_size,
                                 uint32_t frames, uint32_t features, cudaStream_t stream)
{
    CHECK_EQ(features, 64);
    CHECK_EQ(new_features.NumCols(), features);
    constexpr int threads_per_block = 64;
    dim3 blocks(frames, batch_size);
    scatter_features_kernel<<<blocks, threads_per_block, 0, stream>>>(context_new_feature_buffers, new_features.Data(),
                                                                                     frames, features);
}