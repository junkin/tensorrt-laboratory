#include "scatter_normalized_features.h"

#include <glog/logging.h>

__global__ void scatter_normalized_features_kernel(float** __restrict__ feature_buffers_by_context, const float* __restrict__ features,
                                                   uint32_t frames_per_batch, uint32_t features_per_frame)
{
    int tidx      = threadIdx.x;
    int batch_num = blockIdx.x;

    int nframes = frames_per_batch;

    float*       ctx_features   = feature_buffers_by_context[batch_num];
    const float* batch_features = features + batch_num * frames_per_batch * features_per_frame;

    for (int i = tidx; i < features_per_frame; i += blockDim.x)
    {
        float mean   = 0;
        float stddev = 1;

        /*
        for (int time_frame = 0; time_frame < nframes; ++time_frame)
        {
            mean += batch_features[time_frame * features_per_frame + i];
        }
        mean /= nframes;

        for (int time_frame = 0; time_frame < nframes; ++time_frame)
        {
            float feat = batch_features[time_frame * features_per_frame + i];
            stddev += (feat - mean) * (feat - mean);
        }
        stddev = sqrt(stddev / nframes);
        */
        
        for (int time_frame = 0; time_frame < nframes; ++time_frame)
        {
            uint32_t offset = time_frame * features_per_frame + i;
            float normalized_feat = (batch_features[offset] - mean) / stddev;
            ctx_features[offset]  = normalized_feat;
        }
    }
}

void scatter_normalized_features(float** context_feature_buffers, const CuMatrix<float>& new_features, uint32_t batch_size,
                                 uint32_t fft_windows_per_batch, cudaStream_t stream)
{
    CHECK_EQ(fft_windows_per_batch, 11);
    CHECK_EQ(new_features.NumCols(), 64);
    constexpr int threads_per_block = 64;
    scatter_normalized_features_kernel<<<batch_size, threads_per_block, 0, stream>>>(context_feature_buffers, new_features.Data(),
                                                                                     fft_windows_per_batch, new_features.NumCols());
}