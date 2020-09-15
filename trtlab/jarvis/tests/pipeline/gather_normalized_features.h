#pragma once

#include "../cuda_matrix.h"

void gather_normalized_features(CuMatrix<float>& normalized_features, float const* const* ctx_features, uint32_t batch_size,
                                uint32_t nframes, uint32_t features_per_frame, cudaStream_t stream, bool use_global_stats = true,
                                const float mean = 0., const float stddev = 1.0);