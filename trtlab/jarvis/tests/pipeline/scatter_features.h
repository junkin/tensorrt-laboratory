#pragma once

#include "../cuda_matrix.h"

void scatter_features(float** context_new_feature_buffers, const CuMatrix<float>& new_features, uint32_t batch_size,
                      uint32_t nframes, uint32_t features, cudaStream_t stream);