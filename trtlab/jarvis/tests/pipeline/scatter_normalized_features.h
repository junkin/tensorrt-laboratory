#pragma once

#include <cuda.h>
#include "../cuda_matrix.h"

void scatter_normalized_features(float** context_feature_buffers, const CuMatrix<float>& new_features, uint32_t batch_size,
                                 uint32_t fft_windows_per_batch, cudaStream_t stream);