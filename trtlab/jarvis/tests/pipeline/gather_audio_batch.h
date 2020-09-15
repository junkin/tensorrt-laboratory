#pragma once

#include "../cuda_matrix.h"

void gather_audio_batch(CuMatrix<float>& audio_batch, int16_t const* const* audio_windows, uint32_t batch_size,
                        cudaStream_t stream);