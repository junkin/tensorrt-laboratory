#pragma once

#include "../cuda_matrix.h"
#include "../cuda_vector.h"
#include "audio_collector.h"

void process_audio(CuMatrix<float>& processed_audio, const CuMatrix<float>& audio_batch, const CuVector<float>& noise, float gain, float dither,
                   float preemph_coeff, uint32_t batch_size, cudaStream_t stream);
