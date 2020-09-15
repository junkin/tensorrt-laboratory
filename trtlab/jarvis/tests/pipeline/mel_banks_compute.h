#pragma once

#include <cuda.h>

#include "../cuda_matrix.h"
#include "../cuda_mel_banks.h"

void mel_banks_compute(CuMatrix<float>& m_new_features, const CuMatrix<float>& pwr_windows, const CudaMelBanks& mel_banks, int batch_size,
                       int fft_windows_per_batch, cudaStream_t stream);