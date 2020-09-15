
#pragma once

#include <cuda.h>
#include "../cuda_matrix.h"

void power_spectrum(CuMatrix<float>& power_spectrum, const CuMatrix<float>& fft_bins, int num_bins, int batch_size,
                    int fft_windows_per_batch, cudaStream_t stream);
