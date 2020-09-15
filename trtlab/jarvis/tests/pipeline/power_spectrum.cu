#include "power_spectrum.h"
#include <glog/logging.h>

__global__ void power_spectrum_kernel(float* __restrict__ power_spectrum, int power_stride, const float* __restrict__ fft_bins,
                                      int fft_stride, int num_freq_bins, int fft_windows_per_batch)
{
    int tid     = threadIdx.x;
    int threads = blockDim.x;

    int batch_id  = blockIdx.y;
    int window_id = blockIdx.x;

    int row_id     = batch_id * fft_windows_per_batch + window_id;
    int fft_offset = fft_stride * row_id;
    int pwr_offset = power_stride * row_id;

//  __syncthreads();
//  printf("batch_id: %d; window_id: %d; row_id=%d; fft_stride: %d; power_stride: %d; threads: %d\n", batch_id, window_id, row_id,
//         fft_stride, power_stride, threads);
//  __syncthreads();

    const float2* fft = reinterpret_cast<const float2*>(fft_bins + fft_offset);
    float*        pwr = power_spectrum + pwr_offset;

    for (int idx = tid; idx < num_freq_bins; idx += threads)
    {
        float2 val = fft[idx];
        pwr[idx]   = val.x * val.x + val.y * val.y;
    }
}

void power_spectrum(CuMatrix<float>& power_spectrum, const CuMatrix<float>& fft_bins, int num_freq_bins, int batch_size,
                    int fft_windows_per_batch, cudaStream_t stream)
{
    CHECK_EQ(power_spectrum.NumCols() * 2, fft_bins.NumCols());
    CHECK_EQ(power_spectrum.NumRows(), fft_bins.NumRows());
    CHECK_EQ(power_spectrum.NumCols(), num_freq_bins);
    CHECK_LE(batch_size * fft_windows_per_batch, power_spectrum.NumRows());

    // check defaults - can relax this check
    //CHECK_EQ(num_freq_bins, 64);
    CHECK_EQ(fft_windows_per_batch, 11);

    int  threads_per_block = 64;
    dim3 blocks(fft_windows_per_batch, batch_size);

    power_spectrum_kernel<<<blocks, threads_per_block, 0, stream>>>(power_spectrum.Data(), power_spectrum.Stride(), fft_bins.Data(),
                                                                    fft_bins.Stride(), num_freq_bins, fft_windows_per_batch);
}