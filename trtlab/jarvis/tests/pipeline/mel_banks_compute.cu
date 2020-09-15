#include "mel_banks_compute.h"

#include <cub/cub.cuh>

// Expects to be called with 32x8 sized thread block.
__global__ void mel_banks_compute_kernel(float* mels, int32_t mels_stride, const float* feats, int32_t feats_stride, const int32_t* offsets,
                                         const int32_t* sizes, const float* const* vecs, int32_t batch_size, int32_t fft_windows_per_batch)

{
    constexpr float energy_floor = 5.96046448e-8; /* 2**-24 used in NeMo */

    // Specialize WarpReduce for type float
    typedef cub::WarpReduce<float> WarpReduce;
    // Allocate WarpReduce shared memory for 8 warps
    __shared__ typename WarpReduce::TempStorage temp_storage[8];

    // warp will work together to compute sum
    int tid = threadIdx.x;
    int wid = threadIdx.y;
    // blocks in the x dimension take different bins
    int bin = blockIdx.x;
    // frame is a combination of blocks in the y dimension and threads in the y
    // dimension
    int raw_frame = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if frame has work to do
    int batch_num = raw_frame / fft_windows_per_batch;
    // Frame within wave
    // int frame = raw_frame - batch_num * fft_windows_per_batch;

    if (batch_num < batch_size)
    {
        int          offset = offsets[bin];
        int          size   = sizes[bin];
        const float* v      = vecs[bin];
        const float* w      = feats + (raw_frame * feats_stride + offset);

        // perfom local sum
        float sum = 0;
        for (int idx = tid; idx < size; idx += 32)
        {
            sum += v[idx] * w[idx];
        }

        // Sum in cub
        sum = WarpReduce(temp_storage[wid]).Sum(sum);
        if (tid == 0)
        {
            // avoid log of zero
            // if (sum < energy_floor) sum = energy_floor;
            sum += energy_floor;
            float val                           = logf(sum);
            mels[raw_frame * mels_stride + bin] = val;
        }
    }
}

void mel_banks_compute(CuMatrix<float>& new_features, const CuMatrix<float>& power_spectrum, const CudaMelBanks& mel_banks, int batch_size,
                       int fft_windows_per_batch, cudaStream_t stream)
{
    int32_t bin_size = 64;

    dim3 mel_threads(32, 8);
    dim3 mel_blocks(bin_size, (batch_size * fft_windows_per_batch + mel_threads.y - 1) / mel_threads.y);

    mel_banks_compute_kernel<<<mel_blocks, mel_threads, 0, stream>>>(new_features.Data(), new_features.Stride(), power_spectrum.Data(),
                                                                     power_spectrum.Stride(), mel_banks.offsets(), mel_banks.sizes(),
                                                                     mel_banks.vecs(), batch_size, fft_windows_per_batch);
}
