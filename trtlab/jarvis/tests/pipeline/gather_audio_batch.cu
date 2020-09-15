
#include "gather_audio_batch.h"

__global__ void gather_audio_batch_kernel(float* __restrict__ buffer, uint32_t buffer_stride, int16_t const* const* audio_windows,
                                          uint32_t window_size)
{
    int b = blockIdx.y;
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    int offset = b * buffer_stride + i;

    if (i < window_size)
    {
        //if(i == 0 && b == 0)
        //printf("b: %d; i: %d; s: %d; o: %d; b: %lf; windows: %x\n", b, i, audio_windows[b][i], offset, buffer[offset], audio_windows);

        buffer[offset] = static_cast<float>(audio_windows[b][i]) / static_cast<float>(INT16_MAX);
    }
}

void gather_audio_batch(CuMatrix<float>& audio_batch, int16_t const* const* audio_windows, uint32_t batch_size, cudaStream_t stream)
{
    constexpr uint32_t threads_per_block = 64;

    auto blocks_x = (audio_batch.Stride() + threads_per_block - 1) / threads_per_block;
    dim3 blocks(blocks_x, batch_size);

    gather_audio_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(audio_batch.Data(), audio_batch.Stride(), audio_windows,
                                                                        audio_batch.NumCols());
}