
#include "process_audio.h"

#include <glog/logging.h>

__global__ void process_audio_kernel(const float* __restrict__ waves_in, int waves_in_stride, float* __restrict__ waves_out,
                                     int waves_out_stride, const float* __restrict__ noise_in, int batch_size, int num_samples, float gain,
                                     float dither, float preemph_coeff, uint32_t padding_size)
{
    int batch_num = blockIdx.y;
    int sample    = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (sample < num_samples && batch_num < batch_size)
    {
        int half_padding_size = padding_size / 2;

        const float* wave_in  = waves_in + waves_in_stride * batch_num;
        float*       wave_out = waves_out + waves_out_stride * batch_num;

        // the preprocessed audio buffer is num_sample + 1 in size
        // we carry over one extra signal so we can accurately compute the preemph 
        // thus the input signal is shifted +1 from the output signal, i.e.
        float signal      = wave_in[sample + 1] * gain;
        float prev_signal = wave_in[sample] * gain;
        float noise       = dither != 0.0f ? dither * noise_in[sample + half_padding_size + 1] : 0.;
        float prev_noise  = dither != 0.0f ? dither * noise_in[sample + half_padding_size] : 0.;
        float out_signal  = signal + noise - preemph_coeff * (prev_signal + prev_noise);

        wave_out[sample + half_padding_size] = out_signal;

        // Apply mirror of wave_out
        if (sample <= half_padding_size)
        {
            int sample_mirror       = half_padding_size - sample;
            wave_out[sample_mirror] = out_signal;
        }
        else if (sample >= num_samples - half_padding_size - 1)
        {
            int fft_id              = num_samples - sample - 1;
            int sample_mirror       = num_samples + half_padding_size + fft_id - 1;
            wave_out[sample_mirror] = out_signal;
        }
    }
}

void process_audio(CuMatrix<float>& processed_audio, const CuMatrix<float>& audio_batch, const CuVector<float>& noise, float gain, float dither,
                   float preemph_coeff, uint32_t batch_size, cudaStream_t stream)
{
    constexpr uint32_t threads_per_block = 64;
    uint32_t padding_size = processed_audio.NumCols() - audio_batch.NumCols() - 1; // -1 to account for the single extra audio signal

    uint32_t blocks_x = (processed_audio.Stride() + threads_per_block - 1) / threads_per_block;
    dim3     blocks(blocks_x, batch_size);

    DVLOG(3) << "launching process_audio_kernel";
    process_audio_kernel<<<blocks, threads_per_block, 0, stream>>>(audio_batch.Data(), audio_batch.Stride(), processed_audio.Data(),
                                                                   processed_audio.Stride(), noise.Data(), batch_size,
                                                                   audio_batch.NumCols(), gain, dither, preemph_coeff, padding_size);
}