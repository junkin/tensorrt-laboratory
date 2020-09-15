#include "extract_fft_windows.h"

#include <glog/logging.h>

static constexpr double M_2PI = 6.283185307179586476925286766559005;

#define CUDA_WINDOWING_FN_MAX_SIZE 512

// the windowing weights are stored in constant memory on the gpus
__constant__ float constant_symbol_padded_windowing_weights[CUDA_WINDOWING_FN_MAX_SIZE];

__global__ void extract_fft_windows_kernel(float* __restrict__ fft_windows, uint32_t fft_window_stride, const float* __restrict__ audio,
                                           uint32_t audio_stride, uint32_t shift_size, uint32_t fft_windows_per_batch)
{
    int tid = threadIdx.x;

    int batch_id  = blockIdx.y;
    int window_id = blockIdx.x;

    const float* audio_start  = audio + audio_stride * batch_id;
    const float* audio_window = audio_start + shift_size * window_id;

    float* fft_window = fft_windows + fft_window_stride * (batch_id * fft_windows_per_batch + window_id);

    fft_window[tid] = audio_window[tid] * constant_symbol_padded_windowing_weights[tid];
}

void extract_fft_windows(CuMatrix<float>& fft_windows, const CuMatrix<float>& audio, const CudaWindowingFunction& wfn,
                         uint32_t fft_windows_per_batch, uint32_t batch_size, cudaStream_t stream)
{
    CHECK_EQ(fft_windows.Stride(), 512);
    uint32_t threads_per_block = fft_windows.Stride();
    dim3     blocks(fft_windows_per_batch, batch_size);

    extract_fft_windows_kernel<<<blocks, threads_per_block, 0, stream>>>(fft_windows.Data(), fft_windows.Stride(), audio.Data(),
                                                                         audio.Stride(), wfn.shift_size(), fft_windows_per_batch);
}

WindowingFunction::WindowingFunction(const Config& cfg)
: m_WindowSize(cfg.extraction_frame_size_ms * cfg.sample_freq_ms), m_ShiftSize(cfg.extraction_frame_shift_ms * cfg.sample_freq_ms)
{
    std::uint32_t fft_length   = cfg.extraction_fft_length;
    std::uint32_t frame_length = window_size();

    // allocate padded vector and zero
    m_Weights.resize(fft_length);
    std::fill(m_Weights.begin(), m_Weights.end(), 0.);

    // actual weights start at this offset
    m_CenterOffset = (fft_length - frame_length) / 2;

    const double a = M_2PI / (frame_length - 1);
    CHECK_LE(fft_length, CUDA_WINDOWING_FN_MAX_SIZE);

    for (std::uint32_t i = 0; i < frame_length; i++)
    {
        double i_fl = static_cast<double>(i);
        if (cfg.extraction_window_type == "hanning")
        {
            m_Weights[i + m_CenterOffset] = 0.5 - 0.5 * std::cos(a * i_fl);
        }
        else if (cfg.extraction_window_type == "hamming")
        {
            m_Weights[i + m_CenterOffset] = 0.54 - 0.46 * std::cos(a * i_fl);
        }
        else if (cfg.extraction_window_type == "povey")
        { // like hamming but goes to zero
            // at edges.
            m_Weights[i + m_CenterOffset] = pow(0.5 - 0.5 * std::cos(a * i_fl), 0.85);
        }
        else if (cfg.extraction_window_type == "rectangular")
        {
            m_Weights[i + m_CenterOffset] = 1.0;
        }
        else if (cfg.extraction_window_type == "blackman")
        {
            m_Weights[i + m_CenterOffset] =
                cfg.extraction_blackman_coeff - 0.5 * std::cos(a * i_fl) + (0.5 - cfg.extraction_blackman_coeff) * std::cos(2 * a * i_fl);
        }
        else
        {
            LOG(FATAL) << "Invalid m_Weights type " << cfg.extraction_window_type;
        }
    }
}

CudaWindowingFunction::CudaWindowingFunction(const Config& cfg) : WindowingFunction(cfg)
{
    CHECK_EQ(cudaMemcpyToSymbol(constant_symbol_padded_windowing_weights, padded_windowing_weights(), padded_window_size() * sizeof(float)),
             cudaSuccess);
}