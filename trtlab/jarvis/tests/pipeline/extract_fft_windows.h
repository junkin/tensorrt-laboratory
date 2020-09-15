#pragma once

#include <vector>

#include "../config.h"
#include "../cuda_matrix.h"
#include "../cuda_vector.h"


// creates a weights vector for the windowing function
// the vector is sized to the next power of two
// the weights are centered in this vector padded with 0s
class WindowingFunction
{
public:
    WindowingFunction(const Config&);
    virtual ~WindowingFunction() = default;

    // access the non-padded weights
    const float *windowing_weights() const { return m_Weights.data() + centering_offset(); }
    std::uint32_t window_size() const { return m_WindowSize; }
    std::uint32_t shift_size() const { return m_ShiftSize; }
    std::uint32_t centering_offset() const { return m_CenterOffset; }

    // access the padded weights
    const float *padded_windowing_weights() const { return m_Weights.data(); }
    std::uint32_t padded_window_size() const { return m_Weights.size(); }

private:
    std::uint32_t m_WindowSize;
    std::uint32_t m_ShiftSize;
    std::uint32_t m_CenterOffset;
    std::vector<float> m_Weights;
};

class CudaWindowingFunction : public WindowingFunction
{
public:
    CudaWindowingFunction(const Config&);
    ~CudaWindowingFunction() override = default;
};

void extract_fft_windows(CuMatrix<float>& fft_windows, const CuMatrix<float>& audio, const CudaWindowingFunction& wfn,
                         uint32_t fft_windows_per_batch, uint32_t batch_size, cudaStream_t stream);