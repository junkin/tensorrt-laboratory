#include "pipeline.h"

#include <cuda.h>
#include <cufft.h>
#include <curand.h>

#include <trtlab/cuda/common.h>
#include <trtlab/cuda/sync.h>

#include <trtlab/tensorrt/model.h>
#include <trtlab/tensorrt/execution_context.h>

#include "cuda_matrix.h"
#include "cuda_vector.h"

#include "context_pointers.h"

#include "pipeline/gather_audio_batch.h"
//#include "pipeline/audio_collector.h"
#include "pipeline/process_audio.h"
#include "pipeline/extract_fft_windows.h"
#include "pipeline/power_spectrum.h"
#include "pipeline/mel_banks_compute.h"
#include "pipeline/scatter_features.h"
#include "pipeline/gather_normalized_features.h"

using namespace trtlab;

namespace
{
    cudaStream_t make_stream()
    {
        cudaStream_t stream;
        CHECK_EQ(cudaStreamCreate(&stream), cudaSuccess);
        return stream;
    };
} // namespace

class Pipeline::Impl
{
public:
    using context_t = Context;

    Impl(const Config& cfg, std::shared_ptr<Resources> resources);
    ~Impl();

    void compute(const batch_t&, release_fn);

    cudaStream_t stream()
    {
        return m_stream;
    }

protected:
    void write_matrix(const CuMatrix<float>&, const std::string);

private:
    cudaStream_t      m_stream;
    cublasHandle_t    m_cublas_handle;
    curandGenerator_t m_curand_handle;

    std::shared_ptr<Resources> m_resources;

    // fft parameters
    const std::uint32_t m_fft_length;
    const std::uint32_t m_num_frequency_bins; // (fft_length / 2 + 1)
    const std::uint32_t m_num_mel_bins;

    // processing audio
    const float m_dither;
    const float m_gain;
    const float m_preemph_coeff;

    // the number of extra samples to pad to the audio window
    // simplifies the kernels for extracting windows of fft_length
    // padding is (fft_length - fft_window_size)
    const std::uint32_t m_audio_padding;

    // number of fft windows an audio buffer window
    // (audio_buffer_window_size - fft_window_size) / fft_shift_size + 1
    const std::uint32_t m_fft_windows_per_batch;

    // gather audio from each context's audio buffer to a contiguous batch
    CuMatrix<float> m_audio_batch;

    // process audio kernel
    CuMatrix<float> m_processed_audio;
    CuVector<float> m_noise;

    // extract windows kernel
    CuMatrix<float> m_fft_windows;

    // fft results
    CuMatrix<float> m_fft_bins;

    // compute magnitude / power of fft results
    CuMatrix<float> m_power_spectrum;

    // new features (mels)
    CuMatrix<float> m_new_features;

    // acoustic inputs
    CuMatrix<float> m_acoustic_inputs;

    // cufft plans by batch size
    std::map<std::size_t, cufftHandle> m_fft_plans_by_batch_size;

    // cuda events
    cudaEvent_t m_release_inputs;
    cudaEvent_t m_release_features;

    // contiguous memory for upto max_batch_size audio buffer and feature buffer pointers
    // this class stages the ptrs needed in kernel to device buffers via pinned host buffers
    // ptrs are initialized for each batch by the init method
    ContextPointers m_ctx_pointers;

    // models - tensorrt engines
    std::unique_ptr<TensorRT::ExecutionContext> m_encoder;
    std::unique_ptr<TensorRT::ExecutionContext> m_decoder;

    memory::descriptor m_trt_device_memory;
    memory::descriptor m_acoustic_intermediates;
    memory::descriptor m_acoustic_outputs;
};

// Public Pipeline Impl

Pipeline::Pipeline(const Config& cfg, std::shared_ptr<Resources> resources) : pImpl{std::make_unique<Impl>(cfg, resources)} {}

Pipeline::~Pipeline() = default;

Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

void Pipeline::compute(const batch_t& batch, release_fn release_inputs)
{
    pImpl->compute(batch, release_inputs);
}

cudaStream_t Pipeline::stream()
{
    pImpl->stream();
}

// Private Pipeline Impl

Pipeline::Impl::Impl(const Config& cfg, std::shared_ptr<Resources> resources)
: m_stream(make_stream()),
  m_resources(resources),
  m_ctx_pointers(cfg.max_batch_size, *resources, m_stream),
  // dedicated buffers/tensors/matrices/vectors
  m_audio_batch(m_stream),
  m_processed_audio(m_stream),
  m_noise(m_stream),
  m_fft_windows(m_stream),
  m_fft_bins(m_stream),
  m_power_spectrum(m_stream),
  m_new_features(m_stream),
  // variables
  m_dither(cfg.extraction_dither),
  m_gain(cfg.extraction_gain),
  m_preemph_coeff(cfg.extraction_preemph_coeff),
  m_fft_length(cfg.extraction_fft_length),
  m_num_frequency_bins(m_fft_length / 2 + 1),
  m_num_mel_bins(cfg.features_count),
  m_audio_padding(m_fft_length - resources->windowing_function().window_size()),
  m_fft_windows_per_batch(cfg.features_per_audio_buffer_window)
{
    // check default for sanity
    CHECK_EQ(cfg.extraction_fft_length, 512);
    CHECK_EQ(m_num_frequency_bins, 257);
    CHECK_EQ(resources->windowing_function().window_size(), 320);
    CHECK_EQ(cfg.audio_buffer_window_size_ms, 120);
    CHECK_EQ(m_fft_windows_per_batch, 11);
    CHECK_EQ(cfg.audio_buffer_window_size_samples, 120 * 16);
    CHECK_EQ(cfg.audio_buffer_preprocessed_window_size_samples, 120 * 16 + 1);

    // gather audio from each context's audio buffer to a contiguous batch
    // the preprocessed audio contains one extra signal of historic / overlapped audio
    // so we can compute the preemph of the first "new" signal in this window
    // the last signal of the previous frame and the first signal of the new frame will be
    // identical if dither == 0; otherwise, they will be similar but differ due to variance
    // in the noise values being applied to the original signal values
    m_audio_batch.Resize(cfg.max_batch_size, cfg.audio_buffer_preprocessed_window_size_samples, kUndefined);

    // noise
    m_noise.Resize(cfg.audio_buffer_preprocessed_window_size_samples + cfg.extraction_fft_length, kUndefined);

    // dither + gain + fft padding kernel
    // pad the front and back of the audio by (fft_length - window_size) / 2
    // the padding ensures the first fft window/frame is centered at 10ms spanning [0m, 20ms]
    m_processed_audio.Resize(cfg.max_batch_size, cfg.audio_buffer_window_size_samples + m_audio_padding, kUndefined);

    // extract fft windows from audio
    // default extraction window size is 20ms
    // default extraction shift size is 10ms
    // default audio buffer is 120ms
    // default fft_windows or features per audio buffer is 11 [(120 - 20) / 10 + 1]
    // todo: create a 3D tensor for clearer indexing: [batch_size, 11, 512]
    // here we are combining the first two indexes: [batch_size * 11, 512]
    // default fft length is 512 (next power of 2)
    // the 512 sample window is padded with zeros at [0, 96) and [416, 512)
    // the audio data is centered in the fft window [96, 416) and weighted by the windowing function
    // default audio window size is 120ms, these windows have been padded by 96 samples
    // which allows for non-branching kernels as the firs
    auto rows = cfg.max_batch_size * m_fft_windows_per_batch;
    auto cols = cfg.extraction_fft_length;
    m_fft_windows.Resize(rows, cols, kUndefined, kStrideEqualNumCols);

    // compute fft on fft_windows (double the number of bin)
    m_fft_bins.Resize(rows, 2 * m_num_frequency_bins, kUndefined, kStrideEqualNumCols);

    // compute the magnitude / power of each of the complex values from the fft
    m_power_spectrum.Resize(rows, m_num_frequency_bins, kUndefined, kStrideEqualNumCols);

    // new features (mels)
    m_new_features.Resize(rows, m_num_mel_bins, kUndefined, kStrideEqualNumCols);

    // acoustic inputs
    m_acoustic_inputs.Resize(cfg.max_batch_size * m_num_mel_bins, cfg.features_buffer_window_size_feats, kUndefined, kStrideEqualNumCols);

    // Initialize cuBlas
    CHECK_EQ(cublasCreate(&m_cublas_handle), CUBLAS_STATUS_SUCCESS);

    // Initialize cuRand
    CHECK_EQ(curandCreateGenerator(&m_curand_handle, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    // To get same random sequence, call srand() before the constructor is invoked,
    CHECK_EQ(curandSetGeneratorOrdering(m_curand_handle, CURAND_ORDERING_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandSetStream(m_curand_handle, m_stream), CURAND_STATUS_SUCCESS);

    // To get same random sequence, call srand() before the method is invoked,
    // CHECK_EQ(curandSetPseudoRandomGeneratorSeed(m_curand_handle, RandInt(128, RAND_MAX)), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandSetGeneratorOffset(m_curand_handle, 0), CURAND_STATUS_SUCCESS);

    // create 4 fft plans: b1, 1/4, 1/2 and max_batch_size
    std::vector<std::size_t> sizes = {cfg.max_batch_size / 4, cfg.max_batch_size / 2, cfg.max_batch_size};

    for (auto& bs : sizes)
    {
        cufftHandle plan;

        int fft_length = m_fft_length;
        CHECK_EQ(cufftPlanMany(&plan, 1, &fft_length, NULL, 1, m_fft_length, NULL, 1, m_num_frequency_bins, CUFFT_R2C,
                               bs * m_fft_windows_per_batch),
                 CUFFT_SUCCESS);
        CHECK_EQ(cufftSetStream(plan, m_stream), CUFFT_SUCCESS);
        CHECK_EQ(m_fft_length, fft_length);
        m_fft_plans_by_batch_size[bs] = plan;
    }

    // cuda events
    CHECK_CUDA(cudaEventCreateWithFlags(&m_release_inputs, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventCreateWithFlags(&m_release_features, cudaEventDisableTiming));

    // deserialize trt engines - this should be global, but due to trt limits
    // we require 1 ICudaEngine per IExeuctionContext for Optimized Profiles
    auto encoder = resources->trt_runtime().deserialize_engine(cfg.acoustic_encoder_path);
    auto decoder = resources->trt_runtime().deserialize_engine(cfg.acoustic_decoder_path);

    m_encoder = std::make_unique<TensorRT::ExecutionContext>(encoder);
    m_decoder = std::make_unique<TensorRT::ExecutionContext>(decoder);

    auto trt_device_memory_size = std::max(m_encoder->engine().getDeviceMemorySize(), m_decoder->engine().getDeviceMemorySize());
    m_trt_device_memory         = resources->device_allocator().allocate_descriptor(trt_device_memory_size);

    // set shared device memory for each execution context
    m_encoder->context().setDeviceMemory(m_trt_device_memory.data());
    m_decoder->context().setDeviceMemory(m_trt_device_memory.data());

    m_encoder->context().setOptimizationProfile(0);
    m_decoder->context().setOptimizationProfile(0);

    nvinfer1::Dims3 input_dims; // (N, 64, 251)
    input_dims.d[0] = cfg.max_batch_size;
    input_dims.d[1] = cfg.features_count;
    input_dims.d[2] = cfg.features_buffer_window_size_feats;

    LOG(INFO) << "input dims: " << TensorRT::Model::dims_info(input_dims);

    m_encoder->context().setBindingDimensions(0, input_dims);

    auto inter_dims = m_encoder->context().getBindingDimensions(1);
    m_acoustic_intermediates =
        resources->device_allocator().allocate_descriptor(inter_dims.d[0] * inter_dims.d[1] * inter_dims.d[2] * sizeof(float));

    m_decoder->context().setBindingDimensions(0, inter_dims);

    auto output_dims = m_decoder->context().getBindingDimensions(1);

    LOG(INFO) << "output dims: " << TensorRT::Model::dims_info(output_dims);

    m_acoustic_outputs =
        resources->device_allocator().allocate_descriptor(output_dims.d[0] * output_dims.d[1] * output_dims.d[2] * sizeof(float));
}

Pipeline::Impl::~Impl()
{
    CHECK_CUDA(cudaStreamSynchronize(m_stream));

    for (auto& pair : m_fft_plans_by_batch_size)
    {
        CHECK_EQ(cufftDestroy(pair.second), CUFFT_SUCCESS);
    }
    CHECK_CUDA(cudaEventDestroy(m_release_inputs));
    CHECK_CUDA(cudaEventDestroy(m_release_features));
    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaStreamDestroy(m_stream));
}

void Pipeline::Impl::compute(const batch_t& batch, release_fn release_inputs)
{
    static std::size_t counter = 0;

    // debugging sanity checks
    DCHECK_EQ(m_audio_batch.NumCols(), batch[0].bytes / sizeof(std::int16_t));
    DCHECK_EQ(m_audio_batch.NumCols() - 1 + m_audio_padding, m_processed_audio.NumCols());

    // copy 3*batch.size() pointers to device memory
    // these are pointers to the context's audio buffer, feature buffer and new features
    m_ctx_pointers.init(batch);

    // collect independing audio windows into a single contiguous batch
    gather_audio_batch(m_audio_batch, m_ctx_pointers.audio_buffers(), batch.size(), m_stream);

    // record an event on teh stream so we can release inputs
    CHECK_EQ(cudaEventRecord(m_release_inputs, m_stream), CUDA_SUCCESS);

    if (m_dither != 0.0f)
    {
        curandGenerateNormal(m_curand_handle, m_noise.Data(), m_noise.Dim(), 0.0 /*mean*/, 1.0 /*stddev*/);
    }

    // dither + gain + fft padding kernel
    // pad the front and back of the audio by (fft_length - window_size) / 2
    // the padding ensures the first fft window/frame is centered at 10ms spanning [0m, 20ms] of each audio buffer
    process_audio(m_processed_audio, m_audio_batch, m_noise, m_gain, m_dither, m_preemph_coeff, batch.size(), m_stream);
    // write_matrix(m_processed_audio, "processed_audio");

    // extract fft windows from audio
    // the audio data will be centered in each fft window @ [96, 416) with the weighting function applied
    // default audio window size is 120ms, which have been padded by 96 samples front and back (prev kernel)
    // to eliminate branching logic in the kernels
    extract_fft_windows(m_fft_windows, m_processed_audio, m_resources->windowing_function(), m_fft_windows_per_batch, batch.size(),
                        m_stream);
    // write_matrix(m_fft_windows, "fft_windows");

    auto lb = m_fft_plans_by_batch_size.lower_bound(batch.size());
    CHECK(lb != m_fft_plans_by_batch_size.end());

    for (int idx = 0, offset = 0; idx < batch.size(); idx += lb->first)
    {
        auto windows = m_fft_windows.Data() + m_fft_windows.Stride() * offset;
        auto bins    = reinterpret_cast<cufftComplex*>(m_fft_bins.Data() + m_fft_bins.Stride() * offset);
        CHECK_EQ(cufftExecR2C(lb->second, windows, bins), CUFFT_SUCCESS);

        offset += lb->first * m_fft_windows_per_batch;
    }
    // write_matrix(m_fft_bins, "fft_bins");

    // compute the magnitude of each of the bins in each window
    power_spectrum(m_power_spectrum, m_fft_bins, m_num_frequency_bins, batch.size(), m_fft_windows_per_batch, m_stream);
    // write_matrix(m_power_spectrum, "power_spectrum");

    // finalize new features (mels)
    mel_banks_compute(m_new_features, m_power_spectrum, m_resources->mel_banks(), batch.size(), m_fft_windows_per_batch, m_stream);
    // write_matrix(m_new_features, "mel_features");

    // scatter features raw / unnormalized features
    scatter_features(m_ctx_pointers.new_feature_buffers(), m_new_features, batch.size(), m_fft_windows_per_batch, m_new_features.NumCols(),
                     m_stream);

    // form up the full inputs for the accoustic model
    gather_normalized_features(m_acoustic_inputs, m_ctx_pointers.feature_buffer(), batch.size(), m_acoustic_inputs.NumCols(),
                               m_num_mel_bins, m_stream);

    void* bindings[2];

    bindings[0] = m_acoustic_inputs.Data();
    bindings[1] = m_acoustic_intermediates.data();
    m_encoder->context().enqueueV2(bindings, m_stream, nullptr);

    bindings[0] = m_acoustic_intermediates.data();
    bindings[1] = m_acoustic_outputs.data();
    m_decoder->context().enqueueV2(bindings, m_stream, nullptr);

    // record an event which on completion signifies that the features buffer can be shifted
    CHECK_EQ(cudaEventRecord(m_release_features, m_stream), CUDA_SUCCESS);

    //  m_accoustic_exec_ctx->enqueue(batch.size(), m_accoustic_inputs.Data(), m_stream);

    cuda_sync<typename context_t::thread_t>::event_sync(m_release_inputs);
    release_inputs();

    cuda_sync<typename context_t::thread_t>::event_sync(m_release_features);
    for (auto& b : batch)
    {
        b.features_window.release();
    }

    cuda_sync<typename context_t::thread_t>::stream_sync(m_stream);
}

#include "cnpy.h"

std::string name(const std::string filename)
{
    static std::size_t counter = 0;
    std::stringstream  ss;
    ss << filename << "_" << counter << ".npy";
    return ss.str();
}

void Pipeline::Impl::write_matrix(const CuMatrix<float>& device_matrix, const std::string filename)
{
    auto bytes = device_matrix.NumRows() * device_matrix.Stride() * sizeof(float);
    auto md    = m_resources->pinned_allocator().allocate_descriptor(bytes);
    CHECK_CUDA(cudaMemcpyAsync(md.data(), device_matrix.Data(), bytes, cudaMemcpyDeviceToHost, m_stream));
    CHECK_CUDA(cudaStreamSynchronize(m_stream));

    cnpy::npy_save(name(filename), static_cast<const float*>(md.data()), {device_matrix.NumRows(), device_matrix.NumCols()}, "w");
}