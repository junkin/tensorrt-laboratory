#include "test_jarvis.h"

#include <trtlab/core/fiber_group.h>
#include <trtlab/cuda/common.h>

#include "jarvis_asr_impl.h"
#include "context_pointers.h"
#include "pipeline/gather_audio_batch.h"

using namespace trtlab;

class TestASR : public BaseASR
{
public:
    TestASR(const Config& config, std::function<void(const batch_t&, std::function<void()>)> batch_fn)
    : BaseASR(config), m_batch_fn(batch_fn)
    {
    }

protected:
    void compute_batch_fn(const batch_t& batch, std::function<void()> release) override
    {
        m_batch_fn(batch, release);
    }

    std::function<void(const batch_t&, std::function<void()>)> m_batch_fn;
};

TEST_F(TestJarvis, gather_audio_batch)
{
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ConfigBuilder builder;
    builder.set_max_thread_count(2);
    builder.set_max_batch_size(32);
    builder.set_batching_window_timeout_ms(1);

    Config          config = builder.get_config();
    Resources       resources(config);
    FiberGroup      fibers(config.thread_count);
    ContextPointers ctx_pointers(config.max_batch_size, resources, stream);
    CuMatrix<float> batched_audio(stream);

    // size according to max batch size x size of the audio buffer window + 1sample
    batched_audio.Resize(config.max_batch_size, config.audio_buffer_preprocessed_window_size_samples, kUndefined);

    // allocate some space to copy the batched buffer back to the host for checking
    auto host_batch = resources.pinned_allocator().allocate_descriptor(batched_audio.NumRows() * batched_audio.Stride() * sizeof(float));

    using batch_t = typename TestASR::batch_t;

    auto on_batch = [&ctx_pointers, &batched_audio, &host_batch, stream](const batch_t& batch, std::function<void()> release) {
        VLOG(1) << "batch_size: " << batch.size();

        // collect ptrs from batch, xfer to device, then perform the gather
        ctx_pointers.init(batch);
        gather_audio_batch(batched_audio, ctx_pointers.audio_buffers(), batch.size(), stream);

        // copy batched audio buffer from device back to host and sync 
        CHECK_CUDA(cudaMemcpyAsync(host_batch.data(), batched_audio.Data(), host_batch.size(), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // check results
        const float* host = static_cast<const float*>(host_batch.data());

        for (int i = 0; i < batch.size(); i++)
        {
            std::size_t  offset = batched_audio.Stride() * i;
            const float* row    = host + offset;

            // input audio in this test is a const value per audio stream
            auto audio    = static_cast<const std::int16_t*>(batch[i].data);

            for (int j = 0; j < batched_audio.NumCols(); j++)
            {
                ASSERT_FLOAT_EQ(static_cast<float>(audio[j]) / INT16_MAX, row[j]);
            }
        }

        // allow the audio buffer windows to be reused
        release();

        // release the reservation on the features buffer
        // normally we would scatter the newly computed features back to the individual
        // contexts' state machines, but for this test, we are only testing the correctness
        // of the batcher
        for (auto& b : batch)
        {
            b.features_window.release();
        }
    };

    // create random wav files with length between 240ms and 12000ms
    // each "wav" has a const std::int16_t value
    TestAudioInputs inputs(config.max_batch_size, 240, 12000, resources.pinned_allocator());

    // Create Test ASR class - Overridding BaseASR::compute_batch_fn with on_batch lambda
    auto test_asr = std::make_shared<TestASR>(config, on_batch);

    // each fiber will pump one audio stream thru the engine
    // vector of futures for the completion of each fiber
    std::vector<boost::fibers::future<void>> futures;

    VLOG(1) << "start";
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < inputs.size(); i++)
    {
        auto [audio, samples] = inputs.audio_input(i);
        futures.push_back(boost::fibers::async([audio, samples, test_asr, i] {
            VLOG(2) << "start " << i;
            auto ctx = test_asr->create_context();
            ctx->append_audio(audio, samples);
            ctx->reset();
            VLOG(2) << "finished " << i;
        }));
    }

    for (auto& f : futures)
    {
        f.get();
    }
    auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    VLOG(1) << "complete - " << elapsed;
}