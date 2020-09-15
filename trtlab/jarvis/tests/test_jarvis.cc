/* Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "test_jarvis.h"

#include <cstdlib>

#include <chrono>
#include <future>
#include <numeric>

#include <trtlab/core/pool.h>
#include <trtlab/core/thread_pool.h>
#include <trtlab/core/fiber_group.h>

#include "builder.h"
#include "config.h"
#include "resources.h"
#include "context.h"
#include "audio_buffer.h"

using namespace trtlab;

#include "pipeline/audio_collector.h"

template <typename T>
bool is_equal(T a, T b)
{
    std::abs(a) - std::abs(b) < 0.0001;
}

TestAudioInputs::TestAudioInputs(std::size_t count, std::uint32_t min_length_ms, std::uint32_t max_length_ms, memory::iallocator& alloc)
{
    std::srand(42);
    m_Audio.reserve(count);
    for (int i = 0; i < count; i++)
    {
        auto audio_samples = rand(min_length_ms, max_length_ms) * 16;
        m_Audio.emplace_back(alloc.allocate_descriptor(audio_samples * sizeof(std::int16_t)));
        auto signal = reinterpret_cast<std::int16_t*>(m_Audio[i].data());
        for (int j = 0; j < audio_samples; j++)
        {
            signal[j] = i + 420;
        }
    }
}

std::size_t TestAudioInputs::size() const
{
    CHECK(m_Audio.size());
    return m_Audio.size();
}

std::pair<const std::int16_t*, std::size_t> TestAudioInputs::audio_input(std::size_t id)
{
    CHECK_LT(id, m_Audio.size());
    auto audio = static_cast<const std::int16_t*>(m_Audio[id].data());
    return std::make_pair(audio, m_Audio[id].size() / sizeof(std::int16_t));
}

std::uint32_t TestAudioInputs::rand(std::uint32_t min, std::uint32_t max)
{
    std::uint32_t random = std::rand() % max + min;
    return random;
}

/*
TEST_F(TestJarvis, GatherKernelDirect)
{
    int batch_size   = 128;
    int signal_count = 16 * 120; // 12ms of 16-bit wav data

    auto pinned = memory::make_allocator(memory::cuda_malloc_host_allocator());
    auto device = memory::make_allocator(memory::cuda_malloc_allocator(0));

    cudaStream_t stream;
    CHECK_EQ(cudaStreamCreate(&stream), cudaSuccess);

    TestAudioInputs inputs(batch_size, 120, pinned);

    auto bytes          = batch_size * signal_count * sizeof(float);
    auto batched_buffer = device.allocate_descriptor(bytes);
    auto host_buffer    = pinned.allocate_descriptor(bytes);

    auto start = std::chrono::high_resolution_clock::now();

    CHECK_EQ(cudaMemsetAsync(batched_buffer.data(), 0, bytes, stream), cudaSuccess);

    AudioBatchCollector audio_batch(std::move(batched_buffer), signal_count, 64);
    for (int i = 0; i < batch_size; i++)
    {
        audio_batch.add_window(reinterpret_cast<std::int16_t*>(inputs.audio_input(i).data()));
    }
    auto output = audio_batch.gather(stream);

    CHECK_EQ(cudaMemcpyAsync(host_buffer.data(), output.data(), bytes, cudaMemcpyDeviceToHost, stream), cudaSuccess);

    CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    LOG(INFO) << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

    float* buffer = static_cast<float*>(host_buffer.data());

    for (int i = 0; i < batch_size; i++)
    {
        float val = static_cast<float>(i + 420) / UINT16_MAX;

        for (int j = 0; j < signal_count; j++)
        {
            auto offset = i * signal_count + j;
            ASSERT_FLOAT_EQ(buffer[offset], val) << "i=" << i << "; j=" << j << "; offset=" << offset;
        }
    }
}
*/
template <typename ContextType>
class TestContext
{
};

TEST_F(TestJarvis, FiberGroup)
{
    std::size_t count = 4;
    FiberGroup  fibers(count);

    using id_t = decltype(std::this_thread::get_id());
    std::set<id_t>                           uniques;
    std::vector<boost::fibers::future<id_t>> futures;

    // launch enough fibers to to spread amounst the threads
    for (int i = 0; i < count * 50; i++)
    {
        futures.emplace_back(boost::fibers::async([]() -> auto {
            boost::this_fiber::sleep_for(std::chrono::milliseconds(1));
            return std::this_thread::get_id();
        }));
    }

    for (auto& f : futures)
    {
        uniques.insert(f.get());
    }

    ASSERT_EQ(uniques.size(), count);
}

TEST_F(TestJarvis, NextPowerOf2)
{
    ASSERT_EQ(next_pow2(1), 1);
    ASSERT_EQ(next_pow2(2), 2);
    ASSERT_EQ(next_pow2(3), 4);
    ASSERT_EQ(next_pow2(4), 4);
    ASSERT_EQ(next_pow2(5), 8);
    ASSERT_EQ(next_pow2(120), 128);
}

#include "jarvis_asr.h"

TEST_F(TestJarvis, StartupAndShutdown)
{
    auto jarvis_asr = JarvisASR::Init(Config());
}

#include "context_pointers.h"

#include <trtlab/cuda/common.h>

TEST_F(TestJarvis, FeaturesUnNormalized)
{
    ConfigBuilder builder;

    // set any config options in the builder

    Config     config = builder.get_config();
    Resources  resources(config);
    FiberGroup fiber_group(config.thread_count);

    auto jarvis_asr = JarvisASR::Init(config);

    const std::size_t iterations       = 1;
    const std::size_t context_count    = 1;
    const std::size_t seconds_of_audio = 1;
    const std::size_t wav_samples      = seconds_of_audio * config.sample_freq;

    // contexts

    DVLOG(1) << "contexts initialized";

    // fake audio data
    TestAudioInputs inputs(context_count, 240, 240, resources.pinned_allocator());

    std::vector<boost::fibers::future<void>> futures;

    LOG(INFO) << "start";
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < context_count; i++)
    {
        auto [audio, samples] = inputs.audio_input(i);
        futures.push_back(boost::fibers::async([audio, samples, jarvis_asr] {
            auto ctx = jarvis_asr->create_context();
            ctx->append_audio(audio, samples);
            ctx->reset();
        }));
    }

    for (auto& f : futures)
    {
        f.get();
    }
    auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    LOG(INFO) << "complete - " << elapsed;
}

TEST_F(TestJarvis, DirectContexts)
{
    ConfigBuilder builder;

    builder.set_acoustic_encoder_path("/work/trtlab/jarvis/models/jasper_encoder.engine");
    builder.set_acoustic_decoder_path("/work/trtlab/jarvis/models/jasper_decoder.engine");

    builder.set_max_pipeline_concurrency(2);
    builder.set_max_batch_size(64);
    builder.set_batching_window_timeout_ms(100);
    // set any config options in the builder

    Config     config = builder.get_config();
    FiberGroup fiber_group(config.thread_count);

    auto jarvis_asr = JarvisASR::Init(config);

    const std::size_t iterations       = 1;
    const std::size_t context_count    = 128;
    const std::size_t seconds_of_audio = 10;
    const std::size_t wav_samples      = seconds_of_audio * config.sample_freq;

    // contexts

    std::vector<std::unique_ptr<IContext>> context_list;
    for (int i = 0; i < context_count; i++)
    {
        context_list.emplace_back(jarvis_asr->create_context());
    }

    DVLOG(1) << "contexts initialized";

    // fake audio data
    //TestAudioInputs(context_count, seconds_of_audio*1000, );

    std::vector<std::vector<std::int16_t>> wavs;
    wavs.resize(context_count);
    for (auto& wav : wavs)
    {
        wav.resize(wav_samples);
        std::fill(wav.begin(), wav.end(), 0);
    }

    std::vector<boost::fibers::future<void>> futures;

    LOG(INFO) << "start";
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < context_count; i++)
    {
        IContext& ctx = *context_list[i];
        auto&     wav = wavs[i];

        CHECK_EQ(wav.size(), wav_samples);
        futures.push_back(boost::fibers::async([&ctx, iterations, &wav, id = i] {
            for (int i = 0; i < iterations; i++)
            {
                DVLOG(3) << "context " << id << " - started iter " << i << " on thread " << std::this_thread::get_id() << " fiber "
                         << boost::this_fiber::get_id();
                ctx.append_audio(wav.data(), wav.size());
                ctx.reset();
            }
        }));
    }

    for (auto& f : futures)
    {
        f.get();
    }
    auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    LOG(INFO) << "complete - " << elapsed;
}