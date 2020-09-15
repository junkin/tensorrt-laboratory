#pragma once

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/descriptor.h>

constexpr std::uint32_t GATHER_MAX_BATCHSIZE = 256;

struct gather_audio_batch_kernel_args
{
    int       batch_size;
    int       window_size;
    int       buffer_stride;
    float*    buffer;
    uint16_t* windows[GATHER_MAX_BATCHSIZE];
};

class AudioBatch;

class AudioBatchCollector final
{
public:
    AudioBatchCollector(std::uint32_t batch_size, std::uint32_t window_size, trtlab::memory::iallocator&,
                        std::uint32_t threads_per_block = 64);
    AudioBatchCollector(trtlab::memory::descriptor&& md, std::uint32_t window_size, std::uint32_t threads_per_block = 64);
    ~AudioBatchCollector();

    void       add_window(std::uint16_t*);
    AudioBatch gather(cudaStream_t stream);

private:
    trtlab::memory::descriptor     m_Descriptor;
    gather_audio_batch_kernel_args m_Args;
    std::uint32_t                  m_ThreadsPerBlock;
    std::uint32_t                  m_BufferBlocks;
    std::uint32_t                  m_MaxBatchSize;
};

class AudioBatch final
{
public:
    AudioBatch(trtlab::memory::descriptor&& md, std::uint32_t batch_size, std::uint32_t window_size, std::uint32_t window_stride)
    : m_Descriptor(std::move(md)), m_BatchSize(batch_size), m_WindowSize(window_size), m_WindowStride(window_stride)
    {
    }
    ~AudioBatch() {}

    AudioBatch(AudioBatch&&) noexcept = default;
    AudioBatch& operator=(AudioBatch&&) noexcept = default;

    float* data()
    {
        return static_cast<float*>(m_Descriptor.data());
    }
    const float* data() const
    {
        return static_cast<const float*>(m_Descriptor.data());
    }

    std::uint32_t sample_count() const
    {
        return m_WindowSize;
    }

    std::uint32_t batch_size() const
    {
        return m_BatchSize;
    }

    std::uint32_t stride() const
    {
        return m_WindowStride;
    }

private:
    trtlab::memory::descriptor m_Descriptor;
    std::uint32_t      m_BatchSize;
    std::uint32_t      m_WindowSize;
    std::uint32_t      m_WindowStride;
};
