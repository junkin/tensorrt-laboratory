#pragma once

#include <trtlab/memory/memory_type.h>

#include <trtlab/core/userspace_threads.h>
#include <trtlab/core/cyclic_windowed_buffer.h>

#include "config.h"
#include "resources.h"

// AudioBuffer is a cyclic windowed buffer in pinned host memory which issues windows to the ASR pipeline
// for processing.  The window size for the preprocessed audio is one sample larger than the window size
// after processing to prevent a discontinuity in the preemph calculation. This buffer will issues windows
// that are +1 sample larging in both window size and overlap size.

class AudioBuffer : public trtlab::cyclic_windowed_task_executor<trtlab::memory::host_memory, trtlab::userspace_threads>
{
public:
    using thread_t        = trtlab::userspace_threads;
    using stack_t         = trtlab::cyclic_windowed_stack<trtlab::memory::host_memory, thread_t>;
    using task_executor_t = trtlab::cyclic_windowed_task_executor<trtlab::memory::host_memory, thread_t>;

    AudioBuffer(const Config& cfg, Resources& resources) : task_executor_t(make_stack(cfg, resources)) {}
    AudioBuffer(AudioBuffer&&) noexcept = default;
    AudioBuffer& operator=(AudioBuffer&&) noexcept = default;
    ~AudioBuffer() override {}

private:
    stack_t make_stack(const Config& cfg, Resources& resources)
    {
        const std::size_t count   = cfg.audio_buffer_window_count;
        const std::size_t size    = cfg.audio_buffer_preprocessed_window_size_bytes;
        const std::size_t overlap = cfg.audio_buffer_preprocessed_window_overlap_bytes;

        auto bytes = trtlab::cyclic_windowed_buffer::min_allocation_size(count, size, overlap);
        auto md    = resources.pinned_allocator().allocate_descriptor(bytes);
        return stack_t(std::move(md), size, overlap);
    }
};
