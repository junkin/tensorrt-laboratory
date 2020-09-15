#include "context.h"
#include "context_impl.h"

#include <trtlab/cuda/common.h>

#include "jarvis_asr_impl.h"

namespace
{
    using thread_t          = typename Context::audio_buffer_t::thread_t;
    using promise_t         = typename thread_t::template promise<void>;
    using shared_future_t   = typename thread_t::template shared_future<void>;
    using device_stack_t    = trtlab::cyclic_windowed_stack<trtlab::memory::device_memory, thread_t>;
    using features_buffer_t = trtlab::cyclic_windowed_reserved_stack<trtlab::memory::device_memory, thread_t>;

    std::unique_ptr<features_buffer_t> make_features_buffer(const Config& cfg, Resources& resources)
    {
        const std::size_t count   = cfg.features_buffer_window_count;
        const std::size_t size    = cfg.features_buffer_window_size_bytes;
        const std::size_t overlap = cfg.features_buffer_window_overlap_bytes;

        auto bytes = trtlab::cyclic_windowed_buffer::min_allocation_size(count, size, overlap);
        auto md    = resources.device_allocator().allocate_descriptor(bytes);
        auto stack = device_stack_t(std::move(md), size, overlap, cudaStreamPerThread);
        return std::make_unique<features_buffer_t>(std::move(stack));
    };

} // namespace

// Private Implementation

Context::Context(std::shared_ptr<BaseASR> asr)
: audio_buffer_t(asr->config(), asr->resources()),
  m_Features(make_features_buffer(asr->config(), asr->resources())),
  m_ASR(asr)
{
}

Context::~Context()                  = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;

void Context::append_audio(const std::int16_t* audio, std::size_t sample_count)
{
    audio_buffer_t::append_data(audio, sample_count * sizeof(std::int16_t));
}

void Context::reset()
{
    audio_buffer_t::reset();
    m_Features->reset();
}

shared_future_t Context::on_compute_window(std::size_t window_id, const void* data, std::size_t size)
{
    DCHECK(m_Features);
    DCHECK(m_ASR);

    auto reservation = m_Features->reserve_window();
    return m_ASR->dispatcher().enqueue({window_id, data, size, std::move(reservation)});
}