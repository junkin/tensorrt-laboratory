#pragma once

#include <trtlab/core/batcher.h>
#include <trtlab/core/dispatcher.h>
#include <trtlab/core/cyclic_windowed_buffer.h>
#include <trtlab/cuda/cyclic_windowed_buffer.h>

#include "context.h"

#include "audio_buffer.h"

class BaseASR;

class Context : public IContext, private AudioBuffer
{
    struct AudioWindow;

public:
    using audio_buffer_t    = AudioBuffer;
    using thread_t          = typename audio_buffer_t::thread_t;
    using features_buffer_t = trtlab::cyclic_windowed_reserved_stack<trtlab::memory::device_memory, thread_t>;
    using batcher_t         = trtlab::StandardBatcher<AudioWindow, thread_t>;
    using dispatcher_t      = trtlab::Dispatcher<batcher_t>;

    Context(std::shared_ptr<BaseASR>);

    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ~Context();

    void append_audio(const std::int16_t* audio, std::size_t sample_count) final override;

    void reset() final override;

private:
    using shared_future_t = typename thread_t::template shared_future<void>;
    using reservation_t   = typename features_buffer_t::reservation;

    struct AudioWindow
    {
        std::size_t   id;
        const void*   data;
        std::size_t   bytes;
        reservation_t features_window;
    };

    // this method will be called by the audio buffer everytime a window is filled
    shared_future_t on_compute_window(std::size_t window_id, const void* data, std::size_t size) final override;

    std::unique_ptr<features_buffer_t> m_Features;
    std::shared_ptr<BaseASR>           m_ASR;
};