#pragma once

#include <trtlab/memory/descriptor.h>

#include "context_impl.h"
#include "resources.h"

class ContextPointers
{
public:
    using batch_t = typename Context::dispatcher_t::batch_t;

    ContextPointers(std::uint32_t, Resources&, cudaStream_t);

    void init(const batch_t& batch);

    const std::int16_t* const* audio_buffers() const
    {
        return d_audio_buffers;
    }

    const float* const* feature_buffer() const
    {
        return d_feature_buffers;
    }

    float** new_feature_buffers()
    {
        return d_new_feature_buffers;
    }

private:
    // the following sets of pointers are initialized on init

    // pointers to each context's audio buffer window for the current batch
    // these are pinned host pointers that need to be moved to the device
    std::int16_t const** h_audio_buffers;
    std::int16_t const** d_audio_buffers;

    // pointers to each context's entire feature buffer for the current batch
    // this includes features from previous audio.  if using the defaults,
    // there are 251 total features in this buffer, 11 new from the current
    // audio window (see next ptr set) and 240 features carried over
    float const** h_feature_buffers;
    float const** d_feature_buffers;

    // pointers to the location in each context's feature buffer to store the
    // new features that will be computed during the feature extraction of the
    // current audio window.  using the defaults, 11 new features
    float** h_new_feature_buffers;
    float** d_new_feature_buffers;

    // backing pinned and device memory
    // we will allocate the single contiguous block for all ptr sets so we
    // can do a single cudaMemcpyAsync
    trtlab::memory::descriptor h_ptrs_md;
    trtlab::memory::descriptor d_ptrs_md;

    std::uint32_t m_max_batch_size;
    cudaStream_t  m_stream;
};