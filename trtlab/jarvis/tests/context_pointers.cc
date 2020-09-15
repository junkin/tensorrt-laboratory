#include "context_pointers.h"

#include <glog/logging.h>

using namespace trtlab;

ContextPointers::ContextPointers(std::uint32_t max_batch_size, Resources& resources, cudaStream_t stream)
: m_max_batch_size(max_batch_size), m_stream(stream)
{
    h_ptrs_md = resources.pinned_allocator().allocate_descriptor(max_batch_size * sizeof(memory::addr_t*) * 3);
    d_ptrs_md = resources.device_allocator().allocate_descriptor(max_batch_size * sizeof(memory::addr_t*) * 3);
}

void ContextPointers::init(const batch_t& batch)
{
    memory::addr_t h_ptrs = reinterpret_cast<memory::addr_t>(h_ptrs_md.data());
    memory::addr_t d_ptrs = reinterpret_cast<memory::addr_t>(d_ptrs_md.data());

    CHECK_GT(batch.size(), 0);

    // host and device array to hold ptrs to each context's audio buffer
    h_audio_buffers = reinterpret_cast<std::int16_t const**>(h_ptrs);
    d_audio_buffers = reinterpret_cast<std::int16_t const**>(d_ptrs);

    // shift by batch size
    h_ptrs += batch.size() * sizeof(memory::addr_t);
    d_ptrs += batch.size() * sizeof(memory::addr_t);

    // host and device array to hold ptrs to each context's audio buffer
    h_feature_buffers = reinterpret_cast<float const**>(h_ptrs);
    d_feature_buffers = reinterpret_cast<float const**>(d_ptrs);

    // shift again
    h_ptrs += batch.size() * sizeof(memory::addr_t);
    d_ptrs += batch.size() * sizeof(memory::addr_t);

    // host and device array to hold ptrs to each context's audio buffer
    h_new_feature_buffers = reinterpret_cast<float**>(h_ptrs);
    d_new_feature_buffers = reinterpret_cast<float**>(d_ptrs);

    // initialize the pointers from the batch
    for (int i = 0; i < batch.size(); i++)
    {
        DVLOG(3) << "b: " << i << "; audio: " << static_cast<const void*>(batch[i].data)
                 << "; feats: " << static_cast<const void*>(batch[i].features_window.window_start)
                 << "; new_feats: " << static_cast<const void*>(batch[i].features_window.data_start);

        h_audio_buffers[i]       = static_cast<const std::int16_t*>(batch[i].data);
        h_feature_buffers[i]     = static_cast<const float*>(batch[i].features_window.window_start);
        h_new_feature_buffers[i] = static_cast<float*>(batch[i].features_window.data_start);
    }

    // copy the pointer values to the device
    CHECK_EQ(cudaMemcpyAsync(d_ptrs_md.data(), h_ptrs_md.data(), 3 * batch.size() * sizeof(memory::addr_t), cudaMemcpyHostToDevice,
                             m_stream),
             cudaSuccess);
}