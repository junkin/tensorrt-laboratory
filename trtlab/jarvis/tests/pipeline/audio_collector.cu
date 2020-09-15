#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

#include <trtlab/core/utils.h>

#include "audio_collector.h"

using namespace trtlab;

__global__ void gather_audio_batch_kernel(gather_audio_batch_kernel_args args)
{
    int b = blockIdx.y;
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    int offset = b * args.buffer_stride + i;

    if (b < args.batch_size && i < args.window_size)
    {
        args.buffer[offset] = static_cast<float>(args.windows[b][i]) / UINT16_MAX;

        // printf("b: %d; i: %d; s: %d; o: %d; b: %lf\n", b, i, args.windows[b][i], offset, args.buffer[offset]);
    }
}

AudioBatchCollector::AudioBatchCollector(std::uint32_t batch_size, std::uint32_t window_size, memory::iallocator& alloc,
                                         std::uint32_t threads_per_block)
: m_ThreadsPerBlock(threads_per_block), m_MaxBatchSize(batch_size)
{
    CHECK_LE(batch_size, GATHER_MAX_BATCHSIZE);

    m_Args.batch_size    = 0;
    m_Args.window_size   = window_size;
    m_Args.buffer_stride = trtlab::round_up(window_size, m_ThreadsPerBlock);

    m_Descriptor  = alloc.allocate_descriptor(batch_size * m_Args.buffer_stride * sizeof(float));
    m_Args.buffer = static_cast<float*>(m_Descriptor.data());

    m_BufferBlocks = m_Args.buffer_stride / m_ThreadsPerBlock + (m_Args.buffer_stride % m_ThreadsPerBlock ? 1 : 0);

    DVLOG(4) << "window_size: " << m_Args.window_size << "; buffer_stride: " << m_Args.buffer_stride
             << "; buffer_blocks: " << m_BufferBlocks << "; thread_per_block: " << m_ThreadsPerBlock;
}

AudioBatchCollector::AudioBatchCollector(memory::descriptor&& md, std::uint32_t window_size, std::uint32_t threads_per_block)
: m_Descriptor(std::move(md)), m_ThreadsPerBlock(threads_per_block), m_MaxBatchSize(GATHER_MAX_BATCHSIZE)
{
    m_Args.batch_size    = 0;
    m_Args.window_size   = window_size;
    m_Args.buffer        = static_cast<float*>(m_Descriptor.data());
    m_Args.buffer_stride = trtlab::round_up(window_size, m_ThreadsPerBlock);

    m_BufferBlocks = m_Args.buffer_stride / m_ThreadsPerBlock + (m_Args.buffer_stride % m_ThreadsPerBlock ? 1 : 0);

    CHECK_LE(m_Descriptor.size(), m_Args.buffer_stride * sizeof(float) * GATHER_MAX_BATCHSIZE);

    DVLOG(4) << "window_size: " << m_Args.window_size << "; buffer_stride: " << m_Args.buffer_stride
             << "; buffer_blocks: " << m_BufferBlocks << "; thread_per_block: " << m_ThreadsPerBlock;
}

AudioBatchCollector::~AudioBatchCollector()
{
    if (m_Descriptor.data())
    {
        LOG(FATAL) << "destroying an incomplete batch";
    }
}

void AudioBatchCollector::add_window(std::uint16_t* window)
{
    CHECK_LT(m_Args.batch_size, m_MaxBatchSize);
    m_Args.windows[m_Args.batch_size++] = window;
}

AudioBatch AudioBatchCollector::gather(cudaStream_t stream)
{
    DCHECK(m_Descriptor.data());
    VLOG(3) << "issuing gathering kernel for " << m_Args.batch_size << " audio streams";
    dim3 blocks(m_BufferBlocks, m_Args.batch_size);
    gather_audio_batch_kernel<<<blocks, m_ThreadsPerBlock, 0, stream>>>(m_Args);
    return AudioBatch(std::move(m_Descriptor), m_Args.batch_size, m_Args.window_size, m_Args.buffer_stride);
}