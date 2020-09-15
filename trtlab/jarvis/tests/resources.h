#pragma once
#include <memory>

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/block_allocators.h>
#include <trtlab/memory/literals.h>
#include <trtlab/memory/transactional_allocator.h>
#include <trtlab/memory/malloc_allocator.h>

#include <trtlab/cuda/memory/cuda_allocators.h>

#include <trtlab/tensorrt/runtime.h>

#include "config.h"
#include "cuda_mel_banks.h"
#include "pipeline/extract_fft_windows.h"

class Resources
{
public:
    Resources(const Config& cfg)
    : m_windowing_function(std::make_unique<CudaWindowingFunction>(cfg)),
      m_mel_banks(std::make_unique<CudaMelBanks>(cfg)),
      m_trt_runtime(std::make_shared<trtlab::TensorRT::StandardRuntime>())
    {
        using namespace trtlab::memory;
        using namespace trtlab::memory::literals;

        auto pinned = make_allocator(cuda_malloc_host_allocator());
        auto device = make_allocator(cuda_malloc_allocator(0));

        m_pinned_allocator = pinned.shared();
        m_device_allocator = device.shared();

        auto raw     = make_allocator(cuda_malloc_allocator(0));
        auto adapter = make_allocator_adapter(std::move(raw));
        auto block   = make_block_allocator<fixed_size_block_allocator>(std::move(adapter), 128_MiB);
        auto counted = make_extended_block_allocator<count_limited_block_allocator>(std::move(block), 3);
        auto arena   = make_cached_block_arena(std::move(counted));
        auto txalloc = make_transactional_allocator(std::move(arena));
        auto alloc   = make_allocator(std::move(txalloc));

        m_device_tmp_allocator = alloc.shared();
    }

    trtlab::memory::iallocator& pinned_allocator()
    {
        return *m_pinned_allocator;
    }
    trtlab::memory::iallocator& device_allocator()
    {
        return *m_device_allocator;
    }
    trtlab::memory::iallocator& device_tmp_allocator()
    {
        return *m_device_allocator;
    }

    const CudaWindowingFunction& windowing_function() const
    {
        return *m_windowing_function;
    }

    const CudaMelBanks& mel_banks() const
    {
        return *m_mel_banks;
    }

    trtlab::TensorRT::StandardRuntime& trt_runtime()
    {
        return *m_trt_runtime;
    }

private:
    std::shared_ptr<trtlab::memory::iallocator>        m_pinned_allocator;
    std::shared_ptr<trtlab::memory::iallocator>        m_device_allocator;
    std::shared_ptr<trtlab::memory::iallocator>        m_device_tmp_allocator;
    std::unique_ptr<CudaWindowingFunction>             m_windowing_function;
    std::unique_ptr<CudaMelBanks>                      m_mel_banks;
    std::shared_ptr<trtlab::TensorRT::StandardRuntime> m_trt_runtime;
};
