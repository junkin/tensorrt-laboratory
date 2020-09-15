#pragma once

struct IMemoryResources
{
    virtual memory::iallocator& persistent_host_allocator() = 0;
    virtual memory::iallocator& persistent_pinned_allocator() = 0;
    virtual memory::iallocator& persistent_device_allocator() = 0;

    virtual memory::iallocator& temporary_host_allocator() = 0;
    virtual memory::iallocator& temporary_pinned_allocator() = 0;
    virtual memory::iallocator& temporary_device_allocator() = 0;
}