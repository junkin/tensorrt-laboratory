#include "manager.h"

#include <unordered_set>

#include "private_interfaces.h"

// Private Interface

struct ResourceManager::Impl : public IResourceManager
{
    Impl(const Config&);
    virtual ~Impl() = default;

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    Impl(Impl&&) noexcept = delete;
    Impl& operator=(Impl&&) noexcept = delete;

    std::unordered_set<std::shared_ptr<IManagedPlugin>> plugins;
};

// Public Implementation

ResourceManager::ResourceManager(const Config& config) : pImpl(std::make_shared<ResourceManager::Impl>(config)) {}

ResourceManager::~ResourceManager() = default;

ResourceManager::ResourceManager(ResourceManager&&) noexcept = default;
ResourceManager& ResourceManager::operator=(ResourceManager&&) noexcept = default;

std::shared_ptr<IResourceManager> ResourceManager::get_manager()
{
    return pImpl;
}

void ResourceManager::register_plugin(std::shared_ptr<IManagedPlugin> plugin)
{
    auto [it, rc] = pImpl->plugins.insert(plugin);
    if(!rc)
    {
        throw std::runtime_error("unable to register plugin (tood: add identifer here)");
    }
}

/*
ManagerImpl::ManagerImpl()
: m_device_count(cuda::nvml::device_count())
{
    float       device_fraction = 0.9;
    std::size_t pinned_size     = 2_GiB;

    m_persistent_host_allocator = memory::make_allocator(memory::malloc_allocator());

    m_persistent_pinned_allocator = make_allocator({pinned_size}, memory::cuda_malloc_host_allocator());
    VLOG(1) << "Initializing Pinned Host Allocator with " << memory::bytes_to_string(pinned_size);
    
    m_persistent_device_allocators.resize(m_device_count);

    for (int i = 0; i < m_device_count(); i++)
    {
        DCHECK_GT(device_fraction, 0.);
        DCHECK_LE(device_fraction, 1.);

        auto info = cuda::nvml::memory_info(i);
        auto  bytes = std::size_t(device_fraction * info.total);
        if (bytes < info.total)
        {
            LOG(WARNING) << "Unable to acquire " << device_fraction * 100 << "% of " << memory::bytes_to_string(info.bytes);
            bytes = std::size_t(p * info.free);
        }
        m_persistent_device_allocators[i] = make_allocator({bytes}, memory::cuda_malloc_allocator(i));
        VLOG(1) << "Initializing Device Allocator for GPU " << i << " with " << memory::bytes_to_string(bytes);
    }
}
*/