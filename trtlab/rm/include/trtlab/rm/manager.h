#pragma once
#include <memory>
#include <experimental/propagate_const>

#include "detail/opaque_interfaces.h"

class Plugin;

class ResourceManager final
{
    struct Impl;
    std::experimental::propagate_const<std::shared_ptr<Impl>> pImpl;

public:
    ResourceManager(const Config&);
    ~ResourceManager();

    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;

    ResourceManager(ResourceManager&&) noexcept;
    ResourceManager& operator=(ResourceManager&&) noexcept;

    template <typename PluginType, typename... Args>
    std::shared_ptr<PluginType> register_plugin(Args&&... args)
    {
        static_assert(std::is_base_of<Plugin, PluginType>::value, "must dervived from Plugin");
        auto plugin = std::make_shared<PluginType>(get_manager(), std::forward<Args>(args)...);
        register_plugin(plugin);
        return plugin;
    }

private:
    std::shared_ptr<IResourceManager> get_manager();
    void register_plugin(std::shared_ptr<IManagedPlugin>);

    friend class Plugin;
};