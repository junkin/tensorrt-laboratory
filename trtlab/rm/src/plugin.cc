#include "plugin.h"

#include "private_interfaces.h"

// Private Interface

struct Plugin::Impl : public IManagedPlugin
{
    Impl(std::shared_ptr<IResourceManager> rm);
    virtual ~Impl() = default;

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    Impl(Impl&&) noexcept = delete;
    Impl& operator=(Impl&&) noexcept = delete;

    std::shared_ptr<IResourceManager> manager;
};

// Public Implementation

Plugin::Plugin(std::shared_ptr<IResourceManager> manager) : pImpl(std::make_shared<Plugin::Impl>(manager)) {}

Plugin::~Plugin() = default;
Plugin::Plugin(Plugin&&) noexcept = default;
Plugin& Plugin::operator=(Plugin&&) noexcept = default;

// Private Implementation

Plugin::Impl::Impl(std::shared_ptr<IResourceManager> rm) : manager(rm) {}