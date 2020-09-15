#pragma once
#include <memory>

#include "config.h"
#include "context.h"

struct IJarvisASR
{
    virtual const Config& config() const = 0;
    virtual std::unique_ptr<IContext> create_context() = 0;
};

struct JarvisASR final
{
    static std::shared_ptr<IJarvisASR> Init(const Config& config);
};