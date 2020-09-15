#pragma once
#include <memory>
#include <experimental/propagate_const>

#include <cuda.h>

#include "config.h"
#include "resources.h"
#include "context_impl.h"

class Pipeline final
{
    using context_t  = Context;
    using batch_t    = typename context_t::dispatcher_t::batch_t;
    using release_fn = std::function<void()>;

    class Impl;
    std::experimental::propagate_const<std::unique_ptr<Impl>> pImpl;

public:
    Pipeline(const Config&, std::shared_ptr<Resources> resources);
    ~Pipeline();

    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void compute(const batch_t&, release_fn);

    cudaStream_t stream();
};
