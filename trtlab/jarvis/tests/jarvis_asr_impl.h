#pragma once

#include <trtlab/core/pool.h>

#include "jarvis_asr.h"
#include "context_impl.h"

#include "pipeline.h"

class BaseASR : public IJarvisASR, private Context::dispatcher_t, public std::enable_shared_from_this<BaseASR>
{
public:
    using context_t          = Context;
    using dispatcher_t       = typename context_t::dispatcher_t;
    using batcher_t          = typename context_t::batcher_t;
    using batch_t            = typename context_t::dispatcher_t::batch_t;

    BaseASR(const Config&);
    virtual ~BaseASR() = default;

    DELETE_COPYABILITY(BaseASR);
    DELETE_MOVEABILITY(BaseASR);

    // IJarvisASR 
    const Config& config() const final override;
    std::unique_ptr<IContext> create_context() final override;

    Resources& resources() { return *m_Resources; }
    dispatcher_t& dispatcher() { return *this; }

protected:
    std::shared_ptr<Resources> shared_resources() { return m_Resources; }

private:
    Config                                            m_Config;
    std::shared_ptr<Resources>                        m_Resources;
};


class ASR : public BaseASR
{
    using thread_t = typename context_t::thread_t;

public:
    ASR(const Config& config);
    ~ASR() override = default;

private:
    void compute_batch_fn(const batch_t&, std::function<void()>) final override;
    
    std::shared_ptr<Pipeline> get_pipeline();
    std::shared_ptr<trtlab::Pool<Pipeline, thread_t>> m_Pipelines;
};