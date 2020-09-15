#include "jarvis_asr_impl.h"

#include "resources.h"

// Public Implementation

std::shared_ptr<IJarvisASR> JarvisASR::Init(const Config& config)
{
    return std::make_shared<ASR>(config);
}

// Private Implementation (BaseASR)

BaseASR::BaseASR(const Config& config)
: dispatcher_t(batcher_t(config.max_batch_size), std::chrono::milliseconds(config.batching_window_timeout_ms)),
  m_Config{config},
  m_Resources{std::make_shared<Resources>(config)}
{
}

const Config& BaseASR::config() const
{
    return m_Config;
}

std::unique_ptr<IContext> BaseASR::create_context()
{
    return std::make_unique<Context>(shared_from_this());
}

// Private Implementation (BaseASR)

ASR::ASR(const Config& config)
: BaseASR(config),
  m_Pipelines{trtlab::Pool<Pipeline, thread_t>::Create()}
{
    VLOG(1) << "creating " << config.max_concurrency << " execution pipelines";
    for (int i = 0; i < config.max_concurrency; i++)
    {
        m_Pipelines->EmplacePush(config, shared_resources());
    }
}

std::shared_ptr<Pipeline> ASR::get_pipeline()
{
    return m_Pipelines->Pop();
}

void ASR::compute_batch_fn(const batch_t& batch, std::function<void()> release_inputs)
{
    auto pipeline = this->get_pipeline();
    pipeline->compute(batch, release_inputs);
}