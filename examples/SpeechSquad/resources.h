#pragma once
#include <memory>

#include "settings.h"

#include <trtlab/core/resources.h>
#include <trtlab/core/pool.h>

#include <nvrpc/client/executor.h>

#include "jarvis_asr.grpc.pb.h"
#include "jarvis_asr.pb.h"

#include "jarvis_nlp.grpc.pb.h"
#include "jarvis_nlp.pb.h"

#include "jarvis_tts.grpc.pb.h"
#include "jarvis_tts.pb.h"

namespace demo
{
    template <typename ResourceType>
    using Pool = trtlab::UniquePool<ResourceType, thread_t>;

    class SpeechSquadResources : public ::trtlab::Resources
    {

    public:
        SpeechSquadResources(std::string asr_url, std::string nlp_url, std::string tts_url);
        ~SpeechSquadResources() override;

        std::shared_ptr<nvrpc::client::Executor> client_executor() { return m_client_executor; }
        std::shared_ptr<nvidia::jarvis::asr::JarvisASR::Stub> asr_stub() { return m_asr_stub; }
        std::shared_ptr<nvidia::jarvis::nlp::JarvisNLP::Stub> nlp_stub() { return m_nlp_stub; }
        std::shared_ptr<nvidia::jarvis::tts::JarvisTTS::Stub> tts_stub() { return m_tts_stub; }

    private:
        std::shared_ptr<nvrpc::client::Executor>              m_client_executor;
        std::shared_ptr<nvidia::jarvis::asr::JarvisASR::Stub> m_asr_stub;
        std::shared_ptr<nvidia::jarvis::nlp::JarvisNLP::Stub> m_nlp_stub;
        std::shared_ptr<nvidia::jarvis::tts::JarvisTTS::Stub> m_tts_stub;
    };

} // namespace demo