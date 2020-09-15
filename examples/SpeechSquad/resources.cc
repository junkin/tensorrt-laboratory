#include <chrono>
#include <grpcpp/impl/codegen/channel_interface.h>

#include "resources.h"

using namespace demo;

#include "jarvis_asr.grpc.pb.h"
#include "jarvis_asr.pb.h"

bool WaitUntilReady(std::shared_ptr<::grpc::ChannelInterface> channel)
{
    std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(10000);

    auto state = channel->GetState(true);
    while (state != GRPC_CHANNEL_READY)
    {
        if (!channel->WaitForStateChange(state, deadline))
        {
            return false;
        }
        state = channel->GetState(true);
    }
    return true;
}

SpeechSquadResources::SpeechSquadResources(std::string asr_url, std::string nlp_url, std::string tts_url)
: m_client_executor(std::make_shared<nvrpc::client::Executor>(1))
{
    auto asr_channel = grpc::CreateChannel(asr_url, grpc::InsecureChannelCredentials());
    m_asr_stub       = nvidia::jarvis::asr::JarvisASR::NewStub(asr_channel);

    auto nlp_channel = grpc::CreateChannel(nlp_url, grpc::InsecureChannelCredentials());
    m_nlp_stub       = nvidia::jarvis::nlp::JarvisNLP::NewStub(nlp_channel);

    auto tts_channel = grpc::CreateChannel(tts_url, grpc::InsecureChannelCredentials());
    m_tts_stub       = nvidia::jarvis::tts::JarvisTTS::NewStub(tts_channel);

    DLOG(INFO) << "establishing connections to downstream jarvis services";

    CHECK(WaitUntilReady(asr_channel)) << "failed to connect to " << asr_url;
    CHECK(WaitUntilReady(nlp_channel)) << "failed to connect to " << nlp_url;
    CHECK(WaitUntilReady(tts_channel)) << "failed to connect to " << tts_url;

    LOG(INFO) << "jarvis asr connection established to "<< asr_url;
    LOG(INFO) << "jarvis nlp connection established to "<< nlp_url;
    LOG(INFO) << "jarvis tts connection established to "<< tts_url;
}

SpeechSquadResources::~SpeechSquadResources() {}