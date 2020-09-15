
#include "test_context.h"

#include <glog/logging.h>

using Input  = SpeechSquadInferRequest;
using Output = SpeechSquadInferResponse;

using namespace demo;
using namespace testing;

void SpeechSquadTextContext::StreamInitialized(std::shared_ptr<ServerStream> stream)
{
    m_audio_received_count = 0;
}

void SpeechSquadTextContext::RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream)
{
    DCHECK_NOTNULL(stream);

    if (input.has_speech_squad_config())
    {
        DCHECK(input.speech_squad_config().input_audio_config().encoding() == AudioEncoding::LINEAR_PCM);
        DLOG(INFO) << "config received";
    }
    else
    {
        // forward audio from speech squad input to jarvis asr
        m_audio_received_count++;
        DLOG(INFO) << "audio received; count=" << m_audio_received_count;
    }
}

void SpeechSquadTextContext::RequestsFinished(std::shared_ptr<ServerStream> stream)
{
    DLOG(INFO) << "client signaled it will stop sending requests";
    DCHECK_NOTNULL(stream);

    SpeechSquadInferResponse squad_response;
    auto                     infer_metadata = squad_response.mutable_metadata();
    infer_metadata->set_squad_question("what is two plus two?");
    infer_metadata->set_squad_answer("the answer is four.");

    DLOG(INFO) << "writing infer meta data";
    stream->WriteResponse(std::move(squad_response));

    static char bytes[128];
    std::memset(bytes, 0, 128);

    DLOG(INFO) << "writing audio data";
    for (int i = 0; i < m_audio_received_count; i++)
    {
        SpeechSquadInferResponse squad_response;
        squad_response.set_audio_content(bytes, 128);
        stream->WriteResponse(std::move(squad_response));
    }

    DLOG(INFO) << "server transaction complete";
    stream->FinishStream();
}
