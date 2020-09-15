
#include "context.h"

#include <glog/logging.h>

using Input  = SpeechSquadInferRequest;
using Output = SpeechSquadInferResponse;

using namespace demo;

void SpeechSquadContext::StreamInitialized(std::shared_ptr<ServerStream> stream)
{
    DCHECK(m_state == State::Uninitialized);
    m_state = State::Initialized;

    // asr client
    auto asr_stub       = GetResources()->asr_stub();
    auto prepare_asr_fn = [asr_stub](::grpc::ClientContext * context, ::grpc::CompletionQueue * cq) -> auto
    {
        return std::move(asr_stub->PrepareAsyncStreamingRecognize(context, cq));
    };

    m_asr_client = std::make_unique<ASRClient>(
        prepare_asr_fn, GetResources()->client_executor(), [](asr_request_t&&) {},
        [this](asr_response_t&& response) { ASRCallbackOnResponse(std::move(response)); });

    // nlp client
    auto nlp_stub       = GetResources()->nlp_stub();
    auto prepare_nlp_fn = [nlp_stub](::grpc::ClientContext * context, const nlp_request_t& request, ::grpc::CompletionQueue* cq) -> auto
    {
        return std::move(nlp_stub->PrepareAsyncNaturalQuery(context, request, cq));
    };

    m_nlp_client = std::make_unique<NLPClient>(prepare_nlp_fn, GetResources()->client_executor());

    // tts client
    auto tts_stub       = GetResources()->tts_stub();
    auto prepare_tts_fn = [tts_stub](::grpc::ClientContext * context, const tts_request_t& request, ::grpc::CompletionQueue* cq) -> auto
    {
        return std::move(tts_stub->PrepareAsyncSynthesizeOnline(context, request, cq));
    };

    auto tts_callback_response = [this](tts_response_t&& response) { TTSCallbackOnResponse(std::move(response)); };

    auto tts_callback_complete = [this](const ::grpc::Status& status) { TTSCallbackOnComplete(status); };

    m_tts_client =
        std::make_unique<TTSClient>(prepare_tts_fn, GetResources()->client_executor(), tts_callback_response, tts_callback_complete);
}

void SpeechSquadContext::RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream)
{
    DCHECK_NOTNULL(stream);

    if (input.has_speech_squad_config())
    {
        DCHECK(m_state == State::Initialized);
        m_state = State::ReceivingAudio;

        // extract the context from the initial request
        m_context = input.speech_squad_config().squad_context();

        // initialize the jarvis async asr stream with the input audio config
        DCHECK(input.speech_squad_config().input_audio_config().encoding() == AudioEncoding::LINEAR_PCM);

        // asr configure request
        asr_request_t request;
        auto          streaming_config = request.mutable_streaming_config();
        streaming_config->set_interim_results(false);
        auto config = streaming_config->mutable_config();
        config->set_encoding(nvidia::jarvis::AudioEncoding::LINEAR_PCM);
        config->set_sample_rate_hertz(input.speech_squad_config().input_audio_config().sample_rate_hertz());
        config->set_language_code(input.speech_squad_config().input_audio_config().language_code());
        config->set_audio_channel_count(input.speech_squad_config().input_audio_config().audio_channel_count());

        // save tts config for when we issue the tts request
        m_tts_config = input.speech_squad_config().output_audio_config();

        // write/send the initial request to jarvis asr
        m_asr_client->Write(std::move(request));
    }
    else
    {
        // forward audio from speech squad input to jarvis asr
        DCHECK(m_state == State::ReceivingAudio);
        asr_request_t request;
        request.set_audio_content(input.audio_content());
        m_asr_client->Write(std::move(request));
    }
}

void SpeechSquadContext::RequestsFinished(std::shared_ptr<ServerStream> stream)
{
    DCHECK(m_state == State::ReceivingAudio);
    m_state = State::AudioUploadComplete;

    // close upload to jarvis asr stream
    m_asr_client->Done();

    // save the server stream to the context so the tts callback handler can
    // forwards tts frames back the client
    m_stream = stream;
}

void SpeechSquadContext::ASRCallbackOnResponse(asr_response_t&& response)
{
    DCHECK(m_asr_client->Status().get().ok());
    DCHECK_GE(response.results_size(), 1);
    DCHECK(response.results()[0].is_final());

    m_question = response.results()[0].alternatives()[0].transcript() + "?";

    DVLOG(1) << this << ": asr complete " << std::endl
             << "q: " << m_question << "; confidence=" << response.results()[0].alternatives()[0].confidence();

    nlp_request_t request;
    request.set_context(m_context);
    request.set_query(m_question);

    m_nlp_client->Enqueue(std::move(request), [this](nlp_request_t& input, nlp_response_t& output, grpc::Status& status) {
        DCHECK(status.ok());
        NLPCallbackOnResponse(output);
    });
}

void SpeechSquadContext::NLPCallbackOnResponse(const nlp_response_t& response)
{
    m_answer    = response.results()[0].answer();
    m_nlp_score = response.results()[0].score();

    DVLOG(1) << this << ": nlp complete." << std::endl
             << "q: " << m_question << std::endl
             << "a: " << m_answer << "; score=" << m_nlp_score;

    // setup the tts request
    tts_request_t request;
    request.set_text(m_answer);
    request.set_language_code(m_tts_config.language_code());
    request.set_encoding(nvidia::jarvis::AudioEncoding::LINEAR_PCM);
    request.set_sample_rate_hz(m_tts_config.sample_rate_hertz());
    m_tts_client->Write(std::move(request));

    // write back to the squad client the initial inference meta data
    SpeechSquadInferResponse squad_response;

    auto infer_metadata = squad_response.mutable_metadata();
    infer_metadata->set_squad_question(m_question);
    infer_metadata->set_squad_answer(m_answer);
    m_stream->WriteResponse(std::move(squad_response));
}

void SpeechSquadContext::TTSCallbackOnResponse(tts_response_t&& tts_response)
{
    SpeechSquadInferResponse squad_response;
    squad_response.set_audio_content(tts_response.audio());
    m_stream->WriteResponse(std::move(squad_response));
}

void SpeechSquadContext::TTSCallbackOnComplete(const ::grpc::Status& status)
{
    m_stream->FinishStream();
}