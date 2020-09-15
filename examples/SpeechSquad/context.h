

/* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#include <memory>

#include <nvrpc/context.h>
#include <nvrpc/client/client_unary.h>
#include <nvrpc/client/client_streaming.h>
#include <nvrpc/client/client_single_up_multiple_down.h>

#include "resources.h"

#include "speech_squad.grpc.pb.h"
#include "speech_squad.pb.h"

#include "jarvis_asr.grpc.pb.h"
#include "jarvis_asr.pb.h"

#include "jarvis_nlp.grpc.pb.h"
#include "jarvis_nlp.pb.h"

#include "jarvis_tts.grpc.pb.h"
#include "jarvis_tts.pb.h"

namespace demo
{
    class SpeechSquadContext final : public nvrpc::StreamingContext<SpeechSquadInferRequest, SpeechSquadInferResponse, SpeechSquadResources>
    {
        void StreamInitialized(std::shared_ptr<ServerStream>) final override;
        void RequestsFinished(std::shared_ptr<ServerStream>) final override;

        void RequestReceived(SpeechSquadInferRequest&& input, std::shared_ptr<ServerStream> stream) final override;

        enum class State
        {
            Uninitialized,
            Initialized,
            ReceivingAudio,
            AudioUploadComplete
        };

        using asr_request_t  = nvidia::jarvis::asr::StreamingRecognizeRequest;
        using asr_response_t = nvidia::jarvis::asr::StreamingRecognizeResponse;

        using nlp_request_t  = nvidia::jarvis::nlp::NaturalQueryRequest;
        using nlp_response_t = nvidia::jarvis::nlp::NaturalQueryResponse;

        using tts_request_t  = nvidia::jarvis::tts::SynthesizeSpeechRequest;
        using tts_response_t = nvidia::jarvis::tts::SynthesizeSpeechResponse;

        // callbacks
        void ASRCallbackOnResponse(asr_response_t&&);
        void NLPCallbackOnResponse(const nlp_response_t&);
        void TTSCallbackOnResponse(tts_response_t&&);
        void TTSCallbackOnComplete(const ::grpc::Status&);

        // state variables
        State       m_state;
        std::string m_context;
        std::string m_question;
        std::string m_answer;
        float       m_nlp_score;
        AudioConfig m_tts_config;

        // store access to the response stream
        std::shared_ptr<ServerStream> m_stream;

        // clients
        using ASRClient = nvrpc::client::ClientStreaming<asr_request_t, asr_response_t>;
        using NLPClient = nvrpc::client::ClientUnary<nlp_request_t, nlp_response_t>;
        using TTSClient = nvrpc::client::ClientSUMD<tts_request_t, tts_response_t>;

        std::unique_ptr<ASRClient> m_asr_client;
        std::unique_ptr<NLPClient> m_nlp_client;
        std::unique_ptr<TTSClient> m_tts_client;
    };

} // namespace demo