#pragma once

#include <gtest/gtest.h>

#include <nvrpc/client/client_streaming_v2.h>

#include "speech_squad.grpc.pb.h"
#include "speech_squad.pb.h"

namespace demo
{
    namespace testing
    {
        class TestClientStreaming : public nvrpc::client::v2::ClientStreaming<SpeechSquadInferRequest, SpeechSquadInferResponse>
        {
            using Client = nvrpc::client::v2::ClientStreaming<SpeechSquadInferRequest, SpeechSquadInferResponse>;

        public:
            using PrepareFn = typename Client::PrepareFn;

            TestClientStreaming(PrepareFn prepare_fn, std::shared_ptr<nvrpc::client::Executor> executor)
            : Client(prepare_fn, executor), m_audio_messages(10), m_sent_count(0), m_recv_count(0)
            {
            }
            ~TestClientStreaming() override {}

            void CallbackOnRequestSent(SpeechSquadInferRequest&&) override
            {
                m_sent_count++;
            }

            void CallbackOnResponseReceived(SpeechSquadInferResponse&&) override
            {
                m_recv_count++;
            }

            void CallbackOnComplete() override
            {
                EXPECT_EQ(m_audio_messages + 1, m_sent_count);
                EXPECT_EQ(m_audio_messages + 1, m_recv_count);
            }

            std::size_t count() const { return m_audio_messages;  }

        private:
            std::size_t m_audio_messages;
            std::size_t m_sent_count;
            std::size_t m_recv_count;
        };
    } // namespace testing
} // namespace demo