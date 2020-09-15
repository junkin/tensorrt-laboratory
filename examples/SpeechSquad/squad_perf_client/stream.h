/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <nvrpc/client/client_streaming_v2.h>

#include "speech_squad.grpc.pb.h"
#include "speech_squad.pb.h"

namespace speech_squad {

class Stream : public nvrpc::client::v2::ClientStreaming<
                   SpeechSquadInferRequest, SpeechSquadInferResponse> {
  using Client = nvrpc::client::v2::ClientStreaming<
      SpeechSquadInferRequest, SpeechSquadInferResponse>;

 public:
  using PrepareFn = typename Client::PrepareFn;
  using ReceiveResponseFn =
      std::function<void(SpeechSquadInferResponse&& response)>;
  using CompleteFn = std::function<void()>;

  Stream(
      PrepareFn prepare_fn, std::shared_ptr<nvrpc::client::Executor> executor,
      ReceiveResponseFn OnReceive, CompleteFn OnComplete)
      : Client(prepare_fn, executor), OnReceive_(OnReceive),
        OnComplete_(OnComplete), m_sent_count(0), m_recv_count(0)
  {
  }
  ~Stream() override {}

  void CallbackOnRequestSent(SpeechSquadInferRequest&& request) override
  {
    m_sent_count++;
  }

  void CallbackOnResponseReceived(SpeechSquadInferResponse&& response) override
  {
    m_recv_count++;
    OnReceive_(std::move(response));
  }

  void CallbackOnComplete() override { OnComplete_(); }

 private:
  ReceiveResponseFn OnReceive_;
  CompleteFn OnComplete_;

  std::size_t m_sent_count;
  std::size_t m_recv_count;
};

}  // namespace speech_squad
