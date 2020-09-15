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

#include "status.h"
#include "stream.h"
#include "utils.h"

namespace speech_squad {
struct Results {
  Results() : first_response(true) {}
  /*** SpeechSquadResponseMeta : Start ***
  // mandatory
  string squad_question = 1;
  string squad_answer = 2;

  // optional
  float squad_confidence = 10;
  string asr_transcription = 11;
  string asr_confidence = 12;
  map<string, float> component_timing = 13;
  *** SpeechSquadResponseMeta : End ***/
  std::string squad_question;
  std::string squad_answer;

  std::vector<std::string> audio_content;

  // Record of statistics
  // The latency between sending the last response and receipt of first response
  // in milliseconds.
  double response_latency;
  std::chrono::time_point<std::chrono::high_resolution_clock>
      last_response_timestamp;
  // Records the time interval between successive responses in milliseconds.
  std::vector<double> response_intervals;

  bool first_response;
  std::mutex mtx;
};

class AudioTask {
 public:
  // The step of processing that the AudioTask is in.
  typedef enum { START, SENDING, SENDING_COMPLETE, RECEIVING_COMPLETE } State;

  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

  AudioTask(
      const std::shared_ptr<AudioData>& audio_data, const uint32_t _corr_id,
      const Stream::PrepareFn& infer_prepare_fn,
      const std::string& language_code, const int32_t chunk_duration_ms,
      const bool print_results,
      std::shared_ptr<SquadEvalDataset> squad_eval_dataset,
      std::shared_ptr<std::ofstream>& output_file,
      std::shared_ptr<std::mutex>& output_file_mtx,
      std::shared_ptr<nvrpc::client::Executor>& executor,
      const TimePoint& start_time);

  TimePoint& NextTimePoint() { return next_time_point_; }
  State GetState() { return state_; }
  double AudioProcessed() { return audio_processed_; }
  std::shared_ptr<Results> GetResult() { return result_; }

  Status Step();
  Status WaitForCompletion();

 private:
  void ReceiveResponse(SpeechSquadInferResponse&& response);
  void FinalizeTask();

  SpeechSquadInferRequest request_;

  std::shared_ptr<AudioData> audio_data_;
  size_t offset_;
  uint32_t corr_id_;
  std::string language_code_;
  int32_t chunk_duration_ms_;
  bool print_results_;
  std::shared_ptr<SquadEvalDataset> squad_eval_dataset_;

  std::shared_ptr<std::ofstream> output_file_;
  std::shared_ptr<std::mutex> output_file_mtx_;

  std::unique_ptr<Stream> stream_;
  std::shared_future<::grpc::Status> stream_future_;
  std::shared_ptr<nvrpc::client::Executor> executor_;

  // Marks the timepoint for the next activity
  TimePoint next_time_point_;
  // Records the timestamp of the last send activity
  TimePoint send_time_;
  // The bytes of audio data to be sent in the next step
  double bytes_to_send_;
  // The total audio processed by this task in seconds
  double audio_processed_;

  // Holds the results of the transaction
  std::shared_ptr<Results> result_;
  // Current state of the task
  State state_;
};

}  // namespace speech_squad
