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

#include "audio_task.h"

namespace speech_squad {

AudioTask::AudioTask(
    const std::shared_ptr<AudioData>& audio_data, const uint32_t corr_id,
    const Stream::PrepareFn& infer_prepare_fn, const std::string& language_code,
    const int32_t chunk_duration_ms, const bool print_results,
    std::shared_ptr<speech_squad::SquadEvalDataset> squad_eval_dataset,
    std::shared_ptr<std::ofstream>& output_file,
    std::shared_ptr<std::mutex>& output_file_mtx,
    std::shared_ptr<nvrpc::client::Executor>& executor,
    const TimePoint& start_time)
    : audio_data_(audio_data), offset_(0), corr_id_(corr_id),
      language_code_(language_code), chunk_duration_ms_(chunk_duration_ms),
      print_results_(print_results), squad_eval_dataset_(squad_eval_dataset),
      output_file_(output_file), output_file_mtx_(output_file_mtx),
      next_time_point_(start_time), audio_processed_(0.), state_(START)
{
  // Prepare the server stream to be used with the transaction
  stream_ = std::make_unique<Stream>(
      infer_prepare_fn, executor,
      [this](SpeechSquadInferResponse&& response) {
        ReceiveResponse(std::move(response));
      },
      [this]() { FinalizeTask(); });
  stream_->SetCorked(true);
  result_ = std::make_shared<Results>();
}

Status
AudioTask::Step()
{
  if (state_ == SENDING_COMPLETE) {
    return Status(
        Status::Code::INTERNAL, "Cannot step further from sending complete");
  }

  // Every step will overwrite this time stamp. The responses will be
  // delivered once sending is complete. At the time of the first response
  // this timestamp will carry the timestamp of the last request.
  send_time_ = std::chrono::high_resolution_clock::now();

  // TODO: Can colllect the delay in scheduling to report the quality
  if (state_ == START) {
    // Send the configuration if at the first step
    auto speech_squad_config = request_.mutable_speech_squad_config();

    // Input Audio Configuration
    speech_squad_config->mutable_input_audio_config()->set_encoding(
        audio_data_->encoding);
    speech_squad_config->mutable_input_audio_config()->set_sample_rate_hertz(
        audio_data_->sample_rate);
    speech_squad_config->mutable_input_audio_config()->set_language_code(
        language_code_);
    speech_squad_config->mutable_input_audio_config()->set_audio_channel_count(
        audio_data_->channels);

    // Ouput Audio Configuration
    speech_squad_config->mutable_output_audio_config()->set_encoding(
        LINEAR_PCM);
    speech_squad_config->mutable_output_audio_config()->set_sample_rate_hertz(
        22050);
    speech_squad_config->mutable_output_audio_config()->set_language_code(
        "en-US");

    auto status = squad_eval_dataset_->GetQuestionContext(
        audio_data_->question_id, speech_squad_config->mutable_squad_context());
    return status;

    stream_->Write(std::move(request_));
    state_ = SENDING;
  } else {
    // Send the audio content if not the first step
    request_.set_audio_content(&audio_data_->data[offset_], bytes_to_send_);
    offset_ += bytes_to_send_;
    stream_->Write(std::move(request_));
  }

  // Set and schedule the next chunk
  size_t chunk_size =
      (audio_data_->sample_rate * chunk_duration_ms_ / 1000) * sizeof(int16_t);
  size_t header_size = (offset_ == 0) ? sizeof(FixedWAVHeader) : 0;
  bytes_to_send_ =
      std::min(audio_data_->data.size() - offset_, chunk_size + header_size);

  // Transition to the sending completion if no more bytes to send
  if (bytes_to_send_ == 0) {
    stream_future_ = stream_->Done();
    state_ = SENDING_COMPLETE;
  } else {
    double current_wait_time = 1000 * (bytes_to_send_ - header_size) /
                               (sizeof(int16_t) * audio_data_->sample_rate);
    // Accumulate the audio content processed
    audio_processed_ += current_wait_time / 1000.;
    next_time_point_ +=
        std::chrono::microseconds((int)current_wait_time * 1000);
  }

  return Status::Success;
}

Status
AudioTask::WaitForCompletion()
{
  auto grpc_status = stream_future_.get();
  state_ = RECEIVING_COMPLETE;
  return Status(grpc_status.error_message());
}

void
AudioTask::ReceiveResponse(SpeechSquadInferResponse&& response)
{
  auto now = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(result_->mtx);
  if (result_->first_response) {
    result_->response_latency =
        std::chrono::duration<double, std::milli>(now - send_time_).count();
  } else {
    result_->response_intervals.push_back(
        std::chrono::duration<double, std::milli>(
            now - result_->last_response_timestamp)
            .count());
  }
  result_->last_response_timestamp = now;

  if (response.has_metadata()) {
    result_->squad_question = response.metadata().squad_question();
    result_->squad_answer = response.metadata().squad_answer();
  } else {
    // TODO: Store the audio content in a more efficient data buffer.
    result_->audio_content.push_back(response.audio_content());
  }
}

void
AudioTask::FinalizeTask()
{
  if (print_results_) {
    std::lock_guard<std::mutex> lock(*output_file_mtx_);
    std::cout << "-----------------------------------------------------------"
              << std::endl;

    std::string filename = audio_data_->filename;
    std::cout << "File: " << filename << std::endl;
    if (result_->squad_question.size() == 0) {
      *output_file_ << "{\"audio_filepath\": \"" << filename << "\",";
      *output_file_ << "\"question\": \"\"}" << std::endl;
    } else {
      *output_file_ << "{\"audio_filepath\": \"" << filename << "\",";
      *output_file_ << "\"question\": \"" << result_->squad_answer << "\"}"
                    << std::endl;
      *output_file_ << "\"answer\": \"" << result_->squad_answer << "\"}"
                    << std::endl;

      std::cout << "SQUAD question: " << result_->squad_question << std::endl;
      std::cout << "SQUAD answer: " << result_->squad_answer << std::endl;
    }
  }
}

}  // namespace speech_squad
