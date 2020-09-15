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

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

#include "audio_task.h"
#include "status.h"
#include "sync_queue.h"
#include "thread-pool.h"

#include "nvrpc/client/executor.h"

namespace speech_squad {

class SpeechSquadClient {
 public:
  SpeechSquadClient(
      std::shared_ptr<grpc::Channel> channel, int32_t num_parallel_requests,
      const size_t num_iterations, const std::string& language_code,
      bool print_transcripts, int32_t chunk_duration_ms,
      const int executor_count, const std::string& output_filename,
      std::shared_ptr<speech_squad::SquadEvalDataset> squad_eval_dataset,
      std::string squad_questions_json, int32_t num_iteration,
      uint64_t offset_duration);

  ~SpeechSquadClient();

  // This function is not thread-safe
  int Run();

 private:
  double TotalAudioProcessed() { return total_audio_processed_; }
  void WaitForReaper();
  void ReaperFunction(SyncQueue<std::unique_ptr<AudioTask>>& awaited_tasks);
  void PrintLatencies();

  std::unique_ptr<SpeechSquadService::Stub> speechsquad_stub_;
  int num_parallel_requests_;
  bool print_results_;
  double chunk_duration_ms_;

  std::shared_ptr<speech_squad::SquadEvalDataset> squad_eval_dataset_;
  std::string squad_questions_json_;
  int32_t num_iterations_;
  std::string language_code_;
  uint64_t offset_duration_;

  Stream::PrepareFn infer_prepare_fn_;
  std::shared_ptr<nvrpc::client::Executor> executor_;
  std::shared_ptr<std::ofstream> output_file_;
  std::shared_ptr<std::mutex> output_file_mtx_;

  std::vector<double> response_latencies_;
  double total_audio_processed_;
  bool sending_complete_;

  std::thread reaper_thread_;
  Status reaper_status_;
};

}  // namespace speech_squad