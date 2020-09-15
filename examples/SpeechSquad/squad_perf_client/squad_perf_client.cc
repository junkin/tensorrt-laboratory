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

#include <alsa/asoundlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <strings.h>

#include "speech_squad_client.h"
#include "status.h"

using grpc::Status;
using grpc::StatusCode;

DEFINE_string(
    squad_questions_json, "questions.json",
    "Json file with location of audio files for each Squad question");
DEFINE_string(
    squad_dataset_json, "dev-v2.0.json", "Json file with Squad dataset");
DEFINE_string(
    speech_squad_uri, "localhost:50051", "URI to access speech-squad-server");
DEFINE_int32(num_iterations, 1, "Number of times to loop over audio files");
DEFINE_int32(
    offset_duration, 100,
    "The minimum time offset in microseconds between the launch of successive "
    "sequences");
DEFINE_int32(
    num_parallel_requests, 1, "Number of parallel requests to keep in flight");
DEFINE_int32(chunk_duration_ms, 100, "Chunk duration in milliseconds");
DEFINE_int32(
    executor_count, 1, "The number of threads to perform streaming I/O");
DEFINE_bool(print_results, true, "Print final results");
DEFINE_string(
    output_filename, "final_results.json",
    "Filename of .json file containing final results");

int
main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  std::stringstream str_usage;
  str_usage << "Usage: speech_squad_streaming_client " << std::endl;
  str_usage << "           --squad_questions_json=<question_json> "
            << std::endl;
  str_usage << "           --squad_dataset_json=<location_of_squad_json> "
            << std::endl;
  str_usage << "           --speech_squad_uri=<server_name:port> " << std::endl;
  str_usage << "           --chunk_duration_ms=<integer> " << std::endl;
  str_usage << "           --executor_count=<integer> " << std::endl;
  str_usage << "           --num_iterations=<integer> " << std::endl;
  str_usage << "           --offset_duration=<integer> " << std::endl;
  str_usage << "           --num_parallel_requests=<integer> " << std::endl;
  str_usage << "           --print_results=<true|false> " << std::endl;
  str_usage << "           --output_filename=<string>" << std::endl;
  ::google::SetUsageMessage(str_usage.str());
  ::google::SetVersionString("0.0.1");

  if (argc < 2) {
    std::cout << ::google::ProgramUsage();
    return 1;
  }

  ::google::ParseCommandLineFlags(&argc, &argv, true);

  if (argc > 1) {
    std::cout << ::google::ProgramUsage();
    return 1;
  }

  auto grpc_channel = grpc::CreateChannel(
      FLAGS_speech_squad_uri, grpc::InsecureChannelCredentials());

  std::chrono::system_clock::time_point deadline =
      std::chrono::system_clock::now() + std::chrono::milliseconds(10000);

  if (!WaitUntilReady(grpc_channel, deadline, FLAGS_speech_squad_uri)) {
    return 1;
  }

  auto squad_eval_dataset = std::make_shared<speech_squad::SquadEvalDataset>();
  speech_squad::Status status =
      squad_eval_dataset->LoadFromJson(FLAGS_squad_dataset_json);

  if (!status.IsOk()) {
    std::cerr << status.AsString() << std::endl;
    return 1;
  }

  speech_squad::SpeechSquadClient speech_squad_client(
      grpc_channel, FLAGS_num_parallel_requests, FLAGS_num_iterations, "en-US",
      FLAGS_print_results, FLAGS_chunk_duration_ms, FLAGS_executor_count,
      FLAGS_output_filename, squad_eval_dataset, FLAGS_squad_questions_json,
      FLAGS_num_iterations, FLAGS_offset_duration);

  return speech_squad_client.Run();
}
