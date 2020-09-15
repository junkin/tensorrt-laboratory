
#include "speech_squad_client.h"

namespace speech_squad {

SpeechSquadClient::SpeechSquadClient(
    std::shared_ptr<grpc::Channel> channel, int32_t num_parallel_requests,
    const size_t num_iterations, const std::string& language_code,
    bool print_transcripts, int32_t chunk_duration_ms, const int executor_count,
    const std::string& output_filename,
    std::shared_ptr<speech_squad::SquadEvalDataset> squad_eval_dataset,
    std::string squad_questions_json, int32_t num_iteration,
    uint64_t offset_duration)
    : speechsquad_stub_(SpeechSquadService::NewStub(channel)),
      num_parallel_requests_(num_parallel_requests),
      print_results_(print_transcripts), chunk_duration_ms_(chunk_duration_ms),
      squad_eval_dataset_(squad_eval_dataset),
      squad_questions_json_(squad_questions_json),
      num_iterations_(num_iterations), language_code_(language_code),
      offset_duration_(offset_duration)
{
  infer_prepare_fn_ = [this](
      ::grpc::ClientContext * context, ::grpc::CompletionQueue * cq) -> auto
  {
    return std::move(
        speechsquad_stub_->PrepareAsyncSpeechSquadInfer(context, cq));
  };

  executor_ = std::make_shared<nvrpc::client::Executor>(executor_count);
  output_file_ = std::make_shared<std::ofstream>();
  output_file_mtx_ = std::make_shared<std::mutex>();

  if (print_results_) {
    output_file_->open(output_filename);
  }
}

SpeechSquadClient::~SpeechSquadClient()
{
  if (print_results_) {
    output_file_->close();
  }
}

int
SpeechSquadClient::Run()
{
  sending_complete_ = false;

  std::vector<std::shared_ptr<AudioData>> all_wav;
  LoadAudioData(all_wav, squad_questions_json_);

  if (all_wav.size() == 0) {
    std::cout << "Exiting.." << std::endl;
    return 1;
  }

  uint32_t all_wav_max = all_wav.size() * num_iterations_;
  response_latencies_.clear();
  response_latencies_.reserve(all_wav_max);

  std::vector<std::unique_ptr<AudioTask>> curr_tasks, next_tasks;
  curr_tasks.reserve(num_parallel_requests_);
  next_tasks.reserve(num_parallel_requests_);

  SyncQueue<std::unique_ptr<AudioTask>> awaited_tasks;
  // Starts a reaper thread. It will sequentially visit all the
  // tasks.
  reaper_thread_ =
      std::thread([this, &awaited_tasks]() { ReaperFunction(awaited_tasks); });

  std::vector<std::shared_ptr<AudioData>> all_wav_repeated;
  all_wav_repeated.reserve(all_wav_max);
  for (uint32_t file_id = 0; file_id < all_wav.size(); file_id++) {
    for (int iter = 0; iter < num_iterations_; iter++) {
      all_wav_repeated.push_back(all_wav[file_id]);
    }
  }

  uint32_t all_wav_i = 0;
  auto start_time = std::chrono::high_resolution_clock::now();
  auto next_time = std::chrono::high_resolution_clock::now();
  while (true) {
    int offset_index = 0;
    auto now = std::chrono::high_resolution_clock::now();
    while (curr_tasks.size() < num_parallel_requests_ &&
           all_wav_i < all_wav_max) {
      auto scheduled_time =
          now + std::chrono::microseconds((offset_index++) * offset_duration_);
      std::unique_ptr<AudioTask> ptr(new AudioTask(
          all_wav_repeated[all_wav_i], all_wav_i, infer_prepare_fn_,
          language_code_, chunk_duration_ms_, print_results_,
          squad_eval_dataset_, output_file_, output_file_mtx_, executor_,
          scheduled_time));
      curr_tasks.emplace_back(std::move(ptr));
      ++all_wav_i;
    }

    // If still empty, done
    if (curr_tasks.empty()) {
      break;
    }

    bool sent_request = false;
    for (size_t itask = 0; itask < curr_tasks.size(); ++itask) {
      AudioTask& task = *(curr_tasks[itask]);

      auto now = std::chrono::high_resolution_clock::now();
      if (now < task.NextTimePoint()) {
        if ((itask == 0) || (next_time < task.NextTimePoint())) {
          next_time = task.NextTimePoint();
        }
        continue;
      }
      sent_request = true;
      auto status = task.Step();
      if (!status.IsOk()) {
        WaitForReaper();
        std::cerr << "Failed to generate specified load. Error details: "
                  << status.AsString() << std::endl;
        return -1;
      }

      if (task.GetState() == AudioTask::SENDING_COMPLETE) {
        next_tasks.push_back(std::move(curr_tasks[itask]));
      } else {
        awaited_tasks.Put(std::move(curr_tasks[itask]));
      }
    }

    // If none of the tasks were ready then sleep till the next activity
    double wait_for =
        std::chrono::duration<double, std::milli>(now - next_time).count();
    usleep(wait_for * 1.e3);

    curr_tasks.swap(next_tasks);
    next_tasks.clear();
  }

  WaitForReaper();
  if (!reaper_status_.IsOk()) {
    std::cerr << "Error encountered while retrieving results. Error details: "
              << reaper_status_.AsString() << std::endl;
    return -1;
  }

  auto current_time = std::chrono::high_resolution_clock::now();
  {
    PrintLatencies();
    std::cout << std::flush;
    double diff_time =
        std::chrono::duration<double, std::milli>(current_time - start_time)
            .count();

    std::cout << "Run time: " << diff_time / 1000. << " sec." << std::endl;
    std::cout << "Total audio processed: " << TotalAudioProcessed() << " sec."
              << std::endl;
    std::cout << "Throughput: " << TotalAudioProcessed() * 1000. / diff_time
              << " RTFX" << std::endl;
  }

  return 0;
}

void
SpeechSquadClient::WaitForReaper()
{
  sending_complete_ = true;
  if (reaper_thread_.joinable()) {
    reaper_thread_.join();
  }
}

void
SpeechSquadClient::ReaperFunction(
    SyncQueue<std::unique_ptr<AudioTask>>& awaited_tasks)
{
  while (!sending_complete_) {
    while (!awaited_tasks.Empty()) {
      auto awaited_task = std::move(awaited_tasks.Get());
      auto new_status = awaited_task->WaitForCompletion();
      if (reaper_status_.IsOk() && (!new_status.IsOk())) {
        reaper_status_ = new_status;
      }
      total_audio_processed_ += awaited_task->AudioProcessed();
      auto this_result = awaited_task->GetResult();
      response_latencies_.push_back(this_result->response_latency);
    }
    usleep(100);
  }
}

void
SpeechSquadClient::PrintLatencies()
{
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::vector<double> latencies;
  for (auto& latency : response_latencies_) {
    latencies.insert(
        latencies.end(), response_latencies_.begin(),
        response_latencies_.end());
  }

  if (latencies.size() > 0) {
    std::sort(latencies.begin(), latencies.end());
    double nresultsf = static_cast<double>(latencies.size());
    size_t per50i = static_cast<size_t>(std::floor(50. * nresultsf / 100.));
    size_t per90i = static_cast<size_t>(std::floor(90. * nresultsf / 100.));
    size_t per95i = static_cast<size_t>(std::floor(95. * nresultsf / 100.));
    size_t per99i = static_cast<size_t>(std::floor(99. * nresultsf / 100.));

    double median = latencies[per50i];
    double lat_90 = latencies[per90i];
    double lat_95 = latencies[per95i];
    double lat_99 = latencies[per99i];

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                 latencies.size();

    std::cout << std::setprecision(5);
    std::cout << "Response Latency (Last Request --- First Response)"
              << " (ms):\n";
    std::cout << "\t\tMedian\t\t90th\t\t95th\t\t99th\t\tAvg\n";
    std::cout << "\t\t" << median << "\t\t" << lat_90 << "\t\t" << lat_95
              << "\t\t" << lat_99 << "\t\t" << avg << std::endl;
  }
}

}  // namespace speech_squad
