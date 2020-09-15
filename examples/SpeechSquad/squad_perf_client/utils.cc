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

#include "utils.h"

bool
WaitUntilReady(
    std::shared_ptr<grpc::Channel> channel,
    std::chrono::system_clock::time_point& deadline,
    std::string speech_squad_uri)
{
  auto state = channel->GetState(true);
  while (state != GRPC_CHANNEL_READY) {
    if (!channel->WaitForStateChange(state, deadline)) {
      std::cout << "Cannot create GRPC channel at uri " << speech_squad_uri
                << std::endl;
      return false;
    }
    state = channel->GetState(true);
  }
  return true;
}

bool
ParseAudioFileHeader(
    std::string file, AudioEncoding& encoding, int& samplerate, int& channels)
{
  std::ifstream file_stream(file);
  FixedWAVHeader header;
  std::streamsize bytes_read = file_stream.rdbuf()->sgetn(
      reinterpret_cast<char*>(&header), sizeof(header));

  if (bytes_read != sizeof(header)) {
    std::cerr << "Error reading file " << file << std::flush << std::endl;
    return false;
  }

  std::string tag(header.chunk_id, 4);
  if (tag == "RIFF") {
    if (header.audioformat == WAVE_FORMAT_PCM)
      // Only supports LINEAR_PCM
      encoding = LINEAR_PCM;
    else
      return false;
    samplerate = header.samplerate;
    channels = header.numchannels;
    return true;
  } else if (tag == "fLaC") {
    return false;
  }
  return false;
}

bool
ParseQuestionsJson(
    const char* path,
    std::vector<std::pair<std::string, std::string>>& questions)
{
  questions.clear();
  std::ifstream manifest_file;
  manifest_file.open(path, std::ifstream::in);

  std::string line;
  std::string audio_filepath_key("audio_filepath");
  std::string question_id_key("id");

  while (std::getline(manifest_file, line, '\n')) {
    if (line == "") {
      continue;
    }

    rapidjson::Document doc;
    // Parse line
    doc.Parse(line.c_str());

    if (!doc.IsObject()) {
      std::cout << "Problem parsing line: " << line << std::endl;
      return false;
    }

    // Get Question ID
    if (!doc.HasMember(question_id_key.c_str())) {
      std::cout << "Line: " << line << " does not contain " << question_id_key
                << " key" << std::endl;
      return false;
    }
    std::string question_id = doc[question_id_key.c_str()].GetString();

    // Get filepath
    if (!doc.HasMember(audio_filepath_key.c_str())) {
      std::cout << "Line: " << line << " does not contain "
                << audio_filepath_key << " key" << std::endl;
      return false;
    }
    std::string filepath = doc[audio_filepath_key.c_str()].GetString();

    questions.emplace_back(std::make_pair(question_id, filepath));
  }

  manifest_file.close();
  return true;
}

void
LoadAudioData(
    std::vector<std::shared_ptr<AudioData>>& all_audio_data, std::string& path)
{
  std::cout << "Loading eval dataset..." << std::flush << std::endl;

  std::vector<std::pair<std::string, std::string>> questions;

  ParseQuestionsJson(path.c_str(), questions);

  for (uint32_t i = 0; i < questions.size(); ++i) {
    std::string question_id = questions[i].first;
    std::string filename = questions[i].second;

    AudioEncoding encoding;
    int samplerate;
    int channels;
    if (!ParseAudioFileHeader(filename, encoding, samplerate, channels)) {
      std::cerr << "Cannot parse audio file header for file " << filename
                << std::endl;
      return;
    }
    std::shared_ptr<AudioData> audio_data = std::make_shared<AudioData>();

    audio_data->sample_rate = samplerate;
    audio_data->filename = filename;
    audio_data->question_id = question_id;
    audio_data->encoding = encoding;
    audio_data->channels = channels;
    audio_data->data.assign(
        std::istreambuf_iterator<char>(std::ifstream(filename).rdbuf()),
        std::istreambuf_iterator<char>());
    all_audio_data.push_back(std::move(audio_data));
  }

  std::cout << "Done loading " << questions.size() << " files" << std::endl;
}
