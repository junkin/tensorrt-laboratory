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

#include "squad_eval_dataset.h"

#include <fstream>
#include <iostream>

#include "status.h"

namespace speech_squad {
Status
SquadEvalDataset::LoadFromJson(const std::string& json_filepath)
{
  std::ifstream json_stream;
  json_stream.open(json_filepath, std::ifstream::in);

  if (!json_stream.is_open()) {
    return Status(
        Status::Code::NOT_FOUND, "Could not open file " + json_filepath);
  }

  std::string line;
  std::getline(json_stream, line, '\n');
  rapidjson::Document doc;
  doc.Parse(line.c_str());

  contexts_.reserve(15000);

  // int question_count = 0;
  // std::ofstream myfile;
  // myfile.open("squad_questions.json");

  const auto& data_array = doc["data"];
  assert(data_array.IsArray());  // attributes is an array
  // Loop over the themes
  for (auto itr = data_array.Begin(); itr != data_array.End(); ++itr) {
    auto& data = *itr;
    assert(data.IsObject());  // each attribute is an object

    const auto& paragraph_array = data["paragraphs"];
    assert(paragraph_array.IsArray());  // attributes is an array

    for (auto itr2 = paragraph_array.Begin(); itr2 != paragraph_array.End();
         ++itr2) {
      auto& paragraph = *itr2;
      assert(paragraph.IsObject());  // each attribute is an object

      std::string context = paragraph["context"].GetString();
      const auto& qas_array = paragraph["qas"];
      // std::cout << "context=" << context << std::endl;

      contexts_.push_back(std::make_shared<std::string>(context));

      for (auto itr3 = qas_array.Begin(); itr3 != qas_array.End(); ++itr3) {
        auto& qa = *itr3;
        assert(qa.IsObject());  // each attribute is an object

        std::string question = qa["question"].GetString();
        std::string question_id = qa["id"].GetString();
        questions_[question_id] = question;
        question_contexts_[question_id] = contexts_.back();

        const auto& answers_array = qa["answers"];
        assert(answers_array.IsArray());  // attributes is an array
        for (auto itr4 = answers_array.Begin(); itr4 != answers_array.End();
             ++itr4) {
          auto& answer = *itr4;
          assert(answer.IsObject());  // each attribute is an object
        }

        // myfile << "{\"audio_filepath\":
        // \"/work/test_files/speech_squad/speech_squad" <<
        // std::to_string(question_count) << ".wav\",\"id\":\"" << question_id
        // << "\" }" << std::endl;

        // question_count++;
        // std::cout << "question=" << question << std::endl;
        // std::cout << "id=" << question_id << std::endl;a
      }
    }
  }
  // myfile.close();

  if (!doc.IsObject()) {
    std::cout << "Problem parsing line: " << line << std::endl;
    return Status(Status::Code::INTERNAL, "Cannot parse Squad json file");
  }

  json_stream.close();
  return Status::Success;
}

Status
SquadEvalDataset::GetQuestion(const std::string& id, std::string* question)
{
  if (questions_.find(id) == questions_.end()) {
    return Status(Status::Code::UNKNOWN, "Question id " + id + " not found");
  } else {
    *question = questions_[id];
    return Status::Success;
  }
}

Status
SquadEvalDataset::GetQuestionContext(
    const std::string& id, std::string* context)
{
  if (question_contexts_.find(id) == question_contexts_.end()) {
    return Status(Status::Code::UNKNOWN, "Question id " + id + " not found");
  } else {
    *context = *(question_contexts_[id]);
    return Status::Success;
  }
}
}  // namespace speech_squad
