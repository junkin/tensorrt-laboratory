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

#include <alsa/asoundlib.h>
#include <dirent.h>
#include <grpcpp/grpcpp.h>
#include <sys/stat.h>

#include <iostream>
#include <sstream>

#include "rapidjson/document.h"
#include "speech_squad.grpc.pb.h"
#include "squad_eval_dataset.h"

#define WAVE_FORMAT_PCM 0x0001
#define WAVE_FORMAT_ALAW 0x0006
#define WAVE_FORMAT_MULAW 0x0007

typedef struct __attribute__((packed)) {
  char chunk_id[4];  // should be "RIFF" in ASCII form
  int32_t chunk_size;
  char format[4];          // should be "WAVE"
  char subchunk1_id[4];    // should be "fmt "
  int32_t subchunk1_size;  // should be 16 for PCM format
  int16_t audioformat;     // should be 1 for PCM
  int16_t numchannels;
  int32_t samplerate;
  int32_t byterate;       // == samplerate * numchannels * bitspersample/8
  int16_t blockalign;     // == numchannels * bitspersample/8
  int16_t bitspersample;  //    8 bits = 8, 16 bits = 16, etc.
  int32_t subchunk2ID;
  int32_t subchunk2size;
} FixedWAVHeader;

struct AudioData {
  std::vector<char> data;
  std::string filename;
  int sample_rate;
  int channels;
  AudioEncoding encoding;
  std::string question_id;
};

bool
WaitUntilReady(
    std::shared_ptr<grpc::Channel> channel,
    std::chrono::system_clock::time_point& deadline,
    std::string speech_squad_uri);

bool
ParseAudioFileHeader(
    std::string file, AudioEncoding& encoding, int& samplerate, int& channels);

bool
ParseQuestionsJson(
    const char* path,
    std::vector<std::pair<std::string, std::string>>& questions);

void
LoadAudioData(
    std::vector<std::shared_ptr<AudioData>>& all_audio_data, std::string& path);
