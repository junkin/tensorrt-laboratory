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

#include "status.h"

namespace speech_squad {
const Status Status::Success(Status::Code::SUCCESS);

std::string
Status::AsString() const
{
  std::string str(CodeString(code_));
  str += ": " + msg_;
  return str;
}

const char*
Status::CodeString(const Code code)
{
  switch (code) {
    case Status::Code::SUCCESS:
      return "OK";
    case Status::Code::UNKNOWN:
      return "Unknown";
    case Status::Code::INTERNAL:
      return "Internal";
    case Status::Code::NOT_FOUND:
      return "Not found";
    case Status::Code::INVALID_ARG:
      return "Invalid argument";
    case Status::Code::UNAVAILABLE:
      return "Unavailable";
    case Status::Code::UNSUPPORTED:
      return "Unsupported";
    case Status::Code::ALREADY_EXISTS:
      return "Already exists";
    default:
      break;
  }

  return "<invalid code>";
}

}  // namespace speech_squad
