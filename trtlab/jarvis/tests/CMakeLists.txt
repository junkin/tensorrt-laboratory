# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

find_package(GTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

enable_language(CUDA)

message("Source dir of i cub = ${CUB_INCLUDE_DIR}")

add_executable(test_jarvis
  test_main.cc
  test_jarvis.cc
  test_config.cc
  test_mel_banks.cc
  test_gather_audio_batch.cc
  test_scatter_features.cc
  test_gather_normalized_features.cc
  test_tensorrt.cc
  
  builder.cc
  context.cc
  context_pointers.cc

  pipeline.cc
  pipeline/gather_audio_batch.cu
  pipeline/process_audio.cu
  pipeline/extract_fft_windows.cu
  pipeline/power_spectrum.cu
  pipeline/mel_banks_compute.cu
  pipeline/scatter_features.cu
  pipeline/gather_normalized_features.cu
  pipeline/scatter_normalized_features.cu

  mel_banks.cc
  cuda_mel_banks.cu
  cuda_vector.cu
  cuda_matrix.cu
  jarvis_asr.cc
)

set_property(TARGET test_jarvis PROPERTY CUDA_STANDARD 14)

target_link_libraries(test_jarvis
  PRIVATE 
    ${PROJECT_NAME}::core
    ${PROJECT_NAME}::cuda
    ${PROJECT_NAME}::tensorrt
    trtlab_memory
    GTest::gtest
    cublas
    cufft
    curand
    cnpy
)

#target_include_directories(test_jarvis PUBLIC ${CUB_INCLUDE_DIR})

target_link_directories(test_jarvis PUBLIC ${INSTALL_ROOT}/lib)

add_test(
  NAME jarvis
  COMMAND $<TARGET_FILE:test_jarvis>
)
