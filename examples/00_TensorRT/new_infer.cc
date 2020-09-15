
/* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include <sys/stat.h>
#include <unistd.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <trtlab/core/fiber_group.h>
#include <trtlab/core/standard_threads.h>
#include <trtlab/core/userspace_threads.h>

#include <trtlab/cuda/common.h>
#include <trtlab/cuda/sync.h>
#include <trtlab/cuda/memory/cuda_allocators.h>

#include <trtlab/tensorrt/runtime.h>
#include <trtlab/tensorrt/model.h>
#include <trtlab/tensorrt/execution_context.h>
#include <trtlab/tensorrt/workspace.h>

#include <trtlab/core/pool.h>

#include <cuda_profiler_api.h>

#include "NvInfer.h"

using namespace trtlab;

static bool ValidateEngine(const char* flagname, const std::string& value)
{
    struct stat buffer;
    return (stat(value.c_str(), &buffer) == 0);
}

DEFINE_string(engine, "/path/to/tensorrt.engine", "TensorRT serialized engine");
DEFINE_validator(engine, &ValidateEngine);
DEFINE_int32(seconds, 10, "Approximate number of seconds for the timing loop");
DEFINE_int32(contexts, 1, "Number of Execution Contexts");
DEFINE_int32(buffers, 0, "Number of Buffers (default: 2x contexts)");
DEFINE_int32(threads, 2, "Number Response Sync Threads");
DEFINE_int32(replicas, 1, "Number of Replicas of the Model to load");
DEFINE_int32(batch_size, 0, "Overrides the max batch_size of the provided engine");

using thread_t = trtlab::userspace_threads;

//#define USE_NVTX

#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[]   = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff};
const int      num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                                                                                                              \
    {                                                                                                                                      \
        int color_id                      = cid;                                                                                           \
        color_id                          = color_id % num_colors;                                                                         \
        nvtxEventAttributes_t eventAttrib = {0};                                                                                           \
        eventAttrib.version               = NVTX_VERSION;                                                                                  \
        eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                                                 \
        eventAttrib.colorType             = NVTX_COLOR_ARGB;                                                                               \
        eventAttrib.color                 = colors[color_id];                                                                              \
        eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;                                                                       \
        eventAttrib.message.ascii         = name;                                                                                          \
        nvtxRangePushEx(&eventAttrib);                                                                                                     \
    }
#define POP_RANGE nvtxRangePop
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE()
#endif

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("infer.x");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    //FiberGroup<boost::fibers::algo::shared_work> fibers(FLAGS_threads);
    auto runtime    = std::make_shared<TensorRT::StandardRuntime>();
    int  batch_size = -1;

    auto workspaces = UniquePool<TensorRT::BenchmarkWorkspace, thread_t>::Create();

    for (int i = 0; i < FLAGS_contexts; i++)
    {
        auto model = runtime->deserialize_engine(FLAGS_engine);
        LOG(INFO) << "binding details" << std::endl << model->bindings_info();
        workspaces->emplace_push(model);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto end   = start + std::chrono::seconds(10);

    std::vector<typename thread_t::future<std::size_t>> futures;

    for (int i = 0; i < workspaces->size(); i++)
    {
        futures.push_back(thread_t::async([ workspaces, i, end ]() -> auto {
            auto                            workspace = workspaces->pop_unique();
            std::size_t                     batches   = 0;
            std::vector<memory::descriptor> inputs;

            while (std::chrono::high_resolution_clock::now() < end)
            {
                PUSH_RANGE("graph workspace", i);
                workspace->async_h2d();
                workspace->enqueue();
                workspace->async_d2h();
                cuda_sync<thread_t>::stream_sync(workspace->stream());
                batches++;
                POP_RANGE();
            }
            return batches;
        }));
    }

    std::size_t batches = 0;
    for (auto& f : futures)
    {
        batches += f.get();
    }

    end = std::chrono::high_resolution_clock::now();

    auto workspace  = workspaces->pop_unique();
    auto inferences = batches * workspace->batch_size();
    auto seconds    = std::chrono::duration<double>(end - start).count();

    LOG(INFO) << workspace->batch_size();
    LOG(INFO) << "time:       :" << seconds;
    LOG(INFO) << "batch_size  :" << workspace->batch_size();
    LOG(INFO) << "batches     :" << batches;
    LOG(INFO) << "inf/sec     :" << inferences / seconds;
    LOG(INFO) << "avg latency :" << 1 / (inferences / seconds);

    LOG(INFO) << "finished";

    //LOG(INFO) << "performing a single inference with profile enabled";
    //cudaProfilerStart();
    //workspace->enqueue();
    //CHECK_CUDA(cudaStreamSynchronize(workspace->stream()));
    //cudaProfilerStop();
    //LOG(INFO) << "profiling complete";

    return 0;
}
