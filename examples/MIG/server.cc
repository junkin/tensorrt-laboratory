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
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <thread>
#include <sys/stat.h>

#include <boost/fiber/all.hpp>

#include <nvrpc/server.h>
#include <nvrpc/service.h>
#include <nvrpc/executor.h>
#include <nvrpc/fiber/executor.h>

#include <trtlab/core/pool.h>
#include <trtlab/core/resources.h>
#include <trtlab/core/thread_pool.h>
#include <trtlab/core/fiber_group.h>

#include <trtlab/cuda/common.h>
#include <trtlab/cuda/sync.h>

#include <trtlab/tensorrt/runtime.h>
#include <trtlab/tensorrt/workspace.h>

#include "birds.pb.h"
#include "birds.grpc.pb.h"

using nvrpc::AsyncRPC;
using nvrpc::AsyncService;
using nvrpc::Context;
using nvrpc::Executor;
using nvrpc::FiberExecutor;
using nvrpc::Server;

using trtlab::FiberGroup;
using trtlab::Resources;
using trtlab::ThreadPool;

static bool ValidateEngine(const char* flagname, const std::string& value)
{
    struct stat buffer;
    return (stat(value.c_str(), &buffer) == 0);
}

// CLI Options
DEFINE_int32(thread_count, 2, "Size of thread pool");
DEFINE_int32(queue_depth, 100, "Number of active RPC in-flight");
DEFINE_string(audio_model, "", "audio model");
DEFINE_string(nlp_model, "", "nlp model");

// Validators
DEFINE_validator(audio_model, &ValidateEngine);
DEFINE_validator(nlp_model, &ValidateEngine);

using thread_t = trtlab::standard_threads;

using model_t     = std::shared_ptr<trtlab::TensorRT::Model>;
using workspace_t = trtlab::TensorRT::TimedBenchmarkWorkspace;

struct DemoResources : public Resources
{
    DemoResources(model_t audio_model, model_t nlp_model) : m_AudioWorkspace(audio_model), m_NLPWorkspace(nlp_model) {}

    workspace_t& audio_workspace()
    {
        return m_AudioWorkspace;
    }
    workspace_t& nlp_workspace()
    {
        return m_NLPWorkspace;
    }

private:
    workspace_t m_AudioWorkspace;
    workspace_t m_NLPWorkspace;
};

// Contexts hold the state and provide the definition of the work to be performed by the RPC.
// Incoming Message = BirdsRPC::Input (RequestType)
// Outgoing Message = BirdsRPC::Output (ResponseType)
class ComputeBatchContext final : public Context<BirdsRPC::BatchBirdRequest, BirdsRPC::BatchBirdResponse, DemoResources>
{
    void ExecuteRPC(BirdsRPC::BatchBirdRequest& input, BirdsRPC::BatchBirdResponse& output) final override
    {
        using namespace std::chrono_literals;

        auto& audio_workspace = GetResources()->audio_workspace();
        auto& nlp_workspace   = GetResources()->nlp_workspace();

        VLOG(5) << "request started";

        // cudaEvent_t cross_workspace_sync;
        // CHECK_CUDA(cudaEventCreateWithFlags(&cross_workspace_sync, cudaEventDisableTiming));

        // audio classification
        // todo: populate bird audio data
        audio_workspace.enqueue_pipeline();

        // mark audio workspace complete
        // CHECK_CUDA(cudaEventRecord(cross_workspace_sync, audio_workspace->stream()));

        // nlp workspace needs to wait on audio workspace
        // CHECK_CUDA(cudaStreamWaitEvent(nlp_workspace->stream(), cross_workspace_sync, 0));

        // todo: populate bert squad question and context
        // nlp_workspace->enqueue_pipeline();

        trtlab::cuda_sync<thread_t>::stream_sync(audio_workspace.stream());

        nlp_workspace.enqueue_pipeline();

        trtlab::cuda_sync<thread_t>::stream_sync(nlp_workspace.stream());

        std::size_t asr_compute_ns = (audio_workspace.get_compute_time_ms() * 1000 * 1000);
        std::size_t nlp_compute_ns = (nlp_workspace.get_compute_time_ms() * 1000 * 1000);

        // a100 mig x 1slice - int8 b1 - 211 inf/sec - 4.739ms (4740us)
        // auto start = std::chrono::high_resolution_clock::now();
        // thread_t::sleep_for(4740us);
        // auto end            = std::chrono::high_resolution_clock::now();
        // auto nlp_compute_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        output.set_batch_id(input.batch_id());
        output.set_batch_size(input.batch_size());

        // write response for each batch item
        for (int i = 0; i < input.batch_size(); i++)
        {
            auto item = output.add_responses();

            item->set_request_id(input.requests(i).request_id());
            item->set_start_timestamp(0);

            item->set_has_asr(true);
            item->set_asr_timestamp(asr_compute_ns);
            // todo: item->set_bird_class_id( );

            item->set_has_nlp(true);
            item->set_nlp_timestamp(asr_compute_ns + nlp_compute_ns);
            item->set_nlp_answer("north america");
        }

        FinishResponse();
        VLOG(5) << "batch request complete";
    }
};

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console

    ::google::InitGoogleLogging("MIG Demo");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    FiberGroup<> fibers(FLAGS_thread_count);

    // initialize trt
    auto runtime = std::make_shared<trtlab::TensorRT::StandardRuntime>();

    // load models
    auto audio_model = runtime->deserialize_engine(FLAGS_audio_model);
    auto nlp_model   = runtime->deserialize_engine(FLAGS_nlp_model);

    LOG(INFO) << "audio model bindings: " << std::endl << audio_model->bindings_info();
    LOG(INFO) << "nlp model bindings: " << std::endl << nlp_model->bindings_info();

    // A server will bind an IP:PORT to listen on
    Server server("0.0.0.0:50051");

    // A server can host multiple services
    LOG(INFO) << "Register Service (BirdsRPC::Inference) with Server";
    auto service = server.RegisterAsyncService<BirdsRPC::Inference>();

    // An RPC has two components that need to be specified when registering with the service:
    //  1) Type of Execution Context (SimpleContext).  The execution context defines the behavor
    //     of the RPC, i.e. it contains the control logic for the execution of the RPC.
    //  2) The Request function (RequestCompute) which was generated by gRPC when compiling the
    //     protobuf which defined the service.  This function is responsible for queuing the
    //     RPC's execution context to the
    LOG(INFO) << "Register RPC (BirdsRPC::Inference::Compute) with Service (BirdsRPC::Inference)";
    auto compute_batch = service->RegisterRPC<ComputeBatchContext>(&BirdsRPC::Inference::AsyncService::RequestComputeBatch);

    LOG(INFO) << "Initializing Resources for RPC (BirdsRPC::Inference::ComputeBatch)";
    auto rpcResources = std::make_shared<DemoResources>(audio_model, nlp_model);

    // Create Executors - Executors provide the messaging processing resources for the RPCs
    // Multiple Executors can be registered with a Server.  The executor is responsible
    // for pulling incoming message off the receive queue and executing the associated
    // context.  By default, an executor only uses a single thread.  A typical usecase is
    // an Executor executes a context, which immediate pushes the work to a thread pool.
    // However, for very low-latency messaging, you might want to use a multi-threaded
    // Executor and a Blocking Context - meaning the Context performs the entire RPC function
    // on the Executor's thread.
    LOG(INFO) << "Creating Fiber Executor";
    auto executor = server.RegisterExecutor(new Executor(1));

    // You can register RPC execution contexts from any registered RPC on any executor.
    // The power of that will become clear in later examples. For now, we will register
    // 10 instances of the BirdsRPC::Inference::Compute RPC's SimpleContext execution context
    // with the Executor.
    LOG(INFO) << "Creating Execution Contexts for RPC (BirdsRPC::Inference::ComputeBatch) with Executor";
    executor->RegisterContexts(compute_batch, rpcResources, FLAGS_queue_depth);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(2000), [] {
        // This is a timeout loop executed every 2seconds
        // Run() with no arguments will run an empty timeout loop every 5 seconds.
        // RunAsync() will return immediately, its your responsibility to ensure the
        // server doesn't go out of scope or a Shutdown will be triggered on your services.
    });
}
