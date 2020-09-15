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
#include "nvml.h"
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/literals.h>

#include <trtlab/core/pool.h>
#include <trtlab/core/thread_pool.h>
#include <trtlab/core/fiber_group.h>
#include <trtlab/core/standard_threads.h>
#include <trtlab/core/userspace_threads.h>

#include <trtlab/cuda/memory/cuda_allocators.h>
#include <trtlab/cuda/common.h>
#include <trtlab/cuda/sync.h>

#include "trtlab/cuda/device_info.h"
#include "trtlab/tensorrt/runtime.h"

#include <trtlab/tensorrt/workspace.h>

#include "nvrpc/context.h"
#include "nvrpc/fiber/executor.h"
#include "nvrpc/executor.h"
#include "nvrpc/server.h"
#include "nvrpc/service.h"

using nvrpc::AsyncRPC;
using nvrpc::AsyncService;
using nvrpc::Context;
using nvrpc::Executor;
using nvrpc::FiberExecutor;
using nvrpc::Server;

using namespace trtlab;
using namespace trtlab::memory::literals;

// Flowers Protos
#include "inference.grpc.pb.h"
#include "inference.pb.h"

using ssd::BatchInput;
using ssd::BatchPredictions;
using ssd::Inference;

using thread_t = trtlab::userspace_threads;

/*
 * External Data Source
 *
 * Attaches to a System V shared memory segment owned by an external resources.
 * Example: the results of an image decode service could use this mechanism to transfer
 *          large tensors to an inference service by simply passing an offset.
 */
float* GetSharedMemory(const std::string& address);

/*
 * Resources - TensorRT InferenceManager + ThreadPools + External Datasource
 */
class FlowersResources : public Resources
{
public:
    using workspace_t = TensorRT::BenchmarkWorkspace;
    using pool_t      = UniquePool<workspace_t, thread_t>;

    FlowersResources(std::shared_ptr<pool_t> workspaces, float* sysv_data, int nthreads)
    : m_Workspaces(workspaces), m_SharedMemory(sysv_data), m_ThreadPool(std::make_unique<ThreadPool>(nthreads))
    {
    }

    float* get_sysv_offset(size_t offset_in_bytes)
    {
        return &m_SharedMemory[offset_in_bytes / sizeof(float)];
    }

    pool_t& workspaces()
    {
        return *m_Workspaces;
    }

private:
    float*                      m_SharedMemory;
    std::shared_ptr<pool_t>     m_Workspaces;
    std::unique_ptr<ThreadPool> m_ThreadPool;
};

/*
 * nvRPC Context - Defines the logic of the RPC.
 */
class FlowersContext final : public Context<BatchInput, BatchPredictions, FlowersResources>
{
    void ExecuteRPC(RequestType& input, ResponseType& output) final override
    {
        //LOG(INFO) << "execute rpc";
        auto flower_data = GetResources()->get_sysv_offset(input.sysv_offset());
        auto workspace   = GetResources()->workspaces().pop_unique();
        auto start = std::chrono::high_resolution_clock::now();
        workspace->async_h2d();
        workspace->enqueue();
        workspace->async_d2h();
        cuda_sync<thread_t>::stream_sync(workspace->stream());
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float>(end - start).count();
        WriteBatchPredictions(input, output, nullptr);
        output.set_compute_time(ms);
        output.set_total_time(ms);
        FinishResponse();
    }

    void WriteBatchPredictions(RequestType& input, ResponseType& output, float* scores)
    {
        int N = input.batch_size();

        // auto nClasses = GetResources()->GetModel("flowers")->GetBinding(1).elementsPerBatchItem;
        // size_t cntr = 0;

        for (int p = 0; p < N; p++)
        {
            auto element = output.add_elements();
            /* Customize the post-processing of the output tensor *\
            float max_val = -1.0;
            int max_idx = -1;
            for (int i = 0; i < nClasses; i++)
            {
                if (max_val < scores[cntr])
                {
                    max_val = scores[cntr];
                    max_idx = i;
                }
                cntr++;
            }
            auto top1 = element->add_predictions();
            top1->set_class_id(max_idx);
            top1->set_score(max_val);
            \* Customize the post-processing of the output tensor */
        }
        output.set_batch_id(input.batch_id());
    }
};

static bool ValidateEngine(const char* flagname, const std::string& value)
{
    struct stat buffer;
    return (stat(value.c_str(), &buffer) == 0);
}

static bool ValidateBytes(const char* flagname, const std::string& value)
{
    trtlab::StringToBytes(value);
    return true;
}

DEFINE_string(engine, "/path/to/tensorrt.engine", "TensorRT serialized engine");
DEFINE_validator(engine, &ValidateEngine);
DEFINE_string(dataset, "127.0.0.1:4444", "GRPC Dataset/SharedMemory Service Address");
DEFINE_int32(contexts, 1, "Number of Execution Contexts");
DEFINE_int32(buffers, 0, "Number of Input/Output Buffers");
DEFINE_string(runtime, "default", "TensorRT Runtime");
DEFINE_int32(execution_threads, 1, "Number of RPC execution threads");
DEFINE_int32(preprocessing_threads, 0, "Number of preprocessing threads");
DEFINE_int32(kernel_launching_threads, 1, "Number of threads to launch CUDA kernels");
DEFINE_int32(postprocessing_threads, 2, "Number of postprocessing threads");
DEFINE_string(max_recv_bytes, "10MiB", "Maximum number of bytes for incoming messages");
DEFINE_validator(max_recv_bytes, &ValidateBytes);
DEFINE_int32(port, 50051, "Port to listen for gRPC requests");
DEFINE_int32(metrics, 50078, "Port to expose metrics for scraping");

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("flowers");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    FiberGroup<boost::fibers::algo::shared_work> fibers(FLAGS_execution_threads + 1);

    // initialize trt
    auto runtime    = std::make_shared<TensorRT::StandardRuntime>();
    int  batch_size = -1;

    using pool_t    = typename FlowersResources::pool_t;
    auto workspaces = pool_t::Create();

    for (int i = 0; i < FLAGS_contexts; i++)
    {
        auto model = runtime->deserialize_engine(FLAGS_engine);
        workspaces->emplace_push(model);
        LOG(INFO) << "bindings info" << std::endl << model->bindings_info();
    }

    // using pinned memory in place of shared memory
    auto pinned_alloc   = memory::make_allocator(memory::cuda_malloc_host_allocator());
    auto pinned_flowers = pinned_alloc.allocate_descriptor(1_GiB);

    // Initialize Resources
    LOG(INFO) << "Initializing Flowers Resources for RPC";
    auto rpcResources = std::make_shared<FlowersResources>(workspaces, static_cast<float*>(pinned_flowers.data()), FLAGS_execution_threads);

    // Create a gRPC server bound to IP:PORT
    std::ostringstream ip_port;
    ip_port << "0.0.0.0:" << FLAGS_port;
    Server server(ip_port.str());

    // Modify MaxReceiveMessageSize
    auto bytes = trtlab::StringToBytes(FLAGS_max_recv_bytes);
    server.Builder().SetMaxReceiveMessageSize(bytes);
    LOG(INFO) << "gRPC MaxReceiveMessageSize = " << trtlab::BytesToString(bytes);

    LOG(INFO) << "Register Service (flowers::Inference) with Server";
    auto inferenceService = server.RegisterAsyncService<Inference>();

    LOG(INFO) << "Register RPC (flowers::Inference::Compute) with Service (flowers::Inference)";
    auto rpcCompute = inferenceService->RegisterRPC<FlowersContext>(&Inference::AsyncService::RequestCompute);

    LOG(INFO) << "Initializing Executor";
    auto executor = server.RegisterExecutor(new FiberExecutor(1));

    LOG(INFO) << "Registering Execution Contexts for RPC (flowers::Inference::Compute) with Executor";
    executor->RegisterContexts(rpcCompute, rpcResources, 10);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(2000), [] {});
}

float* GetSharedMemory(const std::string& address)
{
    /* data in shared memory should go here - for the sake of quick examples just use and emptry
     * array */
    return nullptr;
    //std::memset(pinned_memory->Data(), (char)(0), pinned_memory->Size());
    //return (float*)pinned_memory->Data();
    // the following code connects to a shared memory service to allow for non-serialized transfers
    // between microservices
    /*
    InfoRequest request;
    Info reply;
    grpc::ClientContext context;
    auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
    auto stub = SharedMemoryDataSet::NewStub(channel);
    auto status = stub->GetInfo(&context, request, &reply);
    CHECK(status.ok()) << "Dataset shared memory request failed";
    DLOG(INFO) << "SysV ShmKey: " << reply.sysv_key();
    int shmid = shmget(reply.sysv_key(), 0, 0);
    DLOG(INFO) << "SysV ShmID: " << shmid;
    float* data = (float*) shmat(shmid, 0, 0);
    CHECK(data) << "SysV Attached failed";
    return data;
    */
}
