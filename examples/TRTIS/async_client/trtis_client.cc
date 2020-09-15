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
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/malloc.h"
#include "trtlab/tensorrt/model.h"
#include "trtlab/tensorrt/utils.h"

// NVIDIA Inference Server Protos
#include "trtlab/trtis/protos/grpc_service.grpc.pb.h"
#include "trtlab/trtis/protos/grpc_service.pb.h"

#include "nvrpc/client/client_unary.h"
#include "nvrpc/client/executor.h"

using trtlab::Allocator;
using trtlab::Malloc;
using trtlab::TensorRT::SizeofDataType;

namespace trtis
{

namespace protos = ::nvidia::inferenceserver;

class Model;
class InferRunner;

class InferenceManager
{
  public:
    struct Builder
    {
        std::string hostname;
        int thread_count;
    };

    InferenceManager(Builder&& builder)
    {
        ::grpc::ChannelArguments ch_args;
        ch_args.SetMaxReceiveMessageSize(-1);
        m_Channel =
            grpc::CreateCustomChannel(builder.hostname, grpc::InsecureChannelCredentials(), ch_args);
        m_Stub = protos::GRPCService::NewStub(m_Channel);
        m_Executor = std::make_shared<::nvrpc::client::Executor>(builder.thread_count);
    }

    std::vector<std::string> Models()
    {
        const auto& status = Status();
        auto model_status = status.server_status().model_status();
        DLOG(INFO) << status.DebugString();
        const protos::ModelConfig* model_config;
        std::vector<std::string> models;
        m_Models.clear();
        for(auto it = model_status.begin(); it != model_status.end(); it++)
        {
            DLOG(INFO) << it->first;
            models.push_back(it->first);
            m_Models[it->first] = std::make_shared<Model>(it->second.config());
        }
        return models;
    }

    std::shared_ptr<InferRunner> Runner(const std::string& model_name)
    {
        auto infer_prepare_fn = [this](::grpc::ClientContext * context,
                                       const protos::InferRequest& request,
                                       ::grpc::CompletionQueue* cq) -> auto
        {
            return std::move(m_Stub->PrepareAsyncInfer(context, request, cq));
        };

        auto runner = std::make_unique<
            ::nvrpc::client::ClientUnary<protos::InferRequest, protos::InferResponse>>(
            infer_prepare_fn, m_Executor);

        return std::make_shared<InferRunner>(GetModel(model_name), std::move(runner));
    }

    std::shared_ptr<Model> GetModel(const std::string& name) const
    {
        auto search = m_Models.find(name);
        LOG_IF(FATAL, search == m_Models.end()) << "Model: " << name << " not found";
        return search->second;
    }

  protected:
    protos::StatusResponse Status()
    {
        ::grpc::ClientContext context;
        protos::StatusRequest request;
        protos::StatusResponse response;
        auto status = m_Stub->Status(&context, request, &response);
        CHECK(status.ok());
        return response;
    }

  private:
    std::string m_Hostname;
    std::map<std::string, std::shared_ptr<Model>> m_Models;
    std::shared_ptr<::grpc::Channel> m_Channel;
    std::unique_ptr<protos::GRPCService::Stub> m_Stub;
    std::shared_ptr<::nvrpc::client::Executor> m_Executor;
};

struct Model : public ::trtlab::TensorRT::BaseModel
{
    Model(const protos::ModelConfig& model)
    {
        SetName(model.name());
        m_MaxBatchSize = model.max_batch_size();
        for(int i = 0; i < model.input_size(); i++)
        {
            const auto& b = model.input(i);
            TensorBindingInfo binding;
            binding.name = b.name();
            binding.isInput = true;
            binding.dtype = nvinfer1::DataType::kFLOAT;
            binding.dtypeSize =
                ::trtlab::TensorRT::SizeofDataType(binding.dtype); // TODO: map trtis DataType enum; model.data_type()
            size_t count = 1;
            for(int j = 0; j < b.dims_size(); j++)
            {
                auto val = b.dims(j);
                binding.dims.push_back(val);
                count *= val;
            }

            binding.elementsPerBatchItem = count;
            binding.bytesPerBatchItem = count * binding.dtypeSize;
            AddBinding(std::move(binding));
        }
        for(int i = 0; i < model.output_size(); i++)
        {
            const auto& b = model.output(i);
            TensorBindingInfo binding;
            binding.name = b.name();
            binding.isInput = false;
            binding.dtype = nvinfer1::DataType::kFLOAT;
            binding.dtypeSize =
                SizeofDataType(binding.dtype); // TODO: map trtis DataType enum; model.data_type()
            size_t count = 1;
            for(int j = 0; j < b.dims_size(); j++)
            {
                auto val = b.dims(j);
                binding.dims.push_back(val);
                count *= val;
            }

            binding.elementsPerBatchItem = count;
            binding.bytesPerBatchItem = count * binding.dtypeSize;
            AddBinding(std::move(binding));
        }
    }
    ~Model() override {}

    int GetMaxBatchSize() const final override { return m_MaxBatchSize; }

  private:
    int m_MaxBatchSize;
};

struct InferRunner
{
    InferRunner(
        std::shared_ptr<Model> model,
        std::unique_ptr<::nvrpc::client::ClientUnary<protos::InferRequest, protos::InferResponse>>
            runner)
        : m_Model(model), m_Runner(std::move(runner))
    {
    }

    using InferResults = float;
    using InferFuture = std::shared_future<InferResults>;

    const ::trtlab::TensorRT::BaseModel& GetModel() const { return *m_Model; }

    InferFuture Infer()
    {
        static auto bytes = Allocator<Malloc>(10 * 1024 * 1024);
        const auto& model = GetModel();
        int batch_size = 1; // will be infered from the input tensors

        // Build InferRequest
        protos::InferRequest request;
        request.set_model_name(model.Name());
        request.set_model_version(-1);
        auto meta_data = request.mutable_meta_data();

        for(auto id : model.GetInputBindingIds())
        {
            const auto& binding = model.GetBinding(id);
            if(binding.isInput)
            {
                auto size = binding.bytesPerBatchItem * batch_size;
                request.add_raw_input(bytes.Data(), size);
                auto meta = meta_data->add_input();
                meta->set_name(binding.name);
                meta->set_batch_byte_size(size);
                for(int i=0; i<binding.dims.size(); i++)
                {
                    meta->add_dims(binding.dims[i]);
                }
            }
        }

        for(auto id : model.GetOutputBindingIds())
        {
            const auto& binding = model.GetBinding(id);
            auto meta = meta_data->add_output();
            meta->set_name(binding.name);
        }

        meta_data->set_batch_size(batch_size);

        auto start = std::chrono::high_resolution_clock::now();
        return m_Runner->Enqueue(std::move(request),
                                 [this, start](protos::InferRequest& request,
                                               protos::InferResponse& response,
                                               ::grpc::Status& status) -> float {
                                    return std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();
                                 });
    }

  private:
    std::shared_ptr<Model> m_Model;
    std::unique_ptr<::nvrpc::client::ClientUnary<protos::InferRequest, protos::InferResponse>>
        m_Runner;
};

}

DEFINE_string(hostname, "localhost:8001", "hostname:port to send infer requests");
DEFINE_int32(count, 100, "number of infer requests to send");
DEFINE_int32(thread_count, 1, "Size of thread pool");

int main(int argc, char *argv[])
{
   FLAGS_alsologtostderr = 1; // It will dump to console
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    trtis::InferenceManager::Builder builder;
    builder.hostname = FLAGS_hostname;
    builder.thread_count = FLAGS_thread_count;

    trtis::InferenceManager manager(std::move(builder));

    for (const auto& m : manager.Models())
    {
        LOG(INFO) << "Model: " << m;
    }

    auto bert = manager.Runner("bert1");

    std::vector<std::shared_future<float>> futures;
    futures.reserve(FLAGS_count);

    bert->Infer().get();

    LOG(INFO) << "Start Requests";
    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<FLAGS_count; i++)
    {
        futures.push_back(bert->Infer());
    }

    LOG(INFO) << "Wait on Requests";
    for(int i=0; i<FLAGS_count; i++)
    {
        futures[i].get();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration<float>(end - start).count();
    LOG(INFO) << "Finished in " << time;
    LOG(INFO) << "Inf/sec: " << (float) FLAGS_count / time;
    return 0;
}
