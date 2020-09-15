#include "test_jarvis.h"

#include <trtlab/core/fiber_group.h>

#include <trtlab/cuda/common.h>
#include <trtlab/cuda/sync.h>
#include <trtlab/cuda/memory/cuda_allocators.h>

#include <trtlab/tensorrt/runtime.h>
#include <trtlab/tensorrt/model.h>
#include <trtlab/tensorrt/execution_context.h>
#include <trtlab/tensorrt/workspace.h>

#include <trtlab/core/pool.h>

#include "NvInfer.h"

using namespace trtlab;

struct instance
{
    cudaStream_t                                stream;
    memory::descriptor                          inputs;
    memory::descriptor                          outputs;
    std::shared_ptr<TensorRT::ExecutionContext> context;
    std::vector<void*>                          bindings;
};

TEST_F(TestJarvis, LoadModel)
{
    FiberGroup fibers(2);
    auto       runtime     = std::make_shared<TensorRT::StandardRuntime>();
    int        concurrency = 4;
    int        batch_size  = -1;

    std::vector<std::unique_ptr<TensorRT::StaticSingleModelGraphWorkspace>> workspaces;

    for (int i = 0; i < concurrency; i++)
    {
        auto model = runtime->deserialize_engine("/work/flowers-152-b08-int8-a100.engine");
        workspaces.push_back(std::make_unique<TensorRT::StaticSingleModelGraphWorkspace>(model));
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto end   = start + std::chrono::seconds(10);

    std::vector<typename userspace_threads::future<std::size_t>> futures;

    for (int i = 0; i < workspaces.size(); i++)
    {
        futures.push_back(userspace_threads::async([&workspaces, i, end ]() -> auto {
            auto&       workspace = workspaces[i];
            std::size_t batches   = 0;
            while (std::chrono::high_resolution_clock::now() < end)
            {
                workspace->enqueue();
                cuda_sync<userspace_threads>::stream_sync(workspace->stream());
                batches++;
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

    auto inferences = batches * workspaces[0]->batch_size();

    LOG(INFO) << workspaces[0]->batch_size();
    LOG(INFO) << "batches: " << batches;
    LOG(INFO) << "inf/sec " << inferences / std::chrono::duration<double>(end - start).count();

    LOG(INFO) << "finished";
}

/*
TEST_F(TestJarvis, LoadModel)
{
    auto runtime = std::make_shared<TensorRT::StandardRuntime>();
    auto pool    = Pool<TensorRT::ExecutionContext>::Create();

    auto cuda_malloc = memory::make_cuda_allocator();

    nvinfer1::Dims input_dims;
    nvinfer1::Dims output_dims;

    std::vector<instance> instances;
    instances.resize(4);

    for (int idx = 0; idx < instances.size(); idx++)
    {
        LOG(INFO) << "model " << idx;

        //auto  model  = runtime->deserialize_engine("/work/trtlab/jarvis/models/jasper_encoder.engine");
        auto  model  = runtime->deserialize_engine("/work/flowers-152-b08-int8-a100.engine");
        auto& engine = model->engine();
        LOG(INFO) << "has implicit batch: " << (engine.hasImplicitBatchDimension() ? "TRUE" : "FALSE");

        LOG(INFO) << model->profiles_info();

        input_dims = engine.getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kOPT);

        // we can only create 1 execution context per optimization profile
        for (int i = 0; i < 1; i++)
        {
            TensorRT::ExecutionContext ctx(model);
            //ctx.context().setOptimizationProfile(0);
            //ctx.context().setBindingDimensions(0, input_dims);
            output_dims = ctx.context().getBindingDimensions(1);

            LOG(INFO) << "execution context device memory requirements: " << ctx.context().getEngine().getDeviceMemorySize();
            ctx.set_device_memory(cuda_malloc.allocate_descriptor(ctx.context().getEngine().getDeviceMemorySize()));
            pool->Push(std::move(ctx));
        }

        instance inst;

        LOG(INFO) << "input_dims: " << TensorRT::Model::dims_info(input_dims);
        LOG(INFO) << "output_dims: " << TensorRT::Model::dims_info(output_dims);

        CHECK_CUDA(cudaStreamCreate(&inst.stream));
        inst.inputs =
            cuda_malloc.allocate_descriptor(input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * sizeof(float));
        inst.outputs = cuda_malloc.allocate_descriptor(output_dims.d[0] * output_dims.d[1] * sizeof(float));
        inst.context = pool->Pop();

        inst.bindings.push_back(inst.inputs.data());
        inst.bindings.push_back(inst.outputs.data());

        CHECK_EQ(inst.inputs.device_context().device_type, kDLGPU);

        instances[idx] = std::move(inst);
    }

    std::size_t batches = 0;

    auto start = std::chrono::high_resolution_clock::now();
    auto end   = start + std::chrono::seconds(10);

    while (std::chrono::high_resolution_clock::now() < end)
    {
        for (int idx = 0; idx < instances.size(); idx++)
        {
            instances[idx].context->context().enqueueV2(instances[idx].bindings.data(), instances[idx].stream, nullptr);
        }
        for (int idx = 0; idx < instances.size(); idx++)
        {
            CHECK_CUDA(cudaStreamSynchronize(instances[idx].stream));
            batches++;
        }
    }

    end = std::chrono::high_resolution_clock::now();

    auto inferences = batches * input_dims.d[0];

    LOG(INFO) << TensorRT::Model::dims_info(input_dims);

    LOG(INFO) << "batches: " << batches;
    LOG(INFO) << "inf/sec " << inferences / std::chrono::duration<double>(end - start).count();

    LOG(INFO) << "finished";
}
*/