#include "test_client.h"
#include "test_context.h"

#include <nvrpc/executor.h>
#include <nvrpc/server.h>

using Input = SpeechSquadInferRequest;
using Output = SpeechSquadInferResponse;

using namespace demo;
using namespace demo::testing;

class SpeechSquadTests : public ::testing::Test {};

TEST_F(SpeechSquadTests, BasicService)
{
    // create and start server
    auto server = std::make_unique<nvrpc::Server>("0.0.0.0:13377");
    auto resources = std::make_shared<SpeechSquadTestResources>();
    auto executor = server->RegisterExecutor(new nvrpc::Executor(1));
    auto service = server->RegisterAsyncService<SpeechSquadService>();
    auto rpc_streaming = service->RegisterRPC<SpeechSquadTextContext>(&SpeechSquadService::AsyncService::RequestSpeechSquadInfer);
    executor->RegisterContexts(rpc_streaming, resources, 10);
    server->AsyncStart();
    EXPECT_TRUE(server->Running());

    // create client
    auto client_executor = std::make_shared<nvrpc::client::Executor>(1);

    auto channel = grpc::CreateChannel("localhost:13377", grpc::InsecureChannelCredentials());
    std::shared_ptr<SpeechSquadService::Stub> stub = SpeechSquadService::NewStub(channel);

    auto infer_prepare_fn = [stub](::grpc::ClientContext * context,
                                   ::grpc::CompletionQueue * cq) -> auto
    {
        return std::move(stub->PrepareAsyncSpeechSquadInfer(context, cq));
    };

    auto client = std::make_unique<TestClientStreaming>(infer_prepare_fn, client_executor);
    client->SetCorked(true);

    // send initial config request
    SpeechSquadInferRequest request;
    auto config = request.mutable_speech_squad_config();
    client->Write(std::move(request));

    // dummy audio data
    static char bytes[128];
    std::memset(bytes, 0, 128); 

    // send audio data
    auto count = client->count();
    for(int i=0; i<count; i++)
    {
        SpeechSquadInferRequest request;
        request.set_audio_content(bytes, 128);
        client->Write(std::move(request));
    }

    auto future = client->Done();
    auto status = future.get();

    DLOG(INFO) << "client complete";
    EXPECT_TRUE(status.ok());

    server->Shutdown();
    EXPECT_FALSE(server->Running());
}