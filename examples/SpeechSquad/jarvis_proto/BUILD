package(default_visibility = ["//visibility:public"])

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

cc_grpc_library(
    name = "audio_proto",
    srcs = [
        ":audio.proto"
    ],
    deps = [],
)

cc_grpc_library(
    name = "jarvis_grpc_asr",
    srcs = [":jarvis_asr.proto"],
    deps = [":audio_proto"],
)

cc_grpc_library(
    name = "jarvis_grpc_tts",
    srcs = [":jarvis_tts.proto"],
    deps = [":audio_proto"]
)

cc_grpc_library(
    name = "jarvis_grpc_core_nlp",
    srcs = [":jarvis_nlp_core.proto"],
    deps = []
)

cc_grpc_library(
    name = "jarvis_grpc_nlp",
    srcs = [":jarvis_nlp.proto"],
    deps = [":jarvis_grpc_core_nlp"]
)

cc_grpc_library(
    name = "jarvis_grpc_health",
    srcs = [":health.proto"],
    deps = []
)
