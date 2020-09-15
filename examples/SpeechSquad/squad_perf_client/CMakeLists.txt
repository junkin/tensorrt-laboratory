set(
  SQUAD_PERF_CLIENT_SRCS
  squad_perf_client.cc
  audio_task.cc
  speech_squad_client.cc
  squad_eval_dataset.cc
  status.cc
  utils.cc
)

set(
  SQUAD_PERF_CLIENT_HDRS
  audio_task.h
  speech_squad_client.h
  squad_eval_dataset.h
  status.h
  stream.h
  sync_queue.h
  thread-pool.h
  utils.h
)

add_executable(
  squad_perf_client
  ${SQUAD_PERF_CLIENT_SRCS}
  ${SQUAD_PERF_CLIENT_HDRS}
)

target_link_libraries(squad_perf_client
  PRIVATE
    trtlab::nvrpc
    trtlab::nvrpc-client
    jarvis_protos
    speech_squad_protos
    gflags
)

find_package(RapidJSON CONFIG REQUIRED)
target_include_directories(squad_perf_client
  PUBLIC
    ${RapidJSON_INCLUDE_DIRS}
)


install(
  TARGETS squad_perf_client
  RUNTIME DESTINATION bin
)