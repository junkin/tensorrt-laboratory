

find_package(gflags 2.2.1 REQUIRED)

add_executable(birds-server.x
    server.cc
)

target_link_libraries(birds-server.x
    nvrpc
    trtlab::tensorrt
    birds-protos
    gflags
)