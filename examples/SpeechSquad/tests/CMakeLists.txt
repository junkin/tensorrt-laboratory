
find_package(GTest)

add_executable(test_speech_squad
  test_context.cc
  test_service.cc
)

target_link_libraries(test_speech_squad
  PRIVATE
    speech_squad
    GTest::gtest_main
)

add_test(
  NAME speech_squad
  COMMAND $<TARGET_FILE:test_speech_squad
)
