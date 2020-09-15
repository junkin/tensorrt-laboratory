#include "test_jarvis.h"

TEST_F(TestJarvis, Config)
{
    Config dc;

    ConfigBuilder builder;

    builder.set_max_batch_size(2);
    builder.set_batching_window_timeout_ms(10000); // 10sec
    builder.set_max_pipeline_concurrency(1);
    builder.set_max_thread_count(1);

    Config cc = builder.get_config();

    ASSERT_EQ(dc.audio_buffer_window_size_ms, cc.audio_buffer_window_size_ms);

    ASSERT_NE(dc.max_batch_size, cc.max_batch_size);
    ASSERT_NE(dc.batching_window_timeout_ms, cc.batching_window_timeout_ms);
    ASSERT_NE(dc.max_concurrency, cc.max_concurrency);
    ASSERT_NE(dc.thread_count, cc.thread_count);
}