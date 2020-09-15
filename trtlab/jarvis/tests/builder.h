#pragma once

#include "config.h"

struct ConfigBuilder
{
    ConfigBuilder();

    Config get_config();

    void set_acoustic_encoder_path(std::string);
    void set_acoustic_decoder_path(std::string);

    // batching
    void set_max_batch_size(std::uint32_t);
    void set_batching_window_timeout_ms(std::uint32_t);

    // number of pipeline objects in the pool
    void set_max_pipeline_concurrency(std::uint32_t);

    // std::thread count
    void set_max_thread_count(std::uint32_t);


private:
    // tensorrt engines
    std::string m_encoder_path;
    std::string m_decoder_path;

    // batching
    std::size_t m_max_batch_size;
    std::size_t m_batching_window_timeout_ms;

    // concurrency
    std::size_t m_max_pipeline_concurrency;
    std::size_t m_max_thread_count;
};