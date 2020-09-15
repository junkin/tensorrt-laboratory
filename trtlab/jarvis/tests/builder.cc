#include "builder.h"

ConfigBuilder::ConfigBuilder()
: m_max_batch_size(256), m_batching_window_timeout_ms(1), m_max_pipeline_concurrency(8), m_max_thread_count(4)
{
}

void ConfigBuilder::set_acoustic_encoder_path(std::string encoder_path)
{
    m_encoder_path = encoder_path;
}

void ConfigBuilder::set_acoustic_decoder_path(std::string decoder_path)
{
    m_decoder_path = decoder_path;
}

void ConfigBuilder::set_max_batch_size(std::uint32_t batch_size)
{
    m_max_batch_size = batch_size;
}

void ConfigBuilder::set_batching_window_timeout_ms(std::uint32_t ms)
{
    m_batching_window_timeout_ms = ms;
}

void ConfigBuilder::set_max_pipeline_concurrency(std::uint32_t concurrency)
{
    m_max_pipeline_concurrency = concurrency;
}

void ConfigBuilder::set_max_thread_count(std::uint32_t count)
{
    m_max_thread_count = count;
}

Config ConfigBuilder::get_config()
{
    auto config = Config(
        // audio properties
        16000,                 // freq
        sizeof(std::uint16_t), // bytes per sample

        // audio buffer
        120, // audio buffer window size in ms
        10,  // audio buffer window overlap in ms
        5,

        // features (mel_opts)
        64,   // number of mel bin
        0.,   // low freq,
        0.,   // high freq,
        true, // normalize

        // features extraction per audio window (frame_opts)
        20,        // fft window size
        10,        // fft shift size
        0.00001,   // dither
        1.0,       // gain
        0.97,      // preemph,
        0.0,       // blackman
        "hanning", // window type
        // features buffer
        // shift_size = # of features in one audio window
        251, // feature buffer window size
        50,  // feature buffer overlap size,

        // acoustic tensorrt engines
        m_encoder_path,
        m_decoder_path,

        // batcher
        m_max_batch_size,             //
        m_batching_window_timeout_ms, //

        // pipeline
        m_max_pipeline_concurrency, //
        m_max_thread_count          //
    );

    VLOG(1) << config;
    return config;
}

std::ostream& operator<<(std::ostream& os, const Config& cfg)
{
    os << std::endl <<  "Jarvis ASR Configuration";
    os << std::endl <<  "========================";
    os << std::endl <<  "sample frequency                 : " << cfg.sample_freq;
    os << std::endl <<  "bytes per sample                 : " << cfg.bytes_per_sample;
    os << std::endl <<  "audio buffer";
    os << std::endl <<  "- window size    (ms ; bytes)    : " << cfg.audio_buffer_window_size_ms << " / " << cfg.audio_buffer_window_size_bytes;
    os << std::endl <<  "- window overlap (ms ; bytes)    : " << cfg.audio_buffer_window_overlap_ms << " / "
                                                                  << cfg.audio_buffer_window_overlap_bytes;
    os << std::endl <<  "- window count                   : " << cfg.audio_buffer_window_count;
    os << std::endl <<  "- minimum required allocation    : "
                                << "TODO";
    os << std::endl <<  "features extraction (performed on each audio buffer window)";
    os << std::endl <<  "- window size  (ms)              : " << cfg.extraction_frame_size_ms;
    os << std::endl <<  "- window shift (ms)              : " << cfg.extraction_frame_shift_ms;
    os << std::endl <<  "- dither                         : " << cfg.extraction_dither;
    os << std::endl <<  "- gain                           : " << cfg.extraction_gain;
    os << std::endl <<  "- preemph_coeff                  : " << cfg.extraction_preemph_coeff;
    os << std::endl <<  "- blackman_coeff                 : " << cfg.extraction_blackman_coeff;
    os << std::endl <<  "- window type                    : " << cfg.extraction_window_type;
    os << std::endl <<  "- fft length                     : " << cfg.extraction_fft_length;
    os << std::endl <<  "- features per audio buffer win  : " << cfg.features_per_audio_buffer_window;
    os << std::endl <<  "features (mels)";
    os << std::endl <<  "- bins per feature               : " << cfg.features_count;
    os << std::endl <<  "- low freq cutoff                : " << cfg.features_low_freq;
    os << std::endl <<  "- high freq cutoff (0 = none)    : " << cfg.features_high_freq;
    os << std::endl <<  "- normalize                      : " << (cfg.features_normalize ? "TRUE" : "FALSE");
    os << std::endl <<  "- bytes per feature              : " << cfg.bytes_per_feature;

    os << std::endl <<  "features buffer";
    os << std::endl <<  "- window size    (feats ; bytes) : " << cfg.features_buffer_window_size_feats << "; "
                                                              << cfg.features_buffer_window_size_bytes;
    os << std::endl <<  "- window overlap (feats ; bytes) : " << cfg.features_buffer_window_overlap_feats << "; "
                                                              << cfg.features_buffer_window_overlap_bytes;
    os << std::endl <<  "- window count                   : " << cfg.features_buffer_window_count;
    os << std::endl <<  "runtime";
    os << std::endl <<  "- max batch size                 : " << cfg.max_batch_size;
    os << std::endl <<  "- batching window timeout (ms)   : " << cfg.batching_window_timeout_ms;
    os << std::endl <<  "- max concurrency                : " << cfg.max_concurrency;
    os << std::endl <<  "- thread count                   : " << cfg.thread_count;
    os << std::endl;
}