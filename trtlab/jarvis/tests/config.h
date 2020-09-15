
#pragma once

#include <glog/logging.h>

static std::uint32_t next_pow2(std::uint32_t v)
{
    DCHECK_GT(v, 0);
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}

struct Config
{
    // clang-format off
    
    Config(
        std::size_t _sample_freq                    = 16000, 
        std::size_t _bytes_per_sample               = sizeof(std::uint16_t),
        // audio buffer
        std::size_t _audio_buffer_window_size_ms    = 120, /* ms */
        std::size_t _audio_buffer_window_overlap_ms = 10,  /* ms */
        std::size_t _audio_buffer_window_count      = 5,
        // features (mel_opts)
        std::size_t _features_count                 = 64,  /* mel_opt num_bins */
        float       _features_low_freq              = 0.,
        float       _features_high_freq             = 0.,
        bool        _features_normalize             = true,
        // features extraction per audio window (frame_opts)
        std::size_t _extraction_frame_size_ms       = 20,  /* ms */
        std::size_t _extraction_frame_shift_ms      = 10,  /* ms */
        float       _extraction_dither              = 0., //0.00001,
        float       _extraction_gain                = 1.0,
        float       _extraction_preemph_coeff       = 0.97,
        float       _extraction_blackman_coeff      = 0.0,
        std::string _extraction_window_type         = "hanning",
        // features buffer
        // shift_size = # of features in one audio window
        std::size_t _features_buffer_window_size    = 251,
        std::size_t _features_buffer_window_count   = 50,
        // tensorrt acoustic models
        std::string _acoustic_encoder_path          = "encoder.engine",
        std::string _acoustic_decoder_path          = "decoder.engine",
        // batcher
        std::size_t _max_batch_size                 = 256,
        std::size_t _batching_window_timeout_ms     = 1,
        // pipeline
        std::size_t _max_concurrency                = 8,
        // threads
        std::size_t _thread_count                   = 4
    )
    : sample_freq(_sample_freq),
      sample_freq_ms(_sample_freq / 1000),
      bytes_per_sample(_bytes_per_sample),

      // audio buffer (processed)
      audio_buffer_window_size_ms(_audio_buffer_window_size_ms),
      audio_buffer_window_size_samples(audio_buffer_window_size_ms * sample_freq_ms),
      audio_buffer_window_size_bytes(audio_buffer_window_size_samples * bytes_per_sample),
      audio_buffer_window_overlap_ms(_audio_buffer_window_overlap_ms),
      audio_buffer_window_overlap_samples(audio_buffer_window_overlap_ms * sample_freq_ms),
      audio_buffer_window_overlap_bytes(audio_buffer_window_overlap_samples * bytes_per_sample),
      audio_buffer_window_count(_audio_buffer_window_count),

      // audio buffer (preprocessed)
      // we need an extra sample to get preemph correct
      audio_buffer_preprocessed_window_size_samples(audio_buffer_window_size_samples + 1),
      audio_buffer_preprocessed_window_size_bytes(audio_buffer_preprocessed_window_size_samples * bytes_per_sample),
      audio_buffer_preprocessed_window_overlap_samples(audio_buffer_window_overlap_samples + 1),
      audio_buffer_preprocessed_window_overlap_bytes(audio_buffer_preprocessed_window_overlap_samples * bytes_per_sample),

      // features
      features_count(_features_count),
      features_low_freq(_features_low_freq),
      features_high_freq(_features_high_freq),
      features_normalize(_features_normalize),
      bytes_per_feature(features_count * sizeof(float)),
      // features extraction per audio window
      extraction_frame_size_ms(_extraction_frame_size_ms),
      extraction_frame_shift_ms(_extraction_frame_shift_ms),
      extraction_dither(_extraction_dither),
      extraction_gain(_extraction_gain),
      extraction_preemph_coeff(_extraction_preemph_coeff),
      extraction_blackman_coeff(_extraction_blackman_coeff),
      extraction_window_type(_extraction_window_type),
      extraction_fft_length(next_pow2(extraction_frame_size_ms * sample_freq_ms)),
      features_per_audio_buffer_window((audio_buffer_window_size_ms - extraction_frame_size_ms) / extraction_frame_shift_ms + 1),
      // features buffer
      features_buffer_window_size_feats(_features_buffer_window_size),
      features_buffer_window_size_bytes(_features_buffer_window_size * bytes_per_feature),
      features_buffer_window_overlap_feats(_features_buffer_window_size - features_per_audio_buffer_window),
      features_buffer_window_overlap_bytes(features_buffer_window_overlap_feats * bytes_per_feature),
      features_buffer_window_count(_features_buffer_window_count),
      // acoustic tensorrt engines
      acoustic_encoder_path(_acoustic_encoder_path),
      acoustic_decoder_path(_acoustic_decoder_path),
      // batching
      max_batch_size(_max_batch_size),
      batching_window_timeout_ms(_batching_window_timeout_ms),
      // concurrency
      max_concurrency(_max_concurrency),
      // threads
      thread_count(_thread_count)
    {
;
    }

    // clang-format on

    const std::size_t sample_freq;
    const std::size_t sample_freq_ms;
    const std::size_t bytes_per_sample;

    // audio buffer
    const std::size_t audio_buffer_window_size_ms;
    const std::size_t audio_buffer_window_size_samples;
    const std::size_t audio_buffer_window_size_bytes;
    const std::size_t audio_buffer_window_overlap_ms;
    const std::size_t audio_buffer_window_overlap_samples;
    const std::size_t audio_buffer_window_overlap_bytes;
    const std::size_t audio_buffer_window_count;

    // to get preemph correct, we need an extra sample in the buffer window and overlap
    // after processing, we will have a window size and overlap as defined by
    // audio_buffer_window_size_samples / audio_buffer_window_overlap_samples
    const std::size_t audio_buffer_preprocessed_window_size_samples;
    const std::size_t audio_buffer_preprocessed_window_size_bytes;
    const std::size_t audio_buffer_preprocessed_window_overlap_samples;
    const std::size_t audio_buffer_preprocessed_window_overlap_bytes;

    // features
    const std::size_t features_count;
    const float       features_low_freq;
    const float       features_high_freq;
    const bool        features_normalize;
    const std::size_t bytes_per_feature;
    // features per window from audio buffer
    const std::size_t extraction_frame_size_ms;
    const std::size_t extraction_frame_shift_ms;
    const float       extraction_dither;
    const float       extraction_gain;
    const float       extraction_preemph_coeff;
    const float       extraction_blackman_coeff;
    const std::string extraction_window_type;
    const std::size_t extraction_fft_length;
    const std::size_t features_per_audio_buffer_window;
    // features buffer
    const std::size_t features_buffer_window_size_feats;
    const std::size_t features_buffer_window_size_bytes;
    const std::size_t features_buffer_window_overlap_feats;
    const std::size_t features_buffer_window_overlap_bytes;
    const std::size_t features_buffer_window_count;
    // acoustic models
    const std::string acoustic_encoder_path;
    const std::string acoustic_decoder_path;
    // batching
    const std::size_t max_batch_size;
    const std::size_t batching_window_timeout_ms;
    // concurrency
    const std::size_t max_concurrency;
    // threads
    const std::size_t thread_count;

    friend std::ostream& operator<<(std::ostream& os, const Config& cfg);
};