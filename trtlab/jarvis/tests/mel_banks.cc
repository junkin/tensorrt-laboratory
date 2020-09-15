#include "mel_banks.h"

#include <cmath>

MelBanks::MelBanks(const Config& cfg) 
{
    std::int32_t num_bins    = cfg.features_count;
    float        sample_freq = cfg.sample_freq;
    float        nyquist     = 0.5 * sample_freq;
    bool         normalize   = cfg.features_normalize;

    float low_freq = cfg.features_low_freq;
    float high_freq;
    if (cfg.features_high_freq > 0.0)
        high_freq = cfg.features_high_freq;
    else
        high_freq = nyquist + cfg.features_high_freq;

    int num_fft_frequencies = cfg.extraction_fft_length / 2 + 1;

    std::vector<float> fft_frequencies, mel_frequencies;
    FftFrequencies(fft_frequencies, sample_freq, cfg.extraction_fft_length);
    MelFrequencies(mel_frequencies, num_bins + 2, low_freq, high_freq);

    std::vector<float> delta_frequencies(num_bins + 1);
    for (int bin = 0; bin < num_bins + 1; bin++)
    {
        delta_frequencies[bin] = mel_frequencies[bin + 1] - mel_frequencies[bin];
    }

    std::vector<std::vector<float>> ramps(num_bins + 2);
    for (int bin = 0; bin < num_bins + 2; bin++)
    {
        ramps[bin].resize(num_fft_frequencies);
        for (int freq = 0; freq < num_fft_frequencies; ++freq)
        {
            ramps[bin][freq] = (mel_frequencies[bin] - fft_frequencies[freq]);
        }
    }

    bins_.resize(num_bins);
    for (int bin = 0; bin < num_bins; bin++)
    {
        std::vector<float> non_zero_weights;
        int                first_index = -1;
        for (int freq = 0; freq < num_fft_frequencies; freq++)
        {
            float lower = -ramps[bin][freq] / delta_frequencies[bin];
            float upper = ramps[bin + 2][freq] / delta_frequencies[bin + 1];

            float min = std::min(lower, upper);
            if (min > 0.)
            {
                non_zero_weights.push_back(min);
                if (first_index < 0)
                    first_index = freq;
            }
        }

        bins_[bin].first = first_index;
        bins_[bin].second.resize(non_zero_weights.size());

        std::copy(non_zero_weights.begin(), non_zero_weights.end(), bins_[bin].second.begin());
    }

    if (normalize)
    {
        std::vector<float> enorm(num_bins);
        for (int bin = 0; bin < num_bins; ++bin)
        {
            enorm[bin] = 2.0 / (mel_frequencies[bin + 2] - mel_frequencies[bin]);
        }

        for (int bin = 0; bin < num_bins; ++bin)
        {
            int size = bins_[bin].second.size();
            for (int i = 0; i < size; ++i)
            {
                bins_[bin].second[i] *= enorm[bin];
            }
        }
    }
}

void MelBanks::FftFrequencies(std::vector<float>& fft_frequencies, float sample_rate, std::int32_t fft_length)
{
    int num_frequencies = fft_length / 2 + 1;
    fft_frequencies.resize(num_frequencies);
    for (int i = 0; i < num_frequencies; ++i)
    {
        fft_frequencies[i] = 1.0 * i / (num_frequencies - 1.) * sample_rate / 2.;
    }
}

void MelBanks::MelFrequencies(std::vector<float>& mel_frequencies, std::int32_t nmels_plus_two, float low_freq, float high_freq)
{
    const float min_mel = HzToMel(low_freq);
    const float max_mel = HzToMel(high_freq);

    mel_frequencies.resize(nmels_plus_two);

    for (int i = 0; i < mel_frequencies.size(); ++i)
    {
        float mel_freq     = min_mel + 1.0 * i / (mel_frequencies.size() - 1.) * (max_mel - min_mel);
        mel_frequencies[i] = MelToHz(mel_freq);
    }
}

float MelBanks::HzToMel(float frequency)
{
    // Use stanley equation
    constexpr float kMinLogHz = 1000.0f;       // Break frequency (Hz)
    constexpr float kStep     = 200.0f / 3.0f; // Step size below break frequency
    float           mel;

    if (frequency < kMinLogHz)
    {
        mel = frequency / kStep;
    }
    else
    {
        // Fill in the log region
        constexpr float kMinLogMel = kMinLogHz / kStep;      // Break frequency (Mel)
        constexpr float kLogStep   = std::log(6.4f) / 27.0f; // step size above break frequency

        mel = kMinLogMel + std::log(frequency / kMinLogHz) / kLogStep;
    }

    return mel;
}

float MelBanks::MelToHz(float mel)
{
    // Use stanley equation
    constexpr float kMinLogHz  = 1000.0f;                // Break frequency (Hz)
    constexpr float kStep      = 200.0f / 3.0f;          // Step size below break frequency
    constexpr float kMinLogMel = kMinLogHz / kStep;      // Break frequency (Mels)
    constexpr float kLogStep   = std::log(6.4f) / 27.0f; // step size above break frequency

    float frequency = mel * kStep;
    // Fill in the log region
    if (mel >= kMinLogMel)
    {
        frequency = kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel));
    }
    return frequency;
}
