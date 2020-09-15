#include "test_jarvis.h"

#include "mel_banks.h"
#include "cuda_mel_banks.h"
#include "pipeline/mel_banks_compute.h"

TEST_F(TestJarvis, MelBanks)
{
    ConfigBuilder builder;
    Config        config = builder.get_config();
    MelBanks      mel_banks(config);

    ASSERT_EQ(mel_banks.GetBins().size(), 64);

    CudaMelBanks cuda_mel_banks(config);
}

TEST_F(TestJarvis, MelBankCompute)
{
    Config       config;
    CudaMelBanks mel_banks(config);

    cudaStream_t stream;
    CHECK_EQ(cudaStreamCreate(&stream), cudaSuccess);

    CuMatrix<float> power_spectrum;
    CuMatrix<float> new_features;

    int bin_size                = 64;
    int batch_size              = 2;
    int frames_per_audio_window = 11;

    power_spectrum.Resize(batch_size * frames_per_audio_window, bin_size, kUndefined, kStrideEqualNumCols);
    power_spectrum.SetZero();

    // load some feature from a numpy array

    new_features.Resize(batch_size * frames_per_audio_window, bin_size, kUndefined, kStrideEqualNumCols);
    new_features.SetZero();

    mel_banks_compute(new_features, power_spectrum, mel_banks, batch_size, frames_per_audio_window, stream);

    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    CHECK_EQ(cudaStreamDestroy(stream), cudaSuccess);
}