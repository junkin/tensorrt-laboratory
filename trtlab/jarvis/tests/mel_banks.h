// feat/mel-computations.h

// Copyright 2009-2011  Phonexia s.r.o.;  Microsoft Corporation
//                2016  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef MEL_COMPUTATIONS_H_
#define MEL_COMPUTATIONS_H_

#include <vector>

#include "config.h"

class MelBanks
{
public:
    MelBanks(const Config&);

    void FftFrequencies(std::vector<float>& fft_frequencies, float sample_rate, std::int32_t fft_length);

    void MelFrequencies(std::vector<float>& mel_frequencies, std::int32_t nmels_plus_two, float low_freq, float high_freq);

    float HzToMel(float frequency);

    float MelToHz(float mel);

    const std::vector<std::pair<std::int32_t, std::vector<float>>>& GetBins() const
    {
        return bins_;
    }

private:
    // the "bins_" vector is a vector, one for each bin, of a pair:
    // (the first nonzero fft-bin), (the vector of weights).
    std::vector<std::pair<std::int32_t, std::vector<float>>> bins_;
};

#endif // MEL_COMPUTATIONS_H_
