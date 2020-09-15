#pragma once
#include <vector>

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/descriptor.h>

#include "mel_banks.h"
#include "cuda_vector.h"

class CudaMelBanks : private MelBanks
{
public:
    CudaMelBanks(const Config&);
    ~CudaMelBanks();

    const float* const* vecs() const
    {
        return d_vecs_ptrs;
    }

    const std::int32_t* offsets() const
    {
        return d_offsets.Data();
    }

    const std::int32_t* sizes() const
    {
        return d_sizes.Data();
    }

private:
    std::vector<float*>       h_vecs;
    std::vector<std::int32_t> h_sizes;
    std::vector<std::int32_t> h_offsets;

    std::vector<CuVector<float>> d_vecs;
    CuVector<std::int32_t>       d_sizes;
    CuVector<std::int32_t>       d_offsets;
    float**                      d_vecs_ptrs;
};