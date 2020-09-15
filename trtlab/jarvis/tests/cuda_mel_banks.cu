
#include "cuda_mel_banks.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <iterator>

CudaMelBanks::CudaMelBanks(const Config& config)
: MelBanks(config)
{
    const auto& bins = MelBanks::GetBins();
    auto        size = bins.size();

    h_vecs.resize(size);
    h_sizes.resize(size);
    h_offsets.resize(size);
    d_vecs.resize(size);



    for (int i = 0; i < bins.size(); i++)
    {
        d_vecs[i].Resize(bins[i].second.size(), kUndefined);
        d_vecs[i].CopyFromVec(bins[i].second);
        h_vecs[i]    = d_vecs[i].Data();
        h_sizes[i]   = d_vecs[i].Dim();
        h_offsets[i] = bins[i].first;
    }

    d_sizes.CopyFromVec(h_sizes);
    d_offsets.CopyFromVec(h_offsets);

    CHECK_EQ(cudaMalloc((void**)&d_vecs_ptrs, size * sizeof(float*)), cudaSuccess);
    CHECK_EQ(cudaMemcpyAsync(d_vecs_ptrs, &h_vecs[0], size * sizeof(float*), cudaMemcpyHostToDevice, cudaStreamPerThread), cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
}

CudaMelBanks::~CudaMelBanks()
{
    CHECK_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
    CHECK_EQ(cudaFree(d_vecs_ptrs), cudaSuccess);

}
