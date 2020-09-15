#include "cuda_matrix.h"
#include "cuda_vector.h"
#include <fstream>
#include <iostream>

#include <glog/logging.h>

/// Constructor with memory initialisation
template <typename T>
CuVector<T>::CuVector(MatrixIndexT size, MatrixResizeType resize_type) : data_(NULL), dim_(0), capacity_(0)
{
    Resize(size, resize_type);
}

template <typename T>
void CuVector<T>::Print(std::string& filename)
{
    std::ofstream file;

    std::vector<T> host(dim_);
    CHECK_EQ(cudaMemcpy(&host[0], data_, dim_ * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);

    file.open(filename);
    for (int col = 0; col < dim_; col++)
    {
        file << host[col];
        if (col != dim_ - 1)
        {
            file << std::endl;
        }
    }
    file.close();
}

template <typename T>
void CuVector<T>::SetZero()
{
    if (dim_ == 0 || data_ == NULL)
        return;

    assert(data_ != NULL);
    CHECK_EQ(cudaMemsetAsync(data_, 0, dim_ * sizeof(T), stream_), cudaSuccess);
}

template <typename T>
void CuVector<T>::CopyFromVec(const std::vector<T>& vec)
{
    this->Resize(vec.size());

    CHECK_EQ(cudaMemcpyAsync(this->data_, &vec[0], vec.size() * sizeof(T), cudaMemcpyHostToDevice, stream_), cudaSuccess);
}

template <typename T>
void CuVector<T>::CheckIfChangedAndCopy(std::vector<T>& new_vector, std::vector<T>& old_vector, int n)
{
    bool changed = false;
    for (int i = 0; i < n; i++)
    {
        if (new_vector[i] != old_vector[i])
        {
            old_vector[i] = new_vector[i];
            changed       = true;
        }
    }
    if (changed)
    {
        this->CopyFromVec(old_vector);
    }
}

template <typename T>
void CuVector<T>::Resize(MatrixIndexT dim, MatrixResizeType t)
{
    assert(t == kSetZero || t == kUndefined);

    if (dim <= this->capacity_)
    {
        if (t == kSetZero)
            this->SetZero();
        this->dim_ = dim;
        return;
    }

    if (dim > this->capacity_ && this->capacity_ != 0)
        this->Destroy();

    if (dim == 0)
        return;

    void*  data;
    size_t bytes = dim * sizeof(T);
    // Round up to multiple of 256
    bytes = (bytes + 255) & ~((size_t)255);
    CHECK_EQ(cudaMalloc(&data, bytes), cudaSuccess);
    this->data_     = static_cast<T*>(data);
    this->dim_      = dim;
    this->capacity_ = dim;
    if (t == kSetZero)
        this->SetZero();
}

template <typename T>
void CuVector<T>::Destroy()
{
    if (this->data_ != NULL)
        CHECK_EQ(cudaFree(this->data_), cudaSuccess);

    this->data_     = NULL;
    this->dim_      = 0;
    this->capacity_ = 0;
}

// Explicit instantiation
template class CuVector<float>;
template class CuVector<float*>;
template class CuVector<int32_t>;
template class CuVector<uint8_t>;
