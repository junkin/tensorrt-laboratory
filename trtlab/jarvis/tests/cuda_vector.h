#ifndef CU_VECTOR_H_
#define CU_VECTOR_H_

#include "cuda_matrix.h"

template <typename T>
class CuVector
{
public:
    /// Dimensions
    MatrixIndexT Dim() const
    {
        return dim_;
    }

    inline T* Data()
    {
        return data_;
    }
    inline const T* Data() const
    {
        return data_;
    }

    CuVector(cudaStream_t stream = cudaStreamPerThread) : data_(NULL), dim_(0), capacity_(0), stream_(stream) {}

    /// Constructor with memory initialisation
    CuVector(MatrixIndexT size, MatrixResizeType resize_type = kSetZero);

    void SetZero();

    void Print(std::string& filename);

    // Function checks if entries from 0 to n-1 are different between
    // host new_vector and old_vector. If so, updates old_vector and cu_vector
    void CheckIfChangedAndCopy(std::vector<T>& new_vector, std::vector<T>& old_vector, int n);

    void Resize(MatrixIndexT dim, MatrixResizeType t = kSetZero);

    void CopyFromVec(const std::vector<T>& vec);

    ~CuVector()
    {
        Destroy();
    }

private:
    void Destroy();

    T*           data_;
    MatrixIndexT dim_;
    MatrixIndexT capacity_;
    cudaStream_t stream_;
};

#endif
