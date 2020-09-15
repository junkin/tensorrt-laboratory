#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include "assert.h"
#include "cuda_matrix.h"

#include <glog/logging.h>

template <typename Real>
CuMatrix<Real>::CuMatrix(MatrixIndexT rows, MatrixIndexT cols, MatrixResizeType resize_type, MatrixStrideType stride_type)
: num_rows_(0), num_cols_(0), stride_(0), data_(NULL)
{
    Resize(rows, cols, resize_type, stride_type);
}

template <typename Real>
void CuMatrix<Real>::Resize(MatrixIndexT rows, MatrixIndexT cols, MatrixResizeType resize_type, MatrixStrideType stride_type)
{
    // This code does not currently support the other resize_type options.
    assert(resize_type == kSetZero || resize_type == kUndefined);

    if (rows * cols == 0)
        assert(rows == 0 && cols == 0);

    MatrixIndexT new_stride;
    MatrixIndexT row_bytes = cols * sizeof(Real);
    if (stride_type == kDefaultStride)
    {
        row_bytes  = (row_bytes + 255) & ~((size_t)255);
        new_stride = row_bytes / sizeof(Real);
    }
    else
    {
        new_stride = cols;
    }

    if (rows * new_stride <= this->capacity_)
    {
        if (resize_type == kSetZero)
            this->SetZero();

        // Update rows, cols and stride
        this->num_rows_ = rows;
        this->num_cols_ = cols;
        this->stride_   = new_stride;

        return;
    }

    if (rows * new_stride > this->capacity_ && this->capacity_ != 0)
        this->Destroy();

    if (rows == 0)
        return;

    if (stride_type == kDefaultStride)
    {
        void* data;
        // Round up row bytes to multiple of 256
        row_bytes = (row_bytes + 255) & ~((size_t)255);
        CHECK_EQ(cudaMalloc(&data, row_bytes * rows), cudaSuccess);
        this->data_   = static_cast<Real*>(data);
        this->stride_ = row_bytes / sizeof(Real);
    }
    else
    { // kStrideEqualNumCols
        size_t bytes = rows * cols * sizeof(Real);
        bytes        = (bytes + 255) & ~((size_t)255);
        void* data;
        CHECK_EQ(cudaMalloc(&data, bytes), cudaSuccess);
        this->data_   = static_cast<Real*>(data);
        this->stride_ = cols;
    }

    this->num_rows_ = rows;
    this->num_cols_ = cols;
    this->capacity_ = this->num_rows_ * this->stride_;

    if (resize_type == kSetZero)
        this->SetZero();
}

template <typename Real>
void CuMatrix<Real>::SetZero()
{
    CHECK_EQ(cudaMemset2DAsync(data_, stride_ * sizeof(Real), 0, num_cols_ * sizeof(Real), num_rows_, stream_), cudaSuccess);
}

template <typename Real>
void CuMatrix<Real>::CopyFromMat(std::vector<std::vector<Real>>& mat, int num_rows)
{
    assert(mat.size() > num_rows);
    MatrixIndexT num_cols = mat[0].size();

    this->Resize(num_rows, num_cols);

    for (int row = 0; row < num_rows; row++)
    {
        CHECK_EQ(cudaMemcpyAsync(this->data_ + row * this->stride_ * sizeof(Real), &mat[row][0], num_cols * sizeof(Real),
                                 cudaMemcpyHostToDevice, stream_),
                 cudaSuccess);
    }
}

template <typename Real>
void CuMatrix<Real>::CopyFromMat(const CuMatrix<Real>& M)
{
    assert(M.NumRows() == num_rows_ && M.NumCols() == num_cols_);

    MatrixIndexT dst_pitch = stride_ * sizeof(Real);
    MatrixIndexT src_pitch = M.Stride() * sizeof(Real);
    MatrixIndexT width     = M.NumCols() * sizeof(Real);
    CHECK_EQ(cudaMemcpy2DAsync(data_, dst_pitch, M.data_, src_pitch, width, M.num_rows_, cudaMemcpyDeviceToDevice, stream_), cudaSuccess);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

template <typename Real>
void CuMatrix<Real>::CopyFromMat(const CuMatrix<Real>& M, int num_rows)
{
    assert(M.NumCols() == num_cols_);

    MatrixIndexT dst_pitch = stride_ * sizeof(Real);
    MatrixIndexT src_pitch = M.Stride() * sizeof(Real);
    MatrixIndexT width     = M.NumCols() * sizeof(Real);
    CHECK_EQ(cudaMemcpy2DAsync(data_, dst_pitch, M.data_, src_pitch, width, num_rows, cudaMemcpyDeviceToDevice, stream_), cudaSuccess);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

template <typename Real>
void CuMatrix<Real>::AddMatMat(Real alpha, const CuMatrix<Real>& A, MatrixTransposeType transA, const CuMatrix<Real>& B,
                               MatrixTransposeType transB, Real beta, cublasHandle_t& cublas_handle)
{
    // CUBLAS is col-major, cudamatrix is row-major, how to do the mapping?
    // keep trans..., just swap A&B matrices: A->B B->A
    MatrixIndexT m  = ((transB == kTrans) ? B.NumRows() : B.NumCols());
    MatrixIndexT n  = ((transA == kTrans) ? A.NumCols() : A.NumRows());
    MatrixIndexT k  = ((transB == kTrans) ? B.NumCols() : B.NumRows());
    MatrixIndexT k1 = ((transA == kTrans) ? A.NumRows() : A.NumCols());

    assert(m == NumCols());
    assert(n == NumRows());
    assert(k == k1);

    if (m == 0)
        return;

    CHECK_EQ(cublas_gemm(cublas_handle, (transB == kTrans ? CUBLAS_OP_T : CUBLAS_OP_N), (transA == kTrans ? CUBLAS_OP_T : CUBLAS_OP_N), m,
                         n, k, alpha, B.data_, B.Stride(), A.data_, A.Stride(), beta, data_, Stride()),
             cudaSuccess);
}

template <>
void CuMatrix<uint8_t>::AddMatMat(uint8_t alpha, const CuMatrix<uint8_t>& A, MatrixTransposeType transA, const CuMatrix<uint8_t>& B,
                                  MatrixTransposeType transB, uint8_t beta, cublasHandle_t& cublas_handle)
{
    assert(0);
}

template <typename Real>
void CuMatrix<Real>::Print(std::string& filename)
{
    std::ofstream file;

    std::vector<Real> host(num_rows_ * stride_);
    CHECK_EQ(cudaMemcpy(&host[0], data_, num_rows_ * stride_ * sizeof(Real), cudaMemcpyDeviceToHost), cudaSuccess);

    file.open(filename);
    for (int row = 0; row < num_rows_; row++)
    {
        for (int col = 0; col < num_cols_; col++)
        {
            file << host[row * stride_ + col];
            if (col != num_cols_ - 1)
            {
                file << ", ";
            }
        }
        file << std::endl;
    }
    file.close();
}

template <typename Real>
void CuMatrix<Real>::Destroy()
{
    if (this->data_ != NULL)
    {
        CHECK_EQ(cudaFree(this->data_), cudaSuccess);
    }

    this->data_     = NULL;
    this->num_rows_ = 0;
    this->num_cols_ = 0;
    this->stride_   = 0;
    this->capacity_ = 0;
}

// Explicit instantiation
template class CuMatrix<float>;
template class CuMatrix<uint8_t>;
