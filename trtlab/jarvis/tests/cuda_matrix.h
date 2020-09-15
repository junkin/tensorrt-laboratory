#ifndef CU_MATRIX_H_
#define CU_MATRIX_H_

#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>

#define MatrixIndexT uint32_t

typedef enum
{
    kSetZero,
    kUndefined,
    kCopyData
} MatrixResizeType;

typedef enum
{
    kDefaultStride,
    kStrideEqualNumCols,
} MatrixStrideType;

typedef enum
{
    kTrans   = 112, // = CblasTrans
    kNoTrans = 111  // = CblasNoTrans
} MatrixTransposeType;

inline cublasStatus_t
cublas_gemm(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, float alpha, const float* A, int lda, const float* B,
    int ldb, float beta, float* C, int ldc)
{
  return cublasSgemm_v2(
      handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline cublasStatus_t
cublas_gemm(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, double alpha, const double* A, int lda,
    const double* B, int ldb, double beta, double* C, int ldc)
{
  return cublasDgemm_v2(
      handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}


template <typename Real>
class CuMatrix
{
public:
    MatrixIndexT NumRows() const
    {
        return num_rows_;
    }
    MatrixIndexT NumCols() const
    {
        return num_cols_;
    }
    MatrixIndexT Stride() const
    {
        return stride_;
    }

    inline const Real* Data() const
    {
        return data_;
    }
    inline Real* Data()
    {
        return data_;
    }

    CuMatrix(cudaStream_t stream = cudaStreamPerThread) : data_(NULL), num_cols_(0), num_rows_(0), stride_(0), capacity_(0), stream_(stream) {}

    /// Constructor with memory initialisation
    CuMatrix(MatrixIndexT rows, MatrixIndexT cols, MatrixResizeType resize_type = kSetZero, MatrixStrideType stride_type = kDefaultStride);

    CuMatrix<Real>& operator=(const CuMatrix<Real>& other)
    {
        this->Resize(other.NumRows(), other.NumCols(), kUndefined);
        this->CopyFromMat(other);
        return *this;
    }

    void Resize(MatrixIndexT rows, MatrixIndexT cols, MatrixResizeType resize_type = kSetZero,
                MatrixStrideType stride_type = kDefaultStride);

    void SetZero();

    void AddMatMat(Real alpha, const CuMatrix<Real>& A, MatrixTransposeType transA, const CuMatrix<Real>& B, MatrixTransposeType transB,
                   Real beta, cublasHandle_t& cublas_handle);

    void CopyFromMat(std::vector<std::vector<Real>>& mat, int num_rows);

    void CopyFromMat(const CuMatrix<Real>& src);

    void CopyFromMat(const CuMatrix<Real>& src, int num_rows);

    void Print(std::string& filename);

    void Destroy();

    ~CuMatrix()
    {
        Destroy();
    }

private:
    Real*        data_; ///< GPU data pointer (or regular matrix data pointer,
    MatrixIndexT num_cols_;
    MatrixIndexT num_rows_;
    MatrixIndexT stride_;
    MatrixIndexT capacity_;
    cudaStream_t stream_;
};

#endif
