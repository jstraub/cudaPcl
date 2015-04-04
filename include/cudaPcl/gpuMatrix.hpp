/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <iostream>
#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <cudaPcl/helper_cuda.h> 

//#include "global.hpp"

using namespace Eigen;
//using boost::shared_ptr;
using std::cout;
using std::endl;

extern void copy_gpu( double *d_from, double *d_to , uint32_t N, 
    uint32_t step, uint32_t offset, uint32_t D);
extern void copy_gpu( float *d_from, float *d_to , uint32_t N, 
    uint32_t step, uint32_t offset, uint32_t D);
extern void copy_gpu( uint32_t *d_from, uint32_t *d_to , uint32_t N, 
    uint32_t step, uint32_t offset, uint32_t D);

namespace cudaPcl {

template <class T>
struct GpuMatrix
{

  GpuMatrix(uint32_t rows, uint32_t cols=1);
  GpuMatrix(const Matrix<T,Dynamic,Dynamic> & data);
  GpuMatrix(const Matrix<T,Dynamic,1> & data);
  GpuMatrix(const std::vector<T> & data);
  GpuMatrix(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& data);
  GpuMatrix(const boost::shared_ptr<Matrix<T,Dynamic,1> > & data);
  ~GpuMatrix();

  void set(T A);
  void set(const T* A, uint32_t rows, uint32_t cols, 
    uint32_t aPitch, uint32_t gpuPitch);
  void set(const T* A, uint32_t rows, uint32_t cols);
  void set(const std::vector<T>& A);
  void set(const Matrix<T,Dynamic,Dynamic>& A);
  void set(const Matrix<T,Dynamic,1>& A);
  void set(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& A);
  void set(const boost::shared_ptr<Matrix<T,Dynamic,1> >& A);
  void setZero();
  void setAsync(const T* A, uint32_t rows, uint32_t cols, cudaStream_t& stream);

  void get(T& a);
  void get(Matrix<T,Dynamic,Dynamic>& A);
  void get(T* A, uint32_t rows, uint32_t cols);
  void get(Matrix<T,Dynamic,1>& A);
//  void get(std::vector<T>& A);
  void get(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& A);
  void get(const boost::shared_ptr<Matrix<T,Dynamic,1> >& A);
  Matrix<T,Dynamic,Dynamic> get(void)
  {
    Matrix<T,Dynamic,Dynamic> d(rows_,cols_);
    this->get(d); return d;
  };
  void getAsync(T* A, uint32_t rows, uint32_t cols, cudaStream_t& stream);

  void copyFromGpu(T* d_A, uint32_t N, uint32_t step, uint32_t offset, uint32_t rows);

  void resize(uint32_t rows, uint32_t cols);

  uint32_t rows(){return rows_;};
  uint32_t cols(){return cols_;};
  T* data(){ assert(initialized_); return data_;};
  bool isInit(){return initialized_;};

  void print(){cout<<rows_<<";"<<cols_<<" init="<<(initialized_?'y':'n')<<endl;};

  static cudaStream_t createStream() 
  {
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    return stream;
  };
  static void deleteStream(cudaStream_t& stream) 
  {
    checkCudaErrors(cudaStreamDestroy(stream));
  };

private: 
  uint32_t rows_;
  uint32_t cols_;
  T * data_;
  bool initialized_;
};

// --------------------------------- impl -------------------------------------

template <class T>
GpuMatrix<T>::GpuMatrix(uint32_t rows, uint32_t cols)
  : rows_(rows), cols_(cols), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
};

template <class T>
GpuMatrix<T>::GpuMatrix(const Matrix<T,Dynamic,Dynamic> & data)
  : rows_(data.rows()), cols_(data.cols()), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  set(data);
};

template <class T>
GpuMatrix<T>::GpuMatrix(const Matrix<T,Dynamic,1> & data)
  : rows_(data.rows()), cols_(1), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  set(data);
};

template <class T>
GpuMatrix<T>::GpuMatrix(const std::vector<T> & data)
  : rows_(data.size()), cols_(1), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  set(data);
};

template <class T>
GpuMatrix<T>::GpuMatrix(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> > & data)
  : rows_(data->rows()), cols_(data->cols()), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  set(data);
};

template <class T>
GpuMatrix<T>::GpuMatrix(const boost::shared_ptr<Matrix<T,Dynamic,1> > & data)
  : rows_(data->rows()), cols_(1), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  set(data);
};

template <class T>
GpuMatrix<T>::~GpuMatrix()
{
  checkCudaErrors(cudaFree(data_));
};

template <class T>
void GpuMatrix<T>::resize(uint32_t rows, uint32_t cols)
{
  if((rows != rows_)||(cols != cols_))
  { 
    rows_ = rows;
    cols_ = cols;
    checkCudaErrors(cudaFree(data_));
    checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  } 
};

//setters 
template <class T>
  void GpuMatrix<T>::set(T A)
{
  resize(1,1);
  assert(1 == cols_);
  assert(1 == rows_);
  checkCudaErrors(cudaMemcpy(data_, &A, cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::set(const T* A, uint32_t rows, uint32_t cols,
    uint32_t aPitch, uint32_t gpuPitch)
{
  resize(rows,cols);
  assert(rows == rows_);
  assert(cols == cols_);
  checkCudaErrors(cudaMemcpy2D(data_, gpuPitch*sizeof(T), A,
        aPitch*sizeof(T), cols_* sizeof(T), rows_,
        cudaMemcpyHostToDevice));
  initialized_ = true;
}

template <class T>
  void GpuMatrix<T>::set(const T* A, uint32_t rows, uint32_t cols)
{
  resize(rows,cols);
  assert(rows == rows_);
  assert(cols == cols_);
  checkCudaErrors(cudaMemcpy(data_, A, cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
}
template <class T>
  void GpuMatrix<T>::set(const std::vector<T>& A)
{
  resize(A.size(),1);
  assert(A.size() == rows_);
  assert(1 == cols_);
  checkCudaErrors(cudaMemcpy(data_, A.data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
}

template <class T>
void GpuMatrix<T>::set(const Matrix<T,Dynamic,Dynamic>& A)
{
  resize(A.rows(),A.cols());
  assert(A.cols() == cols_);
  assert(A.rows() == rows_);
  checkCudaErrors(cudaMemcpy(data_, A.data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::set(const Matrix<T,Dynamic,1>& A)
{
  resize(A.rows(),A.cols());
  assert(A.cols() == cols_);
  assert(A.rows() == rows_);
  checkCudaErrors(cudaMemcpy(data_, A.data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::set(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& A)
{
  resize(A->rows(),A->cols());
  assert(A->cols() == cols_);
  assert(A->rows() == rows_);
  checkCudaErrors(cudaMemcpy(data_, A->data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::set(const boost::shared_ptr<Matrix<T,Dynamic,1> >& A)
{
  resize(A->rows(),A->cols());
  assert(A->cols() == cols_);
  assert(A->rows() == rows_);
  checkCudaErrors(cudaMemcpy(data_, A->data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::setZero()
{
  Matrix<T,Dynamic,Dynamic> A = Matrix<T,Dynamic,Dynamic>::Zero(rows_,cols_);
  checkCudaErrors(cudaMemcpy(data_, A.data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

 // getters
template <class T>
  void GpuMatrix<T>::get(T& a)
{
  assert(1 == cols_);
  assert(1 == rows_);
  checkCudaErrors(cudaMemcpy(&a,data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
}

template <class T>
void GpuMatrix<T>::get(T* A, uint32_t rows, uint32_t cols)
{
//  assert(cols == cols_);
//  assert(rows == rows_);
//  cout<<cols_*rows_<<" "<<cols_*rows_*sizeof(T)<<" "<<sizeof(T)<<endl;
  checkCudaErrors(cudaMemcpy(A,data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
};

template <class T>
void GpuMatrix<T>::get(Matrix<T,Dynamic,Dynamic>& A)
{
  A.resize(rows_,cols_);
  assert(A.cols() == cols_);
  assert(A.rows() == rows_);
  checkCudaErrors(cudaMemcpy(A.data(),data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
};

template <class T>
void GpuMatrix<T>::get(Matrix<T,Dynamic,1>& A)
{
  A.resize(rows_);
  assert(A.cols() == cols_);
  assert(A.rows() == rows_);
  checkCudaErrors(cudaMemcpy(A.data(),data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
};

template <class T>
void GpuMatrix<T>::get(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& A)
{
  A->resize(rows_,cols_);
  assert(A->cols() == cols_);
  assert(A->rows() == rows_);
  checkCudaErrors(cudaMemcpy(A->data(),data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
};

template <class T>
void GpuMatrix<T>::get(const boost::shared_ptr<Matrix<T,Dynamic,1> >& A)
{
  A->resize(rows_);
  assert(A->cols() == cols_);
  assert(A->rows() == rows_);
  checkCudaErrors(cudaMemcpy(A->data(),data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
};

template <class T>
void GpuMatrix<T>::copyFromGpu(T* d_A, uint32_t N, uint32_t step, uint32_t offset, uint32_t rows)
{
  // input d_A is N long and step per element
  // output (this class) is N long and rows per element
  resize(rows,N);
//  checkCudaErrors(cudaMemcpy(A->data(),data_, cols_*rows_*sizeof(T),            
//                        cudaMemcpyDeviceToHost));
  if(step != rows || offset!=0)
    copy_gpu(d_A, data_, N, step, offset, rows);
  else{
    // can just do simple mem copy
    checkCudaErrors(cudaMemcpy(data_, d_A, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToDevice));
  }

  initialized_ = true;
};


template <class T>
void GpuMatrix<T>::setAsync(const T* A, uint32_t rows, uint32_t cols, cudaStream_t& stream)
{
  resize(rows,cols);
  assert(rows == rows_);
  assert(cols == cols_);
  checkCudaErrors(cudaMemcpyAsync(data_, A, cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice,stream));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::getAsync(T* A, uint32_t rows, uint32_t cols, cudaStream_t& stream)
{
  assert(cols == cols_);
  assert(rows == rows_);
  checkCudaErrors(cudaMemcpyAsync(A,data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost,stream));
};


}
