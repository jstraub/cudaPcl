/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <iostream>
#include <stdint.h>
#include <algorithm>
#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include <jsCore/gpuMatrix.hpp>

#include <cudaPcl/root_includes.hpp>
#include <cudaPcl/convolutionSeparable_common_small.h>
#include <cudaPcl/cuda_pc_helpers.h>

using namespace Eigen;
using std::cout;
using std::endl;

void depth2xyzGPU(uint16_t* d, float* x, float* y, float* z,
    float invF, int w, int h, float *xyz=NULL);
void depth2xyzGPU(uint16_t* d, double* x, double* y, double* z,
    double invF, int w, int h, double *xyz=NULL);

void depth2xyzFloatGPU(float* d, float* x, float* y, float* z,
    float invF, int w, int h, float *xyz=NULL);
void depth2xyzFloatGPU(double* d, double* x, double* y, double* z,
    double invF, int w, int h, double *xyz=NULL);

void copyShuffleGPU(float* in, float* out, uint32_t* ind, int32_t N, int32_t step);

void haveDataGpu(float* d_x, uint8_t* d_haveData, int32_t N, uint32_t step);
void haveDataGpu(double* d_x, uint8_t* d_haveData, int32_t N, uint32_t step);

void derivatives2normalsGPU(float* d_x, float* d_y, float* d_z,
    float* d_xu, float* d_yu, float* d_zu,
    float* d_xv, float* d_yv, float* d_zv,
    float* d_n, uint8_t* d_haveData, int w, int h);

void derivatives2normalsGPU(double* d_x, double* d_y, double* d_z,
    double* d_xu, double* d_yu, double* d_zu,
    double* d_xv, double* d_yv, double* d_zv,
    double* d_n, uint8_t* d_haveData, int w, int h);

void derivatives2normalsGPU(float* d_z, float* d_zu, float* d_zv, float* d_n,
    uint8_t* d_haveData, float invF, int w, int h);
void derivatives2normalsGPU(double* d_z, double* d_zu, double* d_zv, double*
    d_n, uint8_t* d_haveData, double invF, int w, int h);

void xyzImg2PointCloudXYZRGB(double* d_xyzImg, float* d_pclXYZRGB, int32_t w,
    int32_t h);

void xyzImg2PointCloudXYZRGB(float* d_xyzImg, float* d_pclXYZRGB, int32_t w,
    int32_t h);

namespace cudaPcl {

struct CfgSmoothNormals
{
  CfgSmoothNormals() : f_d(540.f), eps(0.2f*0.2f), B(9), compress(true){};
  float f_d;
  float eps;
  uint32_t B;
  bool compress;
};

template<typename T>
class NormalExtractSimpleGpu
{
public:
  NormalExtractSimpleGpu(T f_depth, uint32_t w, uint32_t h, bool compress = false);
  ~NormalExtractSimpleGpu();

  void compute(const uint16_t* data, uint32_t w, uint32_t h);
  void compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr & pc, T radius=0.03);

  void computeGpu(T* d_depth, uint32_t w, uint32_t h);
  void computeAreaWeights();
  void compressNormals( uint32_t w, uint32_t h);
//  void compressNormalsCpu(float* n, uint32_t w, uint32_t h);

  void setNormalsCpu(T* n, uint32_t w, uint32_t h);
  void setNormalsCpu(const pcl::PointCloud<pcl::Normal>& normals);

  // compressed normals
  cv::Mat normalsComp(int32_t& nComp);
  float* d_normalsComp(int32_t& nComp){nComp = nComp_; return d_normalsComp_.data();};

  void uncompressCpu(const uint32_t* in, uint32_t Nin,  uint32_t* out, uint32_t Nout);

  cv::Mat compInd();
  uint32_t* d_compInd(){return d_compInd_.data();};

  // normals as a 3 channel float image
  cv::Mat normalsImg();
  T* d_normalsImg(){return d_nImg_.data();}; // get pointer to memory with normal image

  // uint8_t where we have data
  cv::Mat haveData();
  uint8_t* d_haveData(){return d_haveData_.data();}; // get pointer to memory with normal image

  // normals as a point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr normalsPc();
  float* d_normalsPcl();
//  pcl::PointCloud<pcl::PointXYZRGB> normals();
//  pcl::PointCloud<pcl::PointXYZ>::ConstPtr pointCloud();
//  pcl::PointCloud<pcl::PointXYZ> pointCloud();

  T* d_weights(){return d_w;};
  uint32_t d_step(){return X_STEP;};
  uint32_t d_offset(){return X_OFFSET;};

protected:

    // sets up the memory on the GPU device
    void prepareCUDA(uint32_t w,uint32_t h);
    // computes derivatives of d_x, d_y, d_z on GPU
    void computeDerivatives(uint32_t w,uint32_t h);
    // smoothes the derivatives (iterations) times
    void smoothDerivatives(uint32_t iterations, uint32_t w,uint32_t h);
    // convert the d_depth into smoothed xyz
    void depth2smoothXYZ(T invF, uint32_t w,uint32_t h);

    T invF_;
    uint32_t w_,h_;
    bool cudaReady_, nCachedPc_, nCachedImg_, pcComputed_, nCachedComp_;
    bool compress_;
    T* d_x, *d_y, *d_z;
    T* d_xu, *d_yu, *d_zu;
    T* d_xv, *d_yv, *d_zv;
    jsc::GpuMatrix<T> d_nImg_;    // normal image - simple 3channel image
    float *d_nPcl; // using pcl conventions as if it were a PointCloud<PointXYZRGB>
    T *d_xyz;
    T *a,*b,*c; // for intermediate computations
    T *d_w;

    jsc::GpuMatrix<uint8_t> d_haveData_;

    int32_t nComp_;
    jsc::GpuMatrix<T> d_normalsComp_;
    jsc::GpuMatrix<uint32_t> d_compInd_;
    cv::Mat normalsComp_;
    cv::Mat compInd_;
    std::vector<uint32_t> indMap_;
    cv::Mat haveData_;

    T *h_nPcl;
    T *h_xyz;
    T *h_dbg;

    uint16_t* d_depth;
    T h_sobel_dif[KERNEL_LENGTH];
    T h_sobel_sum[KERNEL_LENGTH];

    pcl::PointCloud<pcl::PointXYZRGB> n_;
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr n_cp_;

    pcl::PointCloud<pcl::PointXYZ> pc_;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr pc_cp_;

    cv::Mat nImg_;
};

// ------------------------------- impl ----------------------------------
template<typename T>
NormalExtractSimpleGpu<T>::NormalExtractSimpleGpu(T f_depth, uint32_t w, uint32_t h, bool compress)
  : invF_(1./f_depth), w_(w), h_(h), cudaReady_(false), nCachedPc_(false),
   nCachedImg_(false), pcComputed_(false), nCachedComp_(false), compress_(compress),
   d_nImg_(h*w,3),
   d_haveData_(h,w), d_normalsComp_(h*w,3), d_compInd_(w*h,1), h_dbg(NULL)
{
  cout<<"calling prepareCUDA"<<endl;
  cout<<cudaReady_<<endl;
  prepareCUDA(w_,h_);
  cout<<"prepareCUDA done"<<endl;
};

template<typename T>
NormalExtractSimpleGpu<T>::~NormalExtractSimpleGpu()
{
  if(!cudaReady_) return;
  checkCudaErrors(cudaFree(d_depth));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_z));
  checkCudaErrors(cudaFree(d_nPcl));
  checkCudaErrors(cudaFree(d_xyz));
  checkCudaErrors(cudaFree(d_xu));
  checkCudaErrors(cudaFree(d_yu));
  checkCudaErrors(cudaFree(d_zu));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
  checkCudaErrors(cudaFree(d_zv));
  checkCudaErrors(cudaFree(a));
  checkCudaErrors(cudaFree(b));
  checkCudaErrors(cudaFree(c));
//#ifdef WEIGHTED
  if(d_w != NULL) checkCudaErrors(cudaFree(d_w));
//#endif
//  free(h_nPcl);
  if(h_dbg != NULL) free(h_dbg);
};


template<typename T>
void NormalExtractSimpleGpu<T>::compute(const uint16_t* data, uint32_t w, uint32_t h)
{
  assert(w_ == w);
  assert(h_ == h);

  checkCudaErrors(cudaMemcpy(d_depth, data, w_ * h_ * sizeof(uint16_t),
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());
  computeGpu(d_depth,w,h);

//  depth2xyzGPU(d_depth, d_x, d_y, d_z,invF_, w_, h_, NULL);
//  // obtain derivatives using sobel
//  computeDerivatives(w_,h_);
//  // obtain the normals using mainly cross product on the derivatives
////  derivatives2normalsGPU(
////      d_x,d_y,d_z,
////      d_xu,d_yu,d_zu,
////      d_xv,d_yv,d_zv,
////      d_nImg_.data(),d_haveData_.data(),w_,h_);
//  derivatives2normalsGPU( d_z, d_zu, d_zv,
//      d_nImg_.data(),d_haveData_.data(),invF_,w_,h_);
//  nCachedPc_ = false;
//  nCachedImg_ = false;
//  pcComputed_ = false;
//  nCachedComp_ = false;
};

template<typename T>
void NormalExtractSimpleGpu<T>::computeGpu(T* depth, uint32_t w, uint32_t h)
{
  assert(w_ == w);
  assert(h_ == h);

  if(true)
  {
    // convert depth image to xyz point cloud
    depth2xyzFloatGPU(depth, d_x, d_y, d_z, invF_, w_, h_, NULL);
//    haveDataGpu(depth,d_haveData_.data(),w*h,1);
    // obtain derivatives using sobel
    computeDerivatives(w_,h_);
    // obtain the normals using mainly cross product on the derivatives
    derivatives2normalsGPU(
        d_x,d_y,d_z,
        d_xu,d_yu,d_zu,
        d_xv,d_yv,d_zv,
        d_nImg_.data(),d_haveData_.data(),w_,h_);

  }else{
    // potentially faster but sth is wrong with the derivatives2normalsGPU
    // I think this approach is numerically instable since I cannot
    // renormalize in between
    setConvolutionKernel(h_sobel_dif);
    convolutionRowsGPU(c,depth,w,h);
    setConvolutionKernel(h_sobel_sum);
    convolutionColumnsGPU(d_zu,c,w,h);
    convolutionRowsGPU(b,depth,w,h);
    setConvolutionKernel(h_sobel_dif);
    convolutionColumnsGPU(d_zv,b,w,h);

    derivatives2normalsGPU(depth, d_zu, d_zv,
        d_nImg_.data(),d_haveData_.data(),invF_,w_,h_);

  }
  haveDataGpu(d_nImg_.data(),d_haveData_.data(),w*h,3);

  nCachedPc_ = false;
  nCachedImg_ = false;
  pcComputed_ = false;
  nCachedComp_ = false;
  if(compress_)
  {
    compressNormals(w,h);
  }
};

template<typename T>
void NormalExtractSimpleGpu<T>::compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, T radius)
{
  // extract surface normals
  pcl::PointCloud<pcl::Normal> normals;
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(pc);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);
  ne.setRadiusSearch (radius);
  ne.compute (normals);
  setNormalsCpu(normals);
};

template<typename T>
void NormalExtractSimpleGpu<T>::setNormalsCpu(const pcl::PointCloud<pcl::Normal>& normals)
{
  assert(normals.width==w_);
  assert(normals.height==h_);
  // use pitchs to copy correctly: pcl:Normals is float[4] where the
  // first 3 are normals and the 4th is the curvature
  d_nImg_.set(normals.getMatrixXfMap().data(),normals.height,normals.width,3,4);
  haveDataGpu(d_nImg_.data(),d_haveData_.data(),normals.height*normals.width ,3);

  nCachedPc_ = false;
  nCachedImg_ = false;
  pcComputed_ = false;
  nCachedComp_ = false;
  if(compress_)
  {
    compressNormals(normals.width,normals.height);
  }
};

template<typename T>
void NormalExtractSimpleGpu<T>::setNormalsCpu(T* n, uint32_t w, uint32_t h)
{
  assert(w==w_);
  assert(h==h_);
  d_nImg_.set(n,w*h,3);
  haveDataGpu(d_nImg_.data(),d_haveData_.data(),w*h,3);

  nCachedPc_ = false;
  nCachedImg_ = false;
  pcComputed_ = false;
  nCachedComp_ = false;
  if(compress_)
  {
    compressNormals(w,h);
  }
};

template<typename T>
void NormalExtractSimpleGpu<T>::compressNormals(uint32_t w, uint32_t h)
{
    cv::Mat haveDat = haveData();
    indMap_.clear();
    indMap_.reserve(w*h);
    for(uint32_t i=0; i<w*h; ++i) 
      if(haveDat.at<uint8_t>(i) ==1)
        indMap_.push_back(i);
    nComp_ = indMap_.size();
    if (nComp_ > 0)
    {
      d_normalsComp_.resize(nComp_,3);
      // just shuffle the first 640 entries -> definitely sufficient to ge random init for the algfoerithms
      std::random_shuffle(indMap_.begin(), indMap_.begin() + std::min(640,nComp_));
      jsc::GpuMatrix<uint32_t> d_indMap_(indMap_); // copy to GPU
      copyShuffleGPU(d_nImg_.data(), d_normalsComp_.data(), d_indMap_.data(), nComp_, 3);
    }
#ifndef NDEBUG
    cout<<"compression of "<<T(w*h-nComp_)/T(w*h)<<"% to "
      <<nComp_<<" datapoints"<<endl;
#endif
};



template<typename T>
void NormalExtractSimpleGpu<T>::uncompressCpu(const uint32_t* in,
    uint32_t Nin, uint32_t* out, uint32_t Nout)
{
  if(indMap_.size() > 0)
  {
    for(uint32_t i=0; i<Nout; ++i) out[i] = INT_MAX;
    for(uint32_t i=0; i<Nin; ++i) out[indMap_[i]] = in[i];
  }
};

template<typename T>
void NormalExtractSimpleGpu<T>::computeDerivatives(uint32_t w,uint32_t h)
{
//  // TODO could reorder stuff here
  setConvolutionKernel(h_sobel_dif);
  convolutionRowsGPU(a,d_x,w,h);
  convolutionRowsGPU(b,d_y,w,h);
  convolutionRowsGPU(c,d_z,w,h);
  setConvolutionKernel(h_sobel_sum);
  convolutionColumnsGPU(d_xu,a,w,h);
  convolutionColumnsGPU(d_yu,b,w,h);
  convolutionColumnsGPU(d_zu,c,w,h);
  convolutionRowsGPU(a,d_x,w,h);
  convolutionRowsGPU(b,d_y,w,h);
  convolutionRowsGPU(c,d_z,w,h);
  setConvolutionKernel(h_sobel_dif);
  convolutionColumnsGPU(d_xv,a,w,h);
  convolutionColumnsGPU(d_yv,b,w,h);
  convolutionColumnsGPU(d_zv,c,w,h);
}


template<typename T>
void NormalExtractSimpleGpu<T>::computeAreaWeights()
{
  if(d_w == NULL)
    checkCudaErrors(cudaMalloc((void **)&d_w, w_ * h_ * sizeof(T)));
//#ifdef WEIGHTED
//  weightsFromCovGPU(d_z, d_w, 30.0f, invF_, w_,h_);
  weightsFromAreaGPU(d_z, d_w, w_,h_);
//#endif
}


template<typename T>
void NormalExtractSimpleGpu<T>::prepareCUDA(uint32_t w,uint32_t h)
{
  // CUDA preparations
  cout << "Allocating and initializing CUDA arrays..."<<endl;
  checkCudaErrors(cudaMalloc((void **)&d_depth, w * h * sizeof(uint16_t)));
  checkCudaErrors(cudaMalloc((void **)&d_x, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_y, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_z, w * h * sizeof(T)));

  checkCudaErrors(cudaMalloc((void **)&d_nPcl, w * h *8* sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_xyz, w * h * 4* sizeof(T)));

  checkCudaErrors(cudaMalloc((void **)&d_xu, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_yu, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_zu, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_xv, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_yv, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_zv, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&a, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&b, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&c, w * h * sizeof(T)));

   d_haveData_.setZero();
//#ifdef WEIGHTED
//#else
  d_w = NULL;
//#endif

  h_sobel_dif[0] = 1;
  h_sobel_dif[1] = 0;
  h_sobel_dif[2] = -1;

  h_sobel_sum[0] = 1;
  h_sobel_sum[1] = 2;
  h_sobel_sum[2] = 1;

  n_ = pcl::PointCloud<pcl::PointXYZRGB>(w,h);
  n_cp_ = pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr(&n_);
  Map<MatrixXf, Aligned, OuterStride<> > nMat =
    n_.getMatrixXfMap(8,8,0);
  h_nPcl = nMat.data();//(T *)malloc(w *h *3* sizeof(T));

  cout<<nMat.rows()<< " "<<nMat.cols()<<" "<<8<<endl;
  cout<<w<<" "<<h<<endl;

  pc_ = pcl::PointCloud<pcl::PointXYZ>(w,h);
  pc_cp_ = pcl::PointCloud<pcl::PointXYZ>::ConstPtr(&pc_);
  Map<MatrixXf, Aligned, OuterStride<> > pcMat =
    pc_.getMatrixXfMap(4,4,0);
  h_xyz = pcMat.data();//(T *)malloc(w *h *3* sizeof(T));

  h_dbg = (T *)malloc(w *h * sizeof(T));

  cudaReady_ = true;
}


template<typename T>
float* NormalExtractSimpleGpu<T>::d_normalsPcl(){
  if(!pcComputed_ && w_ > 0 && h_ > 0)
  {
    xyzImg2PointCloudXYZRGB(d_nImg_.data(), d_nPcl,w_,h_);
    pcComputed_ = true;
  }
  return d_nPcl;
}; // get pointer to memory with pcl point cloud PointCloud<PointXYZRGB>

template<typename T>
pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr NormalExtractSimpleGpu<T>::normalsPc()
{
  //TODO bad error here
  if(!nCachedPc_)
  {
    if(!pcComputed_ && w_ > 0 && h_ > 0)
    {
      xyzImg2PointCloudXYZRGB(d_nImg_.data(), d_nPcl,w_,h_);
      pcComputed_ = true;
    }
//    cout<<"w="<<w_<<" h="<<h_<<" "<<X_STEP<<" "<<sizeof(T)<<endl;
    checkCudaErrors(cudaMemcpy(h_nPcl, d_nPcl, w_*h_ *sizeof(float)* 8,
          cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    nCachedPc_ = true;
  }
  return n_cp_;
};


template<typename T>
cv::Mat NormalExtractSimpleGpu<T>::haveData()
{
//  if(!nCachedImg_)
//  {

//    cv::Mat haveData = cv::Mat::ones(h_,w_,CV_8U)*2; // not necessary
    haveData_ = cv::Mat (h_,w_,CV_8U);
    checkCudaErrors(cudaMemcpy(haveData_.data, d_haveData_.data(),
          w_*h_ *sizeof(uint8_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
//    nCachedImg_ = true;
//  }
  return haveData_;
};

template<typename T>
cv::Mat NormalExtractSimpleGpu<T>::compInd()
{
//  if(!nCachedImg_)
//  {
    compInd_ = cv::Mat (h_,w_,CV_32S); // there is no CV_32U - but an just type cast between those
    d_compInd_.get((uint32_t*)compInd.data,nComp_,1);
//    nCachedImg_ = true;
//  }
  return compInd_;
};

template<>
cv::Mat NormalExtractSimpleGpu<float>::normalsImg()
{
  if(!nCachedImg_)
  {
//    cout<<"NormalExtractSimpleGpu::normalsImg size: "<<w_<<" "<<h_<<endl;
    nImg_ = cv::Mat(h_,w_,CV_32FC3);
    checkCudaErrors(cudaMemcpy(nImg_.data, d_nImg_.data(), w_*h_
          *sizeof(float)*3, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    nCachedImg_ = true;
  }
  return nImg_;
};

template<>
cv::Mat NormalExtractSimpleGpu<double>::normalsImg()
{
  if(!nCachedImg_)
  {
    cout<<"NormalExtractSimpleGpu::normalsImg size: "<<w_<<" "<<h_<<endl;
    nImg_ = cv::Mat(h_,w_,CV_64FC3);
    checkCudaErrors(cudaMemcpy(nImg_.data, d_nImg_.data(), w_*h_
          *sizeof(double)*3, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    nCachedImg_ = true;
  }
  return nImg_;
};


template<>
cv::Mat NormalExtractSimpleGpu<float>::normalsComp(int32_t& nComp)
{
  if(!nCachedComp_)
  {
//    cout<<nComp_<<endl;
    normalsComp_ = cv::Mat(nComp_,1,CV_32FC3);
//    cout<<normalsComp_.total()<<" "
//      <<normalsComp_.total()*normalsComp_.elemSize()<<" "
//      <<normalsComp_.elemSize()<<endl;
    d_normalsComp_.get((float*)normalsComp_.data,nComp_,3);
    nCachedComp_ = true;
  }
  nComp = nComp_;
  return normalsComp_;
}

template<>
cv::Mat NormalExtractSimpleGpu<double>::normalsComp(int32_t& nComp)
{
  if(!nCachedComp_)
  {
    normalsComp_ = cv::Mat(nComp_,1,CV_64FC3);
    d_normalsComp_.get((double*)normalsComp_.data,nComp_,3);
    nCachedComp_ = true;
  }
  nComp = nComp_;
  return normalsComp_;
}

//template<typename T>
//pcl::PointCloud<pcl::PointXYZRGB> NormalExtractGpu<T>::normals()
//{
//  if(!nCachedPc_)
//  {
//    checkCudaErrors(cudaMemcpy(h_nPcl, d_nImg_, w_*h_* X_STEP *sizeof(T),
//          cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaDeviceSynchronize());
//    nCachedPc_ = true;
//  }
//  return n_;
//};
//template<typename T>
//pcl::PointCloud<pcl::PointXYZ>::ConstPtr NormalExtractSimpleGpu<T>::pointCloud()
//{
////  if(!pcCached_)
////  {
////    checkCudaErrors(cudaMemcpy(h_xyz, d_xyz, w_*h_*4 *sizeof(T),
////          cudaMemcpyDeviceToHost));
////    checkCudaErrors(cudaDeviceSynchronize());
////    pcCached_ = true;
////  }
//  return pc_cp_;
//};
}
