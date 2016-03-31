/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once
#include <iostream>
#include <stdint.h>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <jsCore/gpuMatrix.hpp>
#include <jsCore/timer.hpp>
#include <jsCore/timerLog.hpp>

using std::cout;
using std::endl;

void depth2floatGPU(uint16_t* d, double* d_float,uint8_t* haveData, int w, int
    h, int outStep);
void depth2floatGPU(uint16_t* d, float* d_float,uint8_t* haveData, int w, int
    h, int outStep);
void integralGpu(double* I, double* Isum,double* IsumT, double* IsumCarry,
    uint32_t w, uint32_t h);
void guidedFilter_ab_gpu(uint8_t* haveData, uint8_t* haveDataAfter, double* a,
    double* b, int32_t* Ns, double* dSum, double* dSqSum, double eps, uint32_t
    B, uint32_t w, uint32_t h);
void guidedFilter_out_gpu(uint8_t* haveData, double* depth, double* aInt,
    double* bInt, int32_t* Ns, float* depthSmooth,  uint32_t B,uint32_t w,
    uint32_t h);
void guidedFilter_out_gpu(uint8_t* haveData, double* depth, double* aInt,
    double* bInt, int32_t* Ns, double* depthSmooth,  uint32_t B,uint32_t w,
    uint32_t h);


void guidedFilter_ab_gpu(double* depth, uint8_t* haveData, uint8_t*
    haveDataAfter, double* a, double* b, int32_t* Ns, double* dSum, double*
    dSqSum, double eps, uint32_t B, uint32_t w, uint32_t h);

namespace cudaPcl {

/*
 * template typespecifies the ouput data-type of the smoothed depth image
 * internally everything is double because of the integral images
 */
template<typename T>
class DepthGuidedFilterGpu
{
 public:
  DepthGuidedFilterGpu(uint32_t w, uint32_t h, double eps, uint32_t B);

  virtual ~DepthGuidedFilterGpu();

  virtual void filter(const cv::Mat& depth);

  cv::Mat getOutput(); 

  T* getDepthDevicePtr(){ return d_depthSmooth.data();};
  uint8_t * d_haveData(){ return d_haveData2.data();};
  cv::Mat haveData() { return haveData2;};

  protected:
  uint32_t w_,h_;
  double eps_;
  uint32_t B_;

  jsc::GpuMatrix<uint16_t> d_depthU16;
  jsc::GpuMatrix<double> d_depth;
  jsc::GpuMatrix<double> d_dSum;
  jsc::GpuMatrix<double> d_dSumT;
  jsc::GpuMatrix<double> d_dSqSum;
  jsc::GpuMatrix<uint8_t> d_haveData_;
  jsc::GpuMatrix<uint8_t> d_haveData2;

  cudaStream_t stream1;
  cudaStream_t stream2;

  jsc::TimerLog tLog;

  cv::Mat aInt;
  cv::Mat bInt;
  cv::Mat Ns;
  cv::Mat dSum; 
  cv::Mat dSqSum;
  cv::Mat haveData_;
  cv::Mat haveData2;
  cv::Mat a;
  cv::Mat b;
  cv::Mat dSmooth;
  cv::Mat dFlt; 
  jsc::GpuMatrix<T> d_depthSmooth;
  jsc::GpuMatrix<double> d_a; 
  jsc::GpuMatrix<double> d_b; 
  jsc::GpuMatrix<int32_t> d_Ns;
  jsc::GpuMatrix<double> d_aInt; 
  jsc::GpuMatrix<double> d_bInt; 

};
}
#include <cudaPcl/depthGuidedFilter_impl.hpp>
