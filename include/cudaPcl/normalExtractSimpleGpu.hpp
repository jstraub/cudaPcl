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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// CUDA runtime
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include <jsCore/gpuMatrix.hpp>

#include <cudaPcl/root_includes.hpp>
#include <cudaPcl/convolutionSeparable_common_small.h>
#include <cudaPcl/cuda_pc_helpers.h>
#include <cudaPcl/openniVisualizer.hpp>

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

  void computeGpu(uint16_t* d_depth);
  void computeGpu(T* d_depth);
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
    void prepareCUDA();
    // computes derivatives of d_x, d_y, d_z on GPU
    void computeDerivatives(uint32_t w,uint32_t h);
    // smoothes the derivatives (iterations) times
    void smoothDerivatives(uint32_t iterations, uint32_t w,uint32_t h);
    // convert the d_depth into smoothed xyz
    void depth2smoothXYZ(T invF, uint32_t w,uint32_t h);

    T invF_;
    // Width and height of in and output
    uint32_t w_,h_;
    // Width and height for internal processing pruposes
    uint32_t wProc_,hProc_;
    bool cudaReady_, nCachedPc_, nCachedImg_, pcComputed_, nCachedComp_;
    bool compress_;
    jsc::GpuMatrix<T> d_z;
    jsc::GpuMatrix<T> d_zu;
    T* d_x, *d_y;
    T* d_xu, *d_yu;
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

    uint16_t* d_depth;
    T h_sobel_dif[KERNEL_LENGTH];
    T h_sobel_sum[KERNEL_LENGTH];

    pcl::PointCloud<pcl::PointXYZRGB> n_;
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr n_cp_;
//    pcl::PointCloud<pcl::PointXYZ> pc_;
//    pcl::PointCloud<pcl::PointXYZ>::ConstPtr pc_cp_;

    cv::Mat nImg_;
};

template <class T>
static void DoNotFree(T*)
{ }
}
#include <cudaPcl/normalExtractSimpleGpu_impl.hpp>
