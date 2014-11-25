/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <iostream>
#include <stdint.h>
#include <Eigen/Dense>

// CUDA runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <root_includes.hpp>

//#include <convolutionSeparable_common.h>
#include <convolutionSeparable_common_small.h>

#include <cuda_pc_helpers.h>

using namespace Eigen;
using std::cout;
using std::endl;

template<typename T>
class NormalExtractGpu
{
public: 
  NormalExtractGpu(T f_depth);
  NormalExtractGpu(T f_depth, uint32_t w, uint32_t h);
  ~NormalExtractGpu();

  void compute(const uint16_t* data, uint32_t w, uint32_t h);
  void computeAreaWeights();

  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr normals();
//  pcl::PointCloud<pcl::PointXYZRGB> normals();
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr pointCloud();
//  pcl::PointCloud<pcl::PointXYZ> pointCloud();
  T* d_normals(){return d_n;};
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
    bool cudaReady_, nCached_, pcCached_;
    T* d_x, *d_y, *d_z;
    T* d_xu, *d_yu, *d_zu;
    T* d_xv, *d_yv, *d_zv;
    T *d_n, *d_xyz;
    T *a,*b,*c; // for intermediate computations
    T *d_w;

    T *h_n;
    T *h_xyz;
    T *h_dbg;

    uint16_t* d_depth;
    T h_sobel_dif[KERNEL_LENGTH_S];
    T h_sobel_sum[KERNEL_LENGTH_S];
    T h_kernel_avg[KERNEL_LENGTH];

    pcl::PointCloud<pcl::PointXYZRGB> n_;
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr n_cp_;

    pcl::PointCloud<pcl::PointXYZ> pc_;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr pc_cp_;

};

// ------------------------------- impl ----------------------------------
