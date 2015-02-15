/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <stdint.h>
#include <iostream>
#include <Eigen/Dense>

// CUDA runtime
#include <cuda_runtime.h>
// Utilities and system includes
#include <cudaPcl/helper_cuda.h>

#include <cudaPcl/timer.hpp>

using namespace Eigen;
using namespace std;

extern void reduction(float *h_mu, float *d_mu,
    float *h_dbg, float *d_dbg, int selection);

extern void reduction(double *h_mu, double *d_mu,
    double *h_dbg, double *d_dbg, int selection);

int main(int argc, char ** argv)
{
  Matrix<float,1,256> dbg;
  Matrix<float,6,4> mu;
  float *d_mu,*d_dbg;
  checkCudaErrors(cudaMalloc((void **)&d_mu, 6*4*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_dbg, 256*sizeof(float)));

  cudaPcl::Timer t0;
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mu.data(),d_mu,dbg.data(),d_dbg,0);
  t0.toctic("100 times old");
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mu.data(),d_mu,dbg.data(),d_dbg,1);
  t0.toctic("100 times new");
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mu.data(),d_mu,dbg.data(),d_dbg,2);
  t0.toctic("100 times newNew");
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mu.data(),d_mu,dbg.data(),d_dbg,4);
  t0.toctic("100 times oldOptimizedMemLayout");
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mu.data(),d_mu,dbg.data(),d_dbg,3);
  t0.toctic("100 times oldOptimized");
  cout<<mu<<endl<<dbg<<endl;

  checkCudaErrors(cudaFree(d_mu));
  checkCudaErrors(cudaFree(d_dbg));

  Matrix<double,1,256> dbgd;
  Matrix<double,6,4> mud;
  double *d_mud,*d_dbgd;
  checkCudaErrors(cudaMalloc((void **)&d_mud, 6*4*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_dbgd, 256*sizeof(double)));

  t0.tic();
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mud.data(),d_mud,dbgd.data(),d_dbgd,0);
  t0.toctic("100 times old");
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mud.data(),d_mud,dbgd.data(),d_dbgd,1);
  t0.toctic("100 times new");
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mud.data(),d_mud,dbgd.data(),d_dbgd,2);
  t0.toctic("100 times newNew");
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mud.data(),d_mud,dbgd.data(),d_dbgd,4);
  t0.toctic("100 times oldOptimizedMemLayout");
  for(uint32_t i=0; i< 1000; ++i)
    reduction(mud.data(),d_mud,dbgd.data(),d_dbgd,3);
  t0.toctic("100 times oldOptimized");
  cout<<mud<<endl<<dbgd<<endl;

  checkCudaErrors(cudaFree(d_mud));
  checkCudaErrors(cudaFree(d_dbgd));

  return 0;
}

