/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <stdint.h>
#include <assert.h>
#include <nvidia/helper_cuda.h>

#define COL_P_X 0
#define COL_P_Y 1
#define COL_P_Z 2
#define COL_N_X 3
#define COL_N_Y 4
#define COL_N_Z 5
#define COL_RSq 6
#define COL_DIM 7

/*
 * compute the xyz images using the inverse focal length invF
 */
template<typename T>
__global__ void surfel_render(T* s, int32_t N, T f, int32_t w, int32_t h, T *d)
{
  const int32_t idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int32_t idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int32_t id = idx+w*idy;

  if(idx<w && idy<h)
  {
    T ray[3];
    ray[0] = T(idx)-(w-1.)*0.5;
    ray[1] = T(idy)-(h-1.)*0.5;
    ray[2] = f;
    T pt[3];
    d[id] = 0.;
    T dMin = 1e20;
    for (int32_t i=0; i<N; ++i) {
      // TODO can precompute this.
      T psDotRayS = s[i*COL_DIM+COL_P_X]*s[i*COL_DIM+COL_N_X] + s[i*COL_DIM+COL_P_Y]*s[i*COL_DIM+COL_N_Y] + s[i*COL_DIM+COL_P_Z]*s[i*COL_DIM+COL_N_Z];
      T dsDotRay = ray[0]*s[i*COL_DIM+COL_N_X] + ray[1]*s[i*COL_DIM+COL_N_Y] + ray[2]*s[i*COL_DIM+COL_N_Z];
      T alpha = psDotRayS / dsDotRay;
      pt[0] = ray[0]*alpha;
      pt[1] = ray[1]*alpha;
      pt[2] = ray[2]*alpha;
      T rSq = (pt[0]-s[i*COL_DIM+COL_P_X])*(pt[0]-s[i*COL_DIM+COL_P_X]) +
        (pt[1]-s[i*COL_DIM+COL_P_Y])*(pt[1]-s[i*COL_DIM+COL_P_Y]) +
        (pt[2]-s[i*COL_DIM+COL_P_Z])*(pt[2]-s[i*COL_DIM+COL_P_Z]);
      if (rSq < s[i*COL_DIM+COL_RSq] && dMin > pt[2]) {
        // ray hit the surfel 
        dMin = pt[2];
      }
    }
    d[id] = dMin > 100.? 0. : dMin;
  }
}

void surfelRenderGPU(float* s, int32_t N, float f, int32_t w, int32_t h, float *d)
{
  dim3 threads(16,16,1);
  dim3 blocks(w,h,1);
  printf("surfelRenderGPU %d x %d",w,h);
  surfel_render<float><<<blocks, threads>>>(s,N,f,w,h,d);
  getLastCudaError("surfelRenderGPU() execution failed\n");
}
void surfelRenderGPU(double* s, int32_t N, double f, int32_t w, int32_t h, double *d)
{
  dim3 threads(16,16,1);
  dim3 blocks(w,h,1);
  printf("surfelRenderGPU %d x %d",w,h);
  surfel_render<double><<<blocks, threads>>>(s,N,f,w,h,d);
  getLastCudaError("surfelRenderGPU() execution failed\n");
}

