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
//#define COL_P_DOT_N 7
#define COL_DIM 7

template<typename T>
inline __device__ T square(T a) { return a*a;}
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
    T n[3];
    T p[3];
    d[id] = 0.;
    T dMin = 1e20;
    for (int32_t i=0; i<N; ++i) {
      p[0] = s[i*COL_DIM+COL_P_X];
      p[1] = s[i*COL_DIM+COL_P_Y];
      p[2] = s[i*COL_DIM+COL_P_Z];
      n[0] = s[i*COL_DIM+COL_N_X];
      n[1] = s[i*COL_DIM+COL_N_Y];
      n[2] = s[i*COL_DIM+COL_N_Z];
      T rSqMax = s[i*COL_DIM+COL_RSq];
      T pDotn = p[0]*n[0]+p[1]*n[1]+p[2]*n[2];
      T dsDotRay = ray[0]*n[0] + ray[1]*n[1] + ray[2]*n[2];
      T alpha = pDotn / dsDotRay;
      pt[0] = ray[0]*alpha;
      pt[1] = ray[1]*alpha;
      pt[2] = ray[2]*alpha;
      T rSq = square(pt[0]-p[0]) + square(pt[1]-p[1]) + square(pt[2]-p[2]);
      if (rSq < rSqMax && dMin > pt[2]) {
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
//  printf("surfelRenderGPU %d x %d",w,h);
  surfel_render<float><<<blocks, threads>>>(s,N,f,w,h,d);
  getLastCudaError("surfelRenderGPU() execution failed\n");
}
void surfelRenderGPU(double* s, int32_t N, double f, int32_t w, int32_t h, double *d)
{
  dim3 threads(16,16,1);
  dim3 blocks(w,h,1);
//  printf("surfelRenderGPU %d x %d",w,h);
  surfel_render<double><<<blocks, threads>>>(s,N,f,w,h,d);
  getLastCudaError("surfelRenderGPU() execution failed\n");
}

