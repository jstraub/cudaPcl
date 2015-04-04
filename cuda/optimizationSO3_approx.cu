/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <stdint.h>
#include <stdio.h>

#include <nvidia/helper_cuda.h>
#include <cudaPcl/cudaSphereHelpers.h>

#define BLOCK_WIDTH 16
#define BLOCK_SIZE BLOCK_WIDTH*BLOCK_WIDTH

// step size of the normals 
// for PointXYZI
#define X_STEP 8
#define X_OFFSET 0
// for PointXYZ
//#define X_STEP 4
//#define X_OFFSET 0

// TODO: try to copy the points in the tangent space out of the memory
//  and process them on CPU
__global__ void meanInTpS2(float *d_p, float *d_q, unsigned short *z, 
    float *mu_karch, int w, int h) //, float *N)
{
  __shared__ float p[3*6];
  // one J per column; BLOCK_SIZE columns; per column first 3 first col of J, 
  // second 3 columns second cols of J 
  // forth row is number of associated points
  __shared__ float mu[BLOCK_SIZE*4*6];
  //__shared__ float Ni[BLOCK_SIZE*6];

  //const int tid = threadIdx.x;
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  // caching 
  if(tid < 3*6) p[tid] = d_p[tid];
#pragma unroll
  for(int s=0; s<6*4; ++s) {
    // this is almost certainly bad ordering
    mu[tid+BLOCK_SIZE*s] = 0.0f;
  }
//#pragma unroll
//  for(int s=0; s<6; ++s) {
//    Ni[tid+BLOCK_SIZE*s] = 0.0f;
//  }
  __syncthreads(); // make sure that ys have been cached

  for(uint32_t ix=0; ix<8; ++ix)
    for(uint32_t iy=0; iy<4; ++iy)
    {
      int id = idx+ix*w/8 + (idy+iy*h/4)*w;
      if (id<w*h)
      {
        uint16_t zi = z[id];
        if(zi<6){ // if point is good
          float q[3], x[3];
          q[0] = d_q[id*X_STEP+X_OFFSET+0]; 
          q[1] = d_q[id*X_STEP+X_OFFSET+1]; 
          q[2] = d_q[id*X_STEP+X_OFFSET+2];
          Log_p(p+zi*3,q,x);
//          float dot = min(1.0f,max(-1.0f,q[0]*p[zi*3+0] + q[1]*p[zi*3+1] 
//                + q[2]*p[zi*3+2]));
//          float theta = acosf(dot);
//          float sinc;
//          if(theta < 1.e-8)
//            sinc = 1.0f;
//          else
//            sinc = theta/sinf(theta);
//          float x[3]; 
//          x[0] = (q[0]-p[zi*3+0]*dot)*sinc;
//          x[1] = (q[1]-p[zi*3+1]*dot)*sinc;
//          x[2] = (q[2]-p[zi*3+2]*dot)*sinc;
          mu[tid+(zi*4+0)*BLOCK_SIZE] += x[0];
          mu[tid+(zi*4+1)*BLOCK_SIZE] += x[1];
          mu[tid+(zi*4+2)*BLOCK_SIZE] += x[2];
          mu[tid+(zi*4+3)*BLOCK_SIZE] += 1.0f;
        }
      }
    }

  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
#pragma unroll
      for( int k=0; k<6*4; ++k) {
        int tidk = k*BLOCK_SIZE+tid;
        mu[tidk] += mu[tidk + s];
      }
    }
    __syncthreads();
  }
  if(tid<6*4) {//  && Ni[(k/3)*BLOCK_SIZE]>0 ) {
    atomicAdd(&mu_karch[tid],mu[tid*BLOCK_SIZE]+mu[tid*BLOCK_SIZE+1]);
  }
}

__global__ void meanInTpS2(float *d_p, float *d_q, unsigned short *z, 
    float* d_weights, float *mu_karch, int w, int h) //, float *N)
{
  __shared__ float p[3*6];
  // one J per column; BLOCK_SIZE columns; per column first 3 first col of J, 
  // second 3 columns second cols of J 
  // forth row is number of associated points
  __shared__ float mu[BLOCK_SIZE*4*6];
  //__shared__ float Ni[BLOCK_SIZE*6];

  //const int tid = threadIdx.x;
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  // caching 
  if(tid < 3*6) p[tid] = d_p[tid];
#pragma unroll
  for(int s=0; s<6*4; ++s) {
    // this is almost certainly bad ordering
    mu[tid+BLOCK_SIZE*s] = 0.0f;
  }
//#pragma unroll
//  for(int s=0; s<6; ++s) {
//    Ni[tid+BLOCK_SIZE*s] = 0.0f;
//  }
  __syncthreads(); // make sure that ys have been cached

  for(uint32_t ix=0; ix<8; ++ix)
    for(uint32_t iy=0; iy<4; ++iy)
    {
      int id = idx+ix*w/8 + (idy+iy*h/4)*w;
      if (id<w*h)
      {
        uint16_t zi = z[id];
        float wi = d_weights[id];
        if(zi<6){ // if point is good
          float q[3],x[3];
          q[0] = d_q[id*X_STEP+X_OFFSET+0]; 
          q[1] = d_q[id*X_STEP+X_OFFSET+1]; 
          q[2] = d_q[id*X_STEP+X_OFFSET+2];
          Log_p(p+zi*3,q,x);
//          float dot = min(1.0f,max(-1.0f,q[0]*p[zi*3+0] + q[1]*p[zi*3+1] 
//                + q[2]*p[zi*3+2]));
//          float theta = acosf(dot);
//          float sinc;
//          if(theta < 1.e-8)
//            sinc = 1.0f;
//          else
//            sinc = theta/sinf(theta);
//          float x[3]; 
//          x[0] = (q[0]-p[zi*3+0]*dot)*sinc;
//          x[1] = (q[1]-p[zi*3+1]*dot)*sinc;
//          x[2] = (q[2]-p[zi*3+2]*dot)*sinc;
          mu[tid+(zi*4+0)*BLOCK_SIZE] += wi*x[0];
          mu[tid+(zi*4+1)*BLOCK_SIZE] += wi*x[1];
          mu[tid+(zi*4+2)*BLOCK_SIZE] += wi*x[2];
          mu[tid+(zi*4+3)*BLOCK_SIZE] += wi;
        }
      }
    }

  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
#pragma unroll
      for( int k=0; k<6*4; ++k) {
        int tidk = k*BLOCK_SIZE+tid;
        mu[tidk] += mu[tidk + s];
      }
    }
    __syncthreads();
  }
  if(tid<6*4) {//  && Ni[(k/3)*BLOCK_SIZE]>0 ) {
    atomicAdd(&mu_karch[tid],mu[tid*BLOCK_SIZE]+mu[tid*BLOCK_SIZE+1]);
  }
}



extern "C" void meanInTpS2GPU(float *h_p, float *d_p, float *h_mu_karch,
    float *d_mu_karch, float *d_q, uint16_t *d_z, float* d_weights ,int w, int h)
{
  for(uint32_t i=0; i<4*6; ++i)
    h_mu_karch[i] =0.0f;
  checkCudaErrors(cudaMemcpy(d_mu_karch, h_mu_karch, 6*4* sizeof(float), 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_p, h_p, 6*3* sizeof(float), 
        cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_WIDTH,BLOCK_WIDTH,1);
  // this way for 640x480 there is no remainders
  //dim3 blocks(w/128+(w%128>0?1:0), h/32+(h%32>0?1:0),1);
  // this still seems to be fastest
  dim3 blocks(w/128+(w%128>0?1:0), h/64+(h%64>0?1:0),1);
  //printf("%d x %d",w/32+(w%32>0?1:0),h/16+(h%16>0?1:0));
  if(d_weights == NULL)
    meanInTpS2<<<blocks,threads>>>(d_p,d_q, d_z, d_mu_karch,w,h);
  else
    meanInTpS2<<<blocks,threads>>>(d_p,d_q, d_z, d_weights, d_mu_karch,w,h);

    checkCudaErrors(cudaDeviceSynchronize());
  
  checkCudaErrors(cudaMemcpy(h_mu_karch, d_mu_karch, 6*4*sizeof(float), 
        cudaMemcpyDeviceToHost));
};

__global__ void sufficientStatisticsOnTpS2(
  float *d_p, float *Rnorths,
  float *d_q, unsigned short *z, int w, int h, 
  float *SSs
  ) //, float *N)
{
  __shared__ float p[3*6];
  // sufficient statistics for whole blocksize
  // 2 (x in TpS @north) + 1 (count) + 4 (outer product in TpS @north)
  // all fo that times 6 for the different axes
  __shared__ float xSSs[BLOCK_SIZE*(2+1+4)*6];
  __shared__ float sRnorths[6*6];

  //const int tid = threadIdx.x;
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  // caching 
  if(tid < 3*6) p[tid] = d_p[tid];
  if(3*6 <= tid && tid <3*6+6*6) sRnorths[tid-3*6] = Rnorths[tid-3*6];
#pragma unroll
  for(int s=0; s<6*7; ++s) {
    // this is almost certainly bad ordering
    xSSs[tid+BLOCK_SIZE*s] = 0.0f;
  }
//#pragma unroll
//  for(int s=0; s<6; ++s) {
//    Ni[tid+BLOCK_SIZE*s] = 0.0f;
//  }
  __syncthreads(); // make sure that ys have been cached

  for(uint32_t ix=0; ix<8; ++ix)
    for(uint32_t iy=0; iy<4; ++iy)
    {
      int id = idx+ix*w/8 + (idy+iy*h/4)*w;
      if (id<w*h)
      {
        uint16_t zi = z[id];
        if(zi<6){ // if point is good
          // copy q into local memory
          float q[3];
          q[0] = d_q[id*X_STEP+X_OFFSET+0]; 
          q[1] = d_q[id*X_STEP+X_OFFSET+1]; 
          q[2] = d_q[id*X_STEP+X_OFFSET+2];
          // transform to TpS^2
          float dot = min(1.0f,max(-1.0f,q[0]*p[zi*3+0] + q[1]*p[zi*3+1] 
                + q[2]*p[zi*3+2]));
          float theta = acosf(dot);
          float sinc;
          if(theta < 1.e-8)
            sinc = 1.0f;
          else
            sinc = theta/sinf(theta);
          float x[3]; 
          x[0] = (q[0]-p[zi*3+0]*dot)*sinc;
          x[1] = (q[1]-p[zi*3+1]*dot)*sinc;
          x[2] = (q[2]-p[zi*3+2]*dot)*sinc;
          // rotate up to north pole
          float xNorth[2];
          xNorth[0] = sRnorths[zi*6+0]*x[0] + sRnorths[zi*6+1]*x[1] 
            + sRnorths[zi*6+2]*x[2];  
          xNorth[1] = sRnorths[zi*6+3]*x[0] + sRnorths[zi*6+4]*x[1] 
            + sRnorths[zi*6+5]*x[2];  
          // input sufficient statistics
          xSSs[tid+(zi*7+0)*BLOCK_SIZE] += xNorth[0];
          xSSs[tid+(zi*7+1)*BLOCK_SIZE] += xNorth[1];
          xSSs[tid+(zi*7+2)*BLOCK_SIZE] += xNorth[0]*xNorth[0];
          xSSs[tid+(zi*7+3)*BLOCK_SIZE] += xNorth[1]*xNorth[0];
          xSSs[tid+(zi*7+4)*BLOCK_SIZE] += xNorth[0]*xNorth[1];
          xSSs[tid+(zi*7+5)*BLOCK_SIZE] += xNorth[1]*xNorth[1];
          xSSs[tid+(zi*7+6)*BLOCK_SIZE] += 1.0f;
        }
      }
    }

  // old reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
#pragma unroll
      for( int k=0; k<6*7; ++k) {
        int tidk = k*BLOCK_SIZE+tid;
        xSSs[tidk] += xSSs[tidk + s];
      }
    }
    __syncthreads();
  }
    if(tid < 6*7) {//  && Ni[(k/3)*BLOCK_SIZE]>0 ) {
      atomicAdd(&SSs[tid],xSSs[tid*BLOCK_SIZE]+xSSs[tid*BLOCK_SIZE+1]);
    }
}


extern "C" void sufficientStatisticsOnTpS2GPU(float *h_p, float *d_p, 
  float *h_Rnorths, float *d_Rnorths, float *d_q, uint16_t *d_z ,int w, int h,
  float *h_SSs, float *d_SSs)
{
  for(uint32_t i=0; i<7*6; ++i)
    h_SSs[i] =0.0f;
  checkCudaErrors(cudaMemcpy(d_SSs, h_SSs, 6*7* sizeof(float), 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_p, h_p, 6*3* sizeof(float), 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Rnorths, h_Rnorths, 6*6* sizeof(float), 
        cudaMemcpyHostToDevice));

  dim3 threads(16,16,1);
  dim3 blocks(w/128+(w%128>0?1:0), h/64+(h%64>0?1:0),1);
  sufficientStatisticsOnTpS2<<<blocks,threads>>>(d_p,d_Rnorths, 
    d_q, d_z,w,h,d_SSs);
  checkCudaErrors(cudaDeviceSynchronize());
  
  checkCudaErrors(cudaMemcpy(h_SSs, d_SSs, 6*7*sizeof(float), 
        cudaMemcpyDeviceToHost));
};
