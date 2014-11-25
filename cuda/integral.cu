/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <helper_cuda.h>
#include <stdio.h>
#include <stdint.h>
//#define BLK_SIZE 128

template<typename T, uint32_t BLK_SIZE>
__global__ void cumSumRows_kernel(T* in, T* inSum, T* carry, uint32_t w, uint32_t h)
{
  __shared__ T tmp[2*BLK_SIZE];
  
  int tdx = threadIdx.x;
  int idx = (blockIdx.x*blockDim.x + threadIdx.x);
  int idy = (blockIdx.y*blockDim.y + threadIdx.y);
  int idx0 = idx*2 + idy*w;
  int idx1 = idx*2+1 + idy*w;
  tmp[2*tdx] = in[idx0] ;
  tmp[2*tdx+1] = in[idx1];

  int offset =1;
  for(int s=BLK_SIZE; s>0; s>>=1)
  {
    __syncthreads();
    if(tdx<s)
      tmp[offset*(2*tdx+2)-1] += tmp[offset*(2*tdx+1)-1]; 
    offset <<= 1;
  }

  if(tdx==0) 
  {
    if( blockIdx.x< w/(2*BLK_SIZE) -1)
      carry[ blockIdx.x + idy*(w/(2*BLK_SIZE))] = tmp[2*BLK_SIZE-1]; 
    tmp[2*BLK_SIZE-1] =0;
  }

  for(int s=1; s<2*BLK_SIZE; s <<=1)
  {
    offset >>=1; 
    __syncthreads();
    if(tdx < s)
    {
      int ai = offset*(2*tdx+1) -1;
      int bi = offset*(2*tdx+2) -1;
      T t = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += t;
    }
  }

  __syncthreads();
  inSum[idx0] = tmp[2*tdx]  ; 
  inSum[idx1] = tmp[2*tdx+1];
}


template<typename T, uint32_t BLK_SIZE>
__global__ void addCarryOver_kernel(T* inOut, T* carry, uint32_t w, uint32_t h)
{
  int idx = (blockIdx.x*blockDim.x + threadIdx.x);
  int idy = (blockIdx.y*blockDim.y + threadIdx.y);

  if(blockIdx.x>0)
  {
    T carr = carry[0 + idy*(w/(2*BLK_SIZE))];
    for(int i=1; i<=blockIdx.x-1 ;++i)
      carr += carry[i + idy*(w/(2*BLK_SIZE))];
    if((idx<w)&&(idy<h))
      inOut[idx+idy*w] += carr;
  }
}

//#define BLK_SIZE_S 16
template<typename T, uint32_t BLK_SIZE>
__global__ void transpose_kernel(T* in, T* out, uint32_t w, uint32_t h)
{
  __shared__ T tmp[BLK_SIZE][BLK_SIZE];
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;

  if((idx<w)&&(idy<h))
    tmp[threadIdx.y][threadIdx.x] = in[idy*w +idx];
  __syncthreads();

  idx = blockIdx.y*blockDim.y + threadIdx.x;
  idy = blockIdx.x*blockDim.x + threadIdx.y;

  if((idx<w)&&(idy<h))
    out[idy*h+idx] = tmp[threadIdx.x][threadIdx.y] ;
}

void integralGpu(double* I, double* Isum, double* IsumT, double* IsumCarry, uint32_t w, uint32_t h)
{
  dim3 threads(256,1,1);
  dim3 threadsHalf(128,1,1);
  dim3 blocks(w/256+(w%256>0?1:0), h,1);
  dim3 threadsSq(16,16,1);
  dim3 blocksSq(w/16+(w%16>0?1:0), h/16+(h%16>0?1:0),1);

//  double* IsumCarry;
//  double* IsumT;
//  checkCudaErrors(cudaMalloc((void **)&IsumCarry, h*(w/(2*128))*sizeof(double))); 
//  checkCudaErrors(cudaMalloc((void **)&IsumT, w*h*sizeof(double))); 

//  printf("integralGPU on %dx%d\n",w,h);
//  printf("integralGPU IsumCarry: %dx%d\n",(w/(2*128)),h);
//

  cumSumRows_kernel<double,128><<<blocks,threadsHalf>>>(I,Isum,IsumCarry,w,h); 
  checkCudaErrors(cudaDeviceSynchronize());
//  addCarryOver_kernel<double,128><<<blocks,threads>>>(Isum,IsumCarry,w,h); 
//  checkCudaErrors(cudaDeviceSynchronize());
  transpose_kernel<double,16><<<blocksSq,threadsSq>>>(Isum,IsumT,w,h); 
  checkCudaErrors(cudaDeviceSynchronize());

  cumSumRows_kernel<double,128><<<blocks,threadsHalf>>>(IsumT,Isum,IsumCarry,w,h); 
  checkCudaErrors(cudaDeviceSynchronize());
//  addCarryOver_kernel<double,128><<<blocks,threads>>>(Isum,IsumCarry,w,h); 
//  checkCudaErrors(cudaDeviceSynchronize());
//  transpose_kernel<double,16><<<blocksSq,threadsSq>>>(Isum,IsumT,w,h); 
//  checkCudaErrors(cudaDeviceSynchronize());

//  checkCudaErrors(cudaFree(IsumT));
//  checkCudaErrors(cudaFree(IsumCarry));
};
