#include <stdio.h>
#include <stdint.h>
#define BLK_SIZE 128

__global__ void integral_kernel(float* in, float* inSum, float* carry, uint32_t w, uint32_t h)
{
  __shared__ float tmp[2*BLK_SIZE];
  
  int tdx = threadIdx.x;
  int idx0 = (threadIdx.x + blockIdx.x*blockDim.x)*2 + (blockIdx.y*blockDim.y + threadIdx.y)*w;
  int idx1 = (threadIdx.x + blockIdx.x*blockDim.x)*2+1 + (blockIdx.y*blockDim.y + threadIdx.y)*w;
  int idx = (blockIdx.x*blockDim.x + threadIdx.x);
  int idy = (blockIdx.y*blockDim.y + threadIdx.y);
  tmp[2*tdx] = in[idx0] ;
  tmp[2*tdx+1] = in[idx1];

//  float carr = carry[blockIdx.x + idy*(w/(2*BLK_SIZE))];
//  if(tdx==0)
//    printf("%d\t%d\t%d\t%d\t%f\n",blockIdx.x,w/(2*BLK_SIZE), idy,blockIdx.x + idy*(w/(2*BLK_SIZE)),carr);
    

  int offset =1;
  for(int s=BLK_SIZE; s>0; s>>=1)
  {
    __syncthreads();
    if(tdx<s)
      tmp[offset*(2*tdx+2)-1] += tmp[offset*(2*tdx+1)-1]; 
    offset <<= 1;
  }
  //TODO save the larges elements tmp[2*BLK_SIZE-1] out to later add it back in
  if(tdx==0) 
  {
//    printf("%d\t%d\t%d\t%d\t%f\n",blockIdx.x,w/(2*BLK_SIZE), idy,blockIdx.x + idy*(w/(2*BLK_SIZE)), tmp[2*BLK_SIZE-1]);
    if( blockIdx.x< w/(2*BLK_SIZE) -1)
      carry[ blockIdx.x + idy*(w/(2*BLK_SIZE))] = tmp[2*BLK_SIZE-1]; //+carr;
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
      float t = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += t;
    }
  }

  __syncthreads();
  inSum[idx0] = tmp[2*tdx]  ; // + carr;
  inSum[idx1] = tmp[2*tdx+1]; // + carr;
}

__global__ void addCarryOver_kernel(float* inOut, float* carry, uint32_t w, uint32_t h)
{
  int tdx = threadIdx.x;
  int idx = (blockIdx.x*blockDim.x + threadIdx.x);
  int idy = (blockIdx.y*blockDim.y + threadIdx.y);

  if(blockIdx.x>0)
  {
    float carr = carry[0 + idy*(w/(2*BLK_SIZE))];
    for(int i=1; i<=blockIdx.x-1 ;++i)
      carr += carry[i + idy*(w/(2*BLK_SIZE))];
    if((idx<w)&&(idy<h))
      inOut[idx+idy*w] += carr;
  }
}

#define BLK_SIZE_S 16
__global__ void transpose_kernel(float* in, float* out, uint32_t w, uint32_t h)
{
  __shared__ float tmp[BLK_SIZE_S][BLK_SIZE_S];
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
