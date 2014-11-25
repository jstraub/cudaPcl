/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */


#include <helper_cuda.h>
#include <stdint.h>

#include <stdio.h>

#define PI  3.141592653589793f
#define BLOCK_SIZE 256

//      int tidk = 0*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 1*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 2*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 3*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 4*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 5*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 6*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 7*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 8*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 9*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 10*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 11*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 12*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 13*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 14*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 15*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 16*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 17*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 18*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 19*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 20*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 21*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 22*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];
//      tidk = 23*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + s];

template<typename T>
__device__ inline T atomicAdd_(T* address, T val)
{};

template<>
__device__ inline double atomicAdd_<double>(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
          __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
};

template<>
__device__ inline float atomicAdd_<float>(float* address, float val)
{
  return atomicAdd(address,val);
};


template <uint16_t blockSize, class T>
__global__ void reduction_oldOptimizedMemLayout(T *mu_karch,T *dbg) //, T *N)
{
  __shared__ T mu[BLOCK_SIZE*4*6];

  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int tidk = tid*6*4;

#pragma unroll
  for(int s=0; s<6*4; ++s) {
    // this is almost certainly bad ordering
    mu[tid+BLOCK_SIZE*s] = 1.0f;
  }

  // old reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>0; s>>=1) {
    int ss = s*6*4;
    if(tid < s)
    {
#pragma unroll
      for( int k=0; k<6*4; ++k) {
        mu[tidk+k] += mu[tidk+k + ss];
      }
    }
    __syncthreads();
  }

  //dbg[tid] = mu[tid+BLOCK_SIZE];

  if(tid<6*4) {//  && Ni[(k/3)*BLOCK_SIZE]>0 ) {
    atomicAdd_<T>(&mu_karch[tid],mu[tid]);
  }
}

template <uint16_t blockSize, class T>
__global__ void reduction_oldOptimized(T *mu_karch,T *dbg) //, T *N)
{
  __shared__ T mu[BLOCK_SIZE*4*6];

  const int tid = threadIdx.x + blockDim.x * threadIdx.y;

#pragma unroll
  for(int s=0; s<6*4; ++s) {
    // this is almost certainly bad ordering
    mu[tid+BLOCK_SIZE*s] = 1.0f;
  }

  // old reduction.....
  __syncthreads(); //sync the threads

  if(blockSize >= 512) 
    if(tid<256){
#pragma unroll
      for( int k=0; k<6*4; ++k) { 
        int tidk = k*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + 256]; 
      }
      __syncthreads();
    }
  if(blockSize >= 256) 
    if(tid<128){
#pragma unroll
      for( int k=0; k<6*4; ++k) { 
        int tidk = k*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + 128]; 
      }
      __syncthreads();
    }
  if(blockSize >= 128) 
    if(tid<64){
#pragma unroll
      for( int k=0; k<6*4; ++k) { 
        int tidk = k*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + 64]; 
      }
      __syncthreads();
    }

    if(blockSize >= 64) 
      if(tid<32){
#pragma unroll
        for( int k=0; k<6*4; ++k) { int tidk = k*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + 32]; }
      }
    __syncthreads();
    if(blockSize >= 32) 
      if(tid<16){
#pragma unroll
        for( int k=0; k<6*4; ++k) { int tidk = k*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + 16]; }
        __syncthreads();
      }
    if(blockSize >= 16) {
      if(tid<8){
#pragma unroll
        for( int k=0; k<6*4; ++k) { int tidk = k*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + 8]; }
      }
      __syncthreads();
    }
    if(blockSize >= 8) {
      if(tid<4){
#pragma unroll
        for( int k=0; k<6*4; ++k) { int tidk = k*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + 4]; }
      }
      __syncthreads();
    }
    if(blockSize >= 4) {
      if(tid<2*6*4){
        int tidk = (tid/2)*BLOCK_SIZE + tid%2;
        mu[tidk] += mu[tidk+2];
          dbg[tid] = tidk;
//      if(tid<2){
//#pragma unroll
//        for( int k=0; k<6*4; ++k) { 
//          int tidk = k*BLOCK_SIZE+tid; mu[tidk] += mu[tidk + 2]; 
//          dbg[k*2+tid] = tidk;
//          dbg[tid+k*2] = tidk;
//        }
      }
    }
      __syncthreads();
//    if(blockSize >= 2) 
//    {
//      if(tid<6*4)
//      {
//        int tidk = tid*BLOCK_SIZE; 
//        mu[tidk] += mu[tidk+1];
//      }
//      __syncthreads();
//    }

//dbg[tid] = mu[tid+BLOCK_SIZE*19];

  if(tid<6*4) {//  && Ni[(k/3)*BLOCK_SIZE]>0 ) {
    int tidk = tid*BLOCK_SIZE; 
    //mu[tidk] += mu[tidk+1];
    atomicAdd_<T>(&mu_karch[tid],mu[tidk]+mu[tidk+1]);
    //atomicAdd_<T>(&mu_karch[tid],mu[tid*BLOCK_SIZE]);
  }
}

template<class T>
__global__ void reduction_old(T *mu_karch, T *dbg) //, T *N)
{
  __shared__ T mu[BLOCK_SIZE*4*6];

  const int tid = threadIdx.x + blockDim.x * threadIdx.y;

#pragma unroll
  for(int s=0; s<6*4; ++s) {
    // this is almost certainly bad ordering
    mu[tid+BLOCK_SIZE*s] = 1.0f;
  }

  // old reduction.....
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
    atomicAdd_<T>(&mu_karch[tid],mu[tid*BLOCK_SIZE]+mu[tid*BLOCK_SIZE+1]);
  }
}

template<class T>
__global__ void reduction_newNew(T *mu_karch,T *dbg) //, T *N)
{
  __shared__ T mu[BLOCK_SIZE*4*6];

  const int tid = threadIdx.x + blockDim.x * threadIdx.y;

#pragma unroll
  for(int s=0; s<6*4; ++s) {
    // this is almost certainly bad ordering
    mu[tid+BLOCK_SIZE*s] = 1.0f;
  }

  // old reduction.....
  __syncthreads(); //sync the threads
  int s = (BLOCK_SIZE)/2; //128
#pragma unroll
  for (uint32_t k=0; k<6*4; ++k)
  {
    int tidk = k*BLOCK_SIZE+tid;
    if(tid<s) mu[tidk] += mu[tidk + s];
  }

  s = (BLOCK_SIZE)/4;//64
#pragma unroll
  for (uint32_t k=0; k<6*2; ++k)
  {
#pragma unroll
    for (uint32_t j=0; j<2; ++j)
    {
      int ss = j*s;
      int tidk = (2*k+j)*BLOCK_SIZE+tid-ss;
      if(ss<=tid && tid<ss+s) mu[tidk] += mu[tidk + s];
    }
//    if(tid<s) 
//    {
//      int tidk = 2*k*BLOCK_SIZE+tid;
//      mu[tidk] += mu[tidk + s];
//    }else{
//      int tidk = (2*k+1)*BLOCK_SIZE+tid-s;
//      mu[tidk] += mu[tidk + s];
//    }
  }
  __syncthreads(); //sync the threads
  s = (BLOCK_SIZE)/8; //32
#pragma unroll
  for (uint32_t k=0; k<6; ++k)
  {
#pragma unroll
    for (uint32_t j=0; j<4; ++j)
    {
      int ss = j*s;
      int tidk = (4*k+j)*BLOCK_SIZE+tid-ss;
      if(ss<=tid && tid<ss+s) mu[tidk] += mu[tidk + s];
    }
  }

  __syncthreads(); //sync the threads
  s = (BLOCK_SIZE)/16; //16
#pragma unroll
  for (uint32_t k=0; k<3; ++k)
  {
#pragma unroll
    for (uint32_t j=0; j<8; ++j)
    {
      int ss = j*s;
      int tidk = (8*k+j)*BLOCK_SIZE+tid-ss;
      if(ss<=tid && tid<ss+s) mu[tidk] += mu[tidk + s];
    }
  }

  __syncthreads(); //sync the threads
  s = (BLOCK_SIZE)/32; //8
  uint32_t k = 0;
#pragma unroll
  for (uint32_t j=0; j<16; ++j)
  {
    int ss = j*s;
    int tidk = (16*k+j)*BLOCK_SIZE+tid-ss;
    if(ss<=tid && tid<ss+s) mu[tidk] += mu[tidk + s];
  }
  k = 1;
#pragma unroll
  for (uint32_t j=0; j<8; ++j)
  {
    int ss = j*s;
    int tidk = (16*k+j)*BLOCK_SIZE+tid-ss;
    if(ss<=tid && tid<ss+s) mu[tidk] += mu[tidk + s];
  }

  __syncthreads(); //sync the threads
  s = (BLOCK_SIZE)/64; //4
  k = 0;
#pragma unroll
  for (uint32_t j=0; j<24; ++j)
  {
    int ss = j*s;
    int tidk = (24*k+j)*BLOCK_SIZE+tid-ss;
    if(ss<=tid && tid<ss+s) mu[tidk] += mu[tidk + s];
  }

  //__syncthreads(); //sync the threads
  s = (BLOCK_SIZE)/128; //2
  k = 0;
#pragma unroll
  for (uint32_t j=0; j<24; ++j)
  {
    int ss = j*s;
    int tidk = (24*k+j)*BLOCK_SIZE+tid-ss;
    if(ss<=tid && tid<ss+s) mu[tidk] += mu[tidk + s];
  }

  //__syncthreads(); //sync the threads
  s = (BLOCK_SIZE)/256; //1
  k = 0;
#pragma unroll
  for (uint32_t j=0; j<24; ++j)
  {
    int tidk = (24*k+j)*BLOCK_SIZE+tid-j;
    if(tid==j) mu[tidk] += mu[tidk + s];
  }

  if(tid<6*4) {//  && Ni[(k/3)*BLOCK_SIZE]>0 ) {
    atomicAdd_<T>(&mu_karch[tid],mu[tid*BLOCK_SIZE]);
  }
}

template<class T>
__global__ void reduction_new(T *mu_karch,T *dbg) //, T *N)
{
  __shared__ T mu[BLOCK_SIZE*4*6];

  const int tid = threadIdx.x + blockDim.x * threadIdx.y;

#pragma unroll
  for(int s=0; s<6*4; ++s) {
    // this is almost certainly bad ordering
    mu[tid+BLOCK_SIZE*s] = 1.0f;
  }


  bool exit=false;
  int tpr = BLOCK_SIZE/(4*6); // threads per row
  //reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int r=0; r<4*6; ++r)
  {
    if (r*tpr <= tid && tid < (r+1)*tpr)
    {
      int tidr = tid - r*tpr; // id in row
      int offset = r*BLOCK_SIZE+tidr;
      //dbg[id] = offset;
#pragma unroll
      for(int s=(BLOCK_SIZE)/2; s>0; s>>=1) 
      {
        int expr = s/tpr; // executions per row
        //dbg[id] = expr;
        for (int ex=0; ex<expr; ++ex)
          mu[offset+ex*tpr] += mu[offset+ex*tpr+s];
        int exprem = s%tpr; // remaining executions
        if (tidr <exprem)
          mu[offset+expr*tpr] += mu[offset+expr*tpr+s];
        __syncthreads();
        if(s==BLOCK_SIZE/4)
        {
        exit=true;
        break;
        }
      }
    }
    if(exit) break;
  }
  //dbg[id] = mu[id+BLOCK_SIZE*3];
  //dbg[id] =tid;
  if(tid<6*4) {//  && Ni[(k/3)*BLOCK_SIZE]>0 ) {
    atomicAdd_<T>(&mu_karch[tid],mu[tid*BLOCK_SIZE]);
  }
}

extern void reduction(float *h_mu, float *d_mu,
    float *h_dbg, float *d_dbg, int selection)
{
  for(uint32_t i=0; i<4*6; ++i)
    h_mu[i] =float(0.0f);
  for(uint32_t i=0; i<256; ++i)
    h_dbg[i] =float(0.0f);
  checkCudaErrors(cudaMemcpy(d_mu, h_mu, 6*4* sizeof(float), 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_dbg, h_dbg, 256* sizeof(float), 
        cudaMemcpyHostToDevice));
  dim3 threads(16,16,1);
  //dim3 blocks(w/16+(w%16>0?1:0), h/16+(h%16>0?1:0),1);
  dim3 blocks(1,1,1);
  if(selection == 0){
    reduction_old<float><<<blocks,threads>>>(d_mu,d_dbg);
  }else if(selection == 1)
  {
    reduction_new<float><<<blocks,threads>>>(d_mu,d_dbg);
  }else if (selection ==2)
    reduction_newNew<float><<<blocks,threads>>>(d_mu,d_dbg);
  else if (selection ==3)
    reduction_oldOptimized<256,float><<<blocks,threads>>>(d_mu,d_dbg);
  else if (selection ==4)
    reduction_oldOptimizedMemLayout<256,float><<<blocks,threads>>>(d_mu,d_dbg);

  checkCudaErrors(cudaDeviceSynchronize());
  
  checkCudaErrors(cudaMemcpy(h_mu, d_mu, 6*4*sizeof(float), 
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_dbg, d_dbg, 256*sizeof(float), 
        cudaMemcpyDeviceToHost));
  
}

extern void reduction(double *h_mu, double *d_mu,
    double *h_dbg, double *d_dbg, int selection)
{
  for(uint32_t i=0; i<4*6; ++i)
    h_mu[i] =double(0.0f);
  for(uint32_t i=0; i<256; ++i)
    h_dbg[i] =double(0.0f);
  checkCudaErrors(cudaMemcpy(d_mu, h_mu, 6*4* sizeof(double), 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_dbg, h_dbg, 256* sizeof(double), 
        cudaMemcpyHostToDevice));
  dim3 threads(16,16,1);
  //dim3 blocks(w/16+(w%16>0?1:0), h/16+(h%16>0?1:0),1);
  dim3 blocks(1,1,1);
  if(selection == 0){
    reduction_old<double><<<blocks,threads>>>(d_mu,d_dbg);
  }else if(selection == 1)
  {
    reduction_new<double><<<blocks,threads>>>(d_mu,d_dbg);
  }else if (selection ==2)
    reduction_newNew<double><<<blocks,threads>>>(d_mu,d_dbg);
  else if (selection ==3)
    reduction_oldOptimized<256,double><<<blocks,threads>>>(d_mu,d_dbg);
  else if (selection ==4)
    reduction_oldOptimizedMemLayout<256,double><<<blocks,threads>>>(d_mu,d_dbg);

  checkCudaErrors(cudaDeviceSynchronize());
  
  checkCudaErrors(cudaMemcpy(h_mu, d_mu, 6*4*sizeof(double), 
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_dbg, d_dbg, 256*sizeof(double), 
        cudaMemcpyDeviceToHost));
  
}
