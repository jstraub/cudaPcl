/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <helper_cuda.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
//#define BLK_SIZE 128

template<typename T, int32_t B>
__device__
inline T integralGetCompLoc(T* A, int32_t i, int32_t j, int32_t w, int32_t h)
{
  const int32_t iNeg = max(i-B,0)*(w+1);
  const int32_t jNeg = max(j-B,0);
  const int32_t iPos = min(i+B,h)*(w+1);
  const int32_t jPos = min(j+B,w);

//  printf("%d %d %d %d %d\n",iNeg/w,jNeg,iPos/w,jPos,B);
  const T a = A[iPos + jPos] //    (min(i+w,A.rows-1),min(j+w,A.cols-1)) 
    - A[iPos + jNeg] //.at<T>(min(i+w,A.rows-1),max(j-w,0)) 
    - A[iNeg + jPos] // .at<T>(max(i-w,0),min(j+w,A.cols-1)) 
    + A[iNeg + jNeg];//.at<T>(max(i-w,0),max(j-w,0));
//  if (print)  printf("%d %d %d %d %d\n",iNeg/w,jNeg,iPos/w,jPos,a);
  return a;
};

template<typename T>
__device__
inline T integralGet(T* A, int32_t lu, int32_t ld, int32_t rd, int32_t ru)
{
  return A[rd] - A[ru] - A[ld] + A[lu];
};

template<int32_t B>
__device__
inline bool integralCheck(uint8_t* haveData, int32_t i, int32_t j, int32_t
    w, int32_t h, int32_t* lu, int32_t* ld, int32_t* rd, int32_t* ru)
{
  int32_t iN = max(i-B,0);
  int32_t jN = max(j-B,0);
  int32_t iP = min(i+B,h-1);
  int32_t jP = min(j+B,w-1);

  *lu = iN*w + jN;
  *ru = iN*w + jP;
  *ld = iP*w + jN;
  *rd = iP*w + jP;
//  while(iN < i && (!haveData[*ru] || !haveData[*lu]))
//  {
//    iN ++;
//    *lu +=w;
//    *ru +=w;
//  }
//  while(i < iP && (!haveData[*rd] || !haveData[*ld]))
//  {
//    iP --;
//    *ld -=w;
//    *rd -=w;
//  }
//
//  while(jN < j && (!haveData[*lu] || !haveData[*ld]))
//  {
//    jN ++;
//    *lu +=1;
//    *ld +=1;
//  }
//
//  while(j < jP && (!haveData[*ru] || !haveData[*rd]))
//  {
//    jP --;
//    *ru -=1;
//    *rd -=1;
//  }
//
  if(!haveData[*lu] || !haveData[*ld] || !haveData[*ru] || !haveData[*rd]
      || iN ==0 || jN == 0 || iP == h-1 || jP == w-1)
  {
//  printf("%d %d: %d %d %d %d\n",i,j,iN,jN,iP,jP);
    return false;
  }
//

  *lu += iN;
  *ru += iN;
  *ld += iP;
  *rd += iP;

//  *lu = iN*(w+1) + jN;
//  *ru = iN*(w+1) + jP;
//  *ld = iP*(w+1) + jN;
//  *rd = iP*(w+1) + jP;
  
//  *lu += max(i-B,0);
//  *ru += max(i-B,0) - jPos + jPosS;
//  *ld += min(i+B,h) + iPos - iPosS;
//  *rd += min(i+B,h) + iPos + jPos - iPosS - jPosS;

  return true;
};

template<typename T, uint32_t BLK_SIZE, int32_t B>
__global__ void guidedFilter_ab_kernel(uint8_t* haveData, uint8_t* haveDataAfter, T* a, T* b, int32_t*
    Ns, T* dSum, T* dSqSum, double eps, uint32_t w, uint32_t h)
{

  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if((idx<w)&&(idy<h) && (haveData[id]))
  {
    int32_t lu,ld,rd,ru;
    if(!integralCheck<B>(haveData,idy,idx,w,h,&lu,&ld,&rd,&ru)) // get the coordinates for integral img
    {
      haveDataAfter[id] = 0;
//      b[id] = 0.0;
//      a[id] = 1.0;
      return;
    }
    const T n = integralGet<int32_t>(Ns,lu,ld,rd,ru); 
    if(n < (2*B)*(2*B))
    {
      haveDataAfter[id] = 0;
//      haveData[id] =0;
//      b[id] = 0.0;
//      a[id] = 1.0;
      return;
    }
    const T muG = integralGet<T>(dSum,lu,ld,rd,ru);
    const T s = integralGet<T>(dSqSum,lu,ld,rd,ru);
    const T muSq = muG*muG;
    const T n1 = n-1.;
    const T a_ = ((n*s-muSq)*n1)/((s*n-muSq+eps*n1*n)*n);

    b[id] = muG*(1. - a_)/n;
    a[id] = a_;
    haveDataAfter[id] = 1;
  }
}

void guidedFilter_ab_gpu(uint8_t* haveData, uint8_t* haveDataAfter, double* a, double* b, int32_t* Ns,
    double* dSum, double* dSqSum, double eps, uint32_t B, uint32_t w, uint32_t
    h)
{
  dim3 threadsSq(16,16,1);
  dim3 blocksSq(w/16+(w%16>0?1:0), h/16+(h%16>0?1:0),1);

  if(B==3){
    guidedFilter_ab_kernel<double,16,3><<<blocksSq,threadsSq>>>(haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==5){
    guidedFilter_ab_kernel<double,16,5><<<blocksSq,threadsSq>>>(haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==6){
    guidedFilter_ab_kernel<double,16,6><<<blocksSq,threadsSq>>>(haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==7){
    guidedFilter_ab_kernel<double,16,7><<<blocksSq,threadsSq>>>(haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==8){
    guidedFilter_ab_kernel<double,16,8><<<blocksSq,threadsSq>>>(haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==9){
    guidedFilter_ab_kernel<double,16,9><<<blocksSq,threadsSq>>>(haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==10){
    guidedFilter_ab_kernel<double,16,10><<<blocksSq,threadsSq>>>(haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

template<typename T, typename Tout, uint32_t BLK_SIZE, int32_t B>
__global__ void guidedFilter_out_kernel(uint8_t* haveData, T* depth, T* aInt, T* bInt, int32_t*
    Ns, Tout* depthSmooth, uint32_t w, uint32_t h, double missingValue)
{

  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if((idx<w)&&(idy<h))
    if(haveData[id])
    {
      int32_t lu,ld,rd,ru;
      if(!integralCheck<B>(haveData,idy,idx,w,h,&lu,&ld,&rd,&ru)) // get the coordinates for integral img
      {
        depthSmooth[id] = missingValue;
//      haveData[id] =0;
        return;
      }
//    integralCheck<B>(haveData,idy,idx,w,h,&lu,&ld,&rd,&ru);
      const T n = integralGet<int32_t>(Ns,lu,ld,rd,ru); 
//    if(n < (2*B)*(2*B))
//    {
//      depthSmooth[id] = 0.0;
////      haveData[id] =0;
//      return;
//    }
      const T muA = integralGet<T>(aInt,lu,ld,rd,ru);
      const T muB = integralGet<T>(bInt,lu,ld,rd,ru);
      depthSmooth[id] = (muA*depth[id] + muB)/n;
    }else{
      depthSmooth[id] = missingValue;
    }
}

void guidedFilter_out_gpu(uint8_t* haveData, double* depth, double* aInt,
    double* bInt, int32_t* Ns, float* depthSmooth, uint32_t B, uint32_t w,
    uint32_t h)
{
  dim3 threadsSq(16,16,1);
  dim3 blocksSq(w/16+(w%16>0?1:0), h/16+(h%16>0?1:0),1);

  if(B==1){
    guidedFilter_out_kernel<double,float,16,1><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }else if(B==2){
    guidedFilter_out_kernel<double,float,16,2><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }else if(B==3){
    guidedFilter_out_kernel<double,float,16,3><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }else if(B==4){
    guidedFilter_out_kernel<double,float,16,4><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }else if(B==5){
    guidedFilter_out_kernel<double,float,16,5><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }else if(B==6){
    guidedFilter_out_kernel<double,float,16,6><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }else if(B==7){
    guidedFilter_out_kernel<double,float,16,7><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }else if(B==8){
    guidedFilter_out_kernel<double,float,16,8><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }else if(B==9){
    guidedFilter_out_kernel<double,float,16,9><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }else if(B==10){
    guidedFilter_out_kernel<double,float,16,10><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0f/0.0f); 
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

void guidedFilter_out_gpu(uint8_t* haveData, double* depth, double* aInt,
    double* bInt, int32_t* Ns, double* depthSmooth, uint32_t B, uint32_t w,
    uint32_t h)
{
  dim3 threadsSq(16,16,1);
  dim3 blocksSq(w/16+(w%16>0?1:0), h/16+(h%16>0?1:0),1);

  if(B==1){
    guidedFilter_out_kernel<double,double,16,1><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }else if(B==2){
    guidedFilter_out_kernel<double,double,16,2><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }else if(B==3){
    guidedFilter_out_kernel<double,double,16,3><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }else if(B==4){
    guidedFilter_out_kernel<double,double,16,4><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }else if(B==5){
    guidedFilter_out_kernel<double,double,16,5><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }else if(B==6){
    guidedFilter_out_kernel<double,double,16,6><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }else if(B==7){
    guidedFilter_out_kernel<double,double,16,7><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }else if(B==8){
    guidedFilter_out_kernel<double,double,16,8><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }else if(B==9){
    guidedFilter_out_kernel<double,double,16,9><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }else if(B==10){
    guidedFilter_out_kernel<double,double,16,10><<<blocksSq,threadsSq>>>(haveData,depth,aInt,bInt,Ns,depthSmooth,w,h,0.0/0.0); 
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

 // ------------------------------- testing -------------------------------------

template<typename T, uint32_t BLK_SIZE, int32_t B>
__global__ void guidedFilter_ab_kernel(T* depth,uint8_t* haveData, uint8_t* haveDataAfter, T* a, T* b, int32_t*
    Ns, T* dSum, T* dSqSum, double eps, uint32_t w, uint32_t h)
{

  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if((idx<w)&&(idy<h) && (haveData[id]))
  {
    int32_t lu,ld,rd,ru;
    if(!integralCheck<B>(haveData,idy,idx,w,h,&lu,&ld,&rd,&ru)) // get the coordinates for integral img
    {
      haveDataAfter[id] = 0;
      return;
    }
    const T n = integralGet<int32_t>(Ns,lu,ld,rd,ru); 
    if(n < (2*B)*(2*B))
    {
      haveDataAfter[id] = 0;
      return;
    }
    const T muG = integralGet<T>(dSum,lu,ld,rd,ru);
    const T s = integralGet<T>(dSqSum,lu,ld,rd,ru);
    const T muSq = muG*muG;
    const T n1 = n-1.;

    const T z = depth[id];
    T epsT = eps;
    if(eps <= 0.)
    {
      epsT = 0.0012+0.0019*(z-0.4)*(z-0.4) + 0.0001/sqrt(z); // leaving out the noise by angle
      epsT *=5.;
      if(idx==300 && idy == 300)
        printf("eps=%f",epsT);
    }

    const T a_ = ((n*s-muSq)*n1)/((s*n-muSq+epsT*n1*n)*n);

    b[id] = muG*(1. - a_)/n;
    a[id] = a_;
    haveDataAfter[id] = 1;
  }
}

void guidedFilter_ab_gpu(double* depth, uint8_t* haveData, uint8_t*
    haveDataAfter, double* a, double* b, int32_t* Ns, double* dSum, double*
    dSqSum, double eps, uint32_t B, uint32_t w, uint32_t h)
{
  dim3 threadsSq(16,16,1);
  dim3 blocksSq(w/16+(w%16>0?1:0), h/16+(h%16>0?1:0),1);

  if(B==3){
    guidedFilter_ab_kernel<double,16,3><<<blocksSq,threadsSq>>>(depth,haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==5){
    guidedFilter_ab_kernel<double,16,5><<<blocksSq,threadsSq>>>(depth,haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==6){
    guidedFilter_ab_kernel<double,16,6><<<blocksSq,threadsSq>>>(depth,haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==7){
    guidedFilter_ab_kernel<double,16,7><<<blocksSq,threadsSq>>>(depth,haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==8){
    guidedFilter_ab_kernel<double,16,8><<<blocksSq,threadsSq>>>(depth,haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==9){
    guidedFilter_ab_kernel<double,16,9><<<blocksSq,threadsSq>>>(depth,haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }else if(B==10){
    guidedFilter_ab_kernel<double,16,10><<<blocksSq,threadsSq>>>(depth,haveData,haveDataAfter,a,b,Ns,dSum,dSqSum,eps,w,h); 
  }
  checkCudaErrors(cudaDeviceSynchronize());
};
