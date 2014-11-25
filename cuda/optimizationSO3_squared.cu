/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */


/*
 * compute the Jacobian of robust squared cost function 
 */
__global__ void robustSquaredAngleCostFctJacobian(float *J, float *x, 
    unsigned short *z, float *mu, float sigma_sq, int w, int h)//, float *dbg)
{
  const int DIM = 3;
  __shared__ float mui[DIM*6];
  // one J per column; BLOCK_SIZE columns; per column first 3 first col of J, 
  // second 3 columns second cols of J 
  __shared__ float J_shared[BLOCK_SIZE*3*3];
  
  const int tid = threadIdx.x + blockDim.x*threadIdx.y;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int id = idx + idy*w;
  // caching 
  if(tid < DIM*6) mui[tid] = mu[tid];
#pragma unroll
  for(int s=0; s<3*3; ++s) {
    J_shared[tid+BLOCK_SIZE*s] = 0.0f;
  }

  __syncthreads(); // make sure that ys have been cached
  if ((idx<w) && (idy<h))
  {
    float xi[3];
    xi[0] = x[id*X_STEP+X_OFFSET+0];
    xi[1] = x[id*X_STEP+X_OFFSET+1];
    xi[2] = x[id*X_STEP+X_OFFSET+2];
    unsigned short k = z[id]; // which MF axis does it belong to
    if (k<6)// && k!=4 && k!=5)
    {
      int j = k/2; // which of the rotation columns does this belong to
      float sign = (- float(k%2) +0.5f)*2.0f; // sign of the axis
      float xiTy = xi[0]*mui[k] + xi[1]*mui[k+6] 
        + xi[2]*mui[k+12];
      xiTy = max(-1.0f,min(1.0f,xiTy));
      float J_ =0.0f;
      if (xiTy > 1.0f-1e-10)
      {
        // limit according to mathematica
        J_ = -2.0f/sigma_sq; 
      }else{
        float err = acosf(xiTy);
        float err_sq = err*err;
        float a = sqrtf(1.0f - xiTy*xiTy);
        float b = (sigma_sq + err_sq);
        // obtained using Mathematica
        J_ = 2.0f*( (err*err_sq/(a*b*b)) - (err/(a*b)) );   
        // TODO could be simplified: see writeup!
      }
      //dbg[id] = J_;
      J_shared[tid+(j*3+0)*BLOCK_SIZE] = sign*J_*xi[0];   
      J_shared[tid+(j*3+1)*BLOCK_SIZE] = sign*J_*xi[1];   
      J_shared[tid+(j*3+2)*BLOCK_SIZE] = sign*J_*xi[2];   
    }else{
      //dbg[id] = 9999.0f;
    }
  }
  //reduction.....
    __syncthreads(); //sync the threads
#pragma unroll
    for(int s=(BLOCK_SIZE)/2; s>0; s>>=1) {
      if(tid < s)
#pragma unroll
        for( int k=0; k<3*3; ++k) {
          int tidk = k*BLOCK_SIZE+tid;
          J_shared[tidk] += J_shared[tidk + s];
        }
      __syncthreads();
    }

#pragma unroll
    for( int k=0; k<3*3; ++k) {
      if(tid==k  && J_shared[k*BLOCK_SIZE]!=0 ) {
        atomicAdd(&J[k],J_shared[k*BLOCK_SIZE]);
      }
    }

//  //reduction.....
//#pragma unroll
//  for( int k=0; k<3*3; ++k) {
//    int tidk = k*BLOCK_SIZE+tid;
//    __syncthreads(); //sync the threads
//#pragma unroll
//    for(int s=(BLOCK_SIZE)/2; s>0; s>>=1) {
//      if(tid < s)
//        J_shared[tidk] += J_shared[tidk + s];
//      __syncthreads();
//    }
//
//    if(tid==0  && J_shared[k*BLOCK_SIZE]!=0 ) {
//      atomicAdd(&J[k],J_shared[k*BLOCK_SIZE]);
//    }
//  }
}

/* 
 * compute normal assignments as well as the costfunction value under that
 * assignment. Normal assignments are computed according based on nearest 
 * distance in the arclength sense.
 */
__global__ void squaredAngleCostFctAssignment(float *cost, uint32_t* N, 
    float *x, unsigned short *z, float* errs, float *mu, float sigma_sq, 
    int w, int h)
{
  const int DIM = 3;
  //__shared__ float xi[BLOCK_SIZE*3];
  __shared__ float mui[DIM*6];
  __shared__ float rho[BLOCK_SIZE];
  __shared__ uint32_t Ni[BLOCK_SIZE];
  
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int id = idx + idy*w;
  // caching 
  if(tid < DIM*6) mui[tid] = mu[tid];
  rho[tid] = 0.0f;
  Ni[tid] = 0;

  __syncthreads(); // make sure that ys have been cached
  if ((idx<w) && (idy<h))
  {
    float xi[3];
    xi[0] = x[id*X_STEP+X_OFFSET+0];
    xi[1] = x[id*X_STEP+X_OFFSET+1];
    xi[2] = x[id*X_STEP+X_OFFSET+2];

    float err_min = 9999999.0f;
    unsigned short k_min = 6;
    if((xi[0]!=xi[0] || xi[1]!=xi[1] || xi[2]!=xi[2]) 
        || xi[0]*xi[0]+xi[1]*xi[1]+xi[2]*xi[2] < 0.9f )
    {
      // if nan
      k_min = 6;
      err_min = .1f; 
      //if(X_STEP == 8) x[id*X_STEP+4] = 6.0f;
    }else{
#pragma unroll
      for (unsigned short k=0; k<6; ++k)
      {
        float xiTy = xi[0]*mui[k] + xi[1]*mui[k+6] + xi[2]*mui[k+12];
        float err = acosf(max(-1.0f,min(1.0f,xiTy)));
        if(err_min > err)
        {
          err_min = err;
          k_min = k;
        }
      }
      rho[tid] = (err_min*err_min)/sigma_sq;
      Ni[tid] = 1;
    }
    z[id] = k_min;
    errs[id] = err_min;
    if(X_STEP == 8) 
    {
      x[id*X_STEP+X_OFFSET+4] = c_rgbForMFaxes[k_min];//float(k_min);
      x[id*X_STEP+X_OFFSET+5] = float(k_min);//xi[0]; //float(k_min);
      x[id*X_STEP+X_OFFSET+6] = err_min; //rgb;//xi[1]; //err_min;
//      x[id*X_STEP+X_OFFSET+7] = 0.0f;//err_min; //err_min;
    }
  }
  //reduction.....
  // TODO: make it faster!
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>0; s>>=1) {
    if(tid < s)
    {
      rho[tid] += rho[tid + s];
      Ni[tid] += Ni[tid + s];
    }
    __syncthreads();
  }

  if(tid==0  && rho[0]!=0.0f) {
    atomicAdd(&cost[0],rho[0]);
  }
  if(tid==1  && Ni[0]!=0 ) {
    atomicAdd(N,Ni[0]);
  }

//  __syncthreads(); //sync the threads
//#pragma unroll
//  for(int s=(BLOCK_SIZE)/2; s>0; s>>=1) {
//    if(tid < s)
//      Ni[tid] += Ni[tid + s];
//    __syncthreads();
//  }
//
//  if(tid==0  && Ni[0]!=0 ) {
//    atomicAdd(N,Ni[0]);
//  }
}
