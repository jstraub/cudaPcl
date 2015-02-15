/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <stdint.h>
#include <cudaPcl/helper_cuda.h> 

template<typename T>
inline __device__
T gaussian1d_gpu_reg(T x, T variance, T sqrt_pi_variance)
{
  T gaussian1d = -(x*x)/(2*variance);
  gaussian1d = __expf(gaussian1d);
  gaussian1d /= sqrt_pi_variance;
  return gaussian1d;
}

__host__ __device__
float gaussian2d(float x, float y, float sigma)
{
  float variance = pow(sigma,2);
  float exponent = -(pow(x,2) + pow(y,2))/(2*variance);
  return expf(exponent) / (2 * M_PI * variance);
}


float* generateGaussianKernel(int radius, float sigma)
{
  int area = (2*radius+1)*(2*radius+1);
  float* res = new float[area];

  for(int x = -radius; x <= radius; x++)
    for(int y = -radius; y <= radius; y++)
    {
      //Co_to_idx inspired
      int position = (x+radius)*(radius*2+1) + y+radius;
      res[position] = gaussian2d(x,y,sigma);
    }
  return res;
}


template<typename T>
__global__
void bilateralFilter_kernel(T *in, T* out, int w, int h, int R, T* kernel, T variance, T sqrt_pi_variance, T sig_xy)
{
  const int idx = blockIdx.x*blockDim.x + threadIdx.x;
  const int idy = blockIdx.y*blockDim.y + threadIdx.y;

  if(idx >= w || idy >= h) return;


  int id = idy*w + idx; //co_to_idx(make_uint2(idx_x,idx_y),dims);
  T I = in[id];

  T res = 0.;
  T normalization = 0.;
  T weight ;
#pragma unroll
  for(int i = -R; i <= R; i++) {
#pragma unroll
    for(int j = -R; j <= R; j++) {

      int x_sample = idx+i;
      int y_sample = idy+j;

      //mirror edges
      if( x_sample < 0) x_sample = -x_sample;
      if( y_sample < 0) y_sample = -y_sample;
      if( x_sample > w - 1) x_sample = w - 1 - i;
      if( y_sample > h - 1) y_sample = h - 1 - j;


      int tempPos = y_sample*w + x_sample;
      T tmpI = in[tempPos];

//      T gauss_spatial = kernel[(i+R) + (j+R)*(2*R+1)]; //co_to_idx(make_uint2(i+radius,j+radius),make_uint2(radius*2+1,radius*2+1))];
      T gauss_spatial =  gaussian2d(i,j,sig_xy);

      weight = gauss_spatial * gaussian1d_gpu_reg<T>((I - tmpI),variance,sqrt_pi_variance);

      normalization += weight;
      res = res + (tmpI * weight);
    }
  }
  out[id] = res/normalization;
}



extern void bilateralFilterGPU(float *in, float* out, int w, int h, 
  uint32_t radius, float sigma_spatial, float sigma_I)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  float* kernel = generateGaussianKernel(radius,sigma_spatial);
  float* d_Kernel;
  //Set up kernel
  checkCudaErrors(cudaMalloc( (void**) &d_Kernel, (2*radius+1)*(2*radius+1) * sizeof(float)));
  checkCudaErrors(cudaMemcpy( d_Kernel, kernel, (2*radius+1)*(2*radius+1)* sizeof(float), cudaMemcpyHostToDevice));

  float variance_I = pow(sigma_I,2);
  float sqrt_pi_variance = sqrt(2.*M_PI*variance_I);

  if(radius == 3){
    bilateralFilter_kernel<float><<<blocks, threads>>>(in,
        out,w,h,3,kernel,variance_I,sqrt_pi_variance,sigma_spatial);
  }else if(radius == 6){
    bilateralFilter_kernel<float><<<blocks, threads>>>(in,
        out,w,h,6,kernel,variance_I,sqrt_pi_variance,sigma_spatial);
  }else if(radius == 9){
    bilateralFilter_kernel<float><<<blocks, threads>>>(in,
        out,w,h,9,kernel,variance_I,sqrt_pi_variance,sigma_spatial);
  }
  cudaFree(d_Kernel);
  delete kernel;

  getLastCudaError("bilateralFilterGPU() execution failed\n");
  
};
