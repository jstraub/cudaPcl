/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#include <normalExtractCUDA.hpp>

template<typename T>
NormalExtractGpu<T>::NormalExtractGpu(T f_depth)
  : invF_(1./f_depth), cudaReady_(false), nCached_(false), pcCached_(false)
{
  cout<<"constructor without cuda init"<<endl;
};

template<typename T>
NormalExtractGpu<T>::NormalExtractGpu(T f_depth, uint32_t w, uint32_t h)
  : invF_(1./f_depth), w_(w), h_(h), cudaReady_(false), nCached_(false), pcCached_(false)
{
  cout<<"calling prepareCUDA"<<endl;
  cout<<cudaReady_<<endl;
  prepareCUDA(w_,h_);
  cout<<"prepareCUDA done"<<endl;
};

template<typename T>
NormalExtractGpu<T>::~NormalExtractGpu()
{
  if(!cudaReady_) return;
  checkCudaErrors(cudaFree(d_depth));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_z));
  checkCudaErrors(cudaFree(d_n));
  checkCudaErrors(cudaFree(d_xyz));
  checkCudaErrors(cudaFree(d_xu));
  checkCudaErrors(cudaFree(d_yu));
  checkCudaErrors(cudaFree(d_zu));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
  checkCudaErrors(cudaFree(d_zv));
  checkCudaErrors(cudaFree(a));
  checkCudaErrors(cudaFree(b));
  checkCudaErrors(cudaFree(c));
//#ifdef WEIGHTED
  if(d_w) checkCudaErrors(cudaFree(d_w));
//#endif
//  free(h_n);
  free(h_dbg);
};

template<typename T>
pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr NormalExtractGpu<T>::normals()
{
  //TODO bad error here
  if(!nCached_)
  {
    cout<<"w="<<w_<<" h="<<h_<<" "<<X_STEP<<" "<<sizeof(T)<<endl;
          cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    nCached_ = true;
  }
  return n_cp_;
};
//template<typename T>
//pcl::PointCloud<pcl::PointXYZRGB> NormalExtractGpu<T>::normals()
//{
//  if(!nCached_)
//  {
//    checkCudaErrors(cudaMemcpy(h_n, d_n, w_*h_* X_STEP *sizeof(T), 
//          cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaDeviceSynchronize());
//    nCached_ = true;
//  }
//  return n_;
//};
template<typename T>
pcl::PointCloud<pcl::PointXYZ>::ConstPtr NormalExtractGpu<T>::pointCloud()
{
//  if(!pcCached_)
//  {
//    checkCudaErrors(cudaMemcpy(h_xyz, d_xyz, w_*h_*4 *sizeof(T), 
//          cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaDeviceSynchronize());
//    pcCached_ = true;
//  }
  return pc_cp_;
};
//template<typename T>
//pcl::PointCloud<pcl::PointXYZ> NormalExtractGpu<T>::pointCloud()
//{
//  if(!pcCached_)
//  {
//    checkCudaErrors(cudaMemcpy(h_xyz, d_xyz, w_*h_*4 *sizeof(T), 
//          cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaDeviceSynchronize());
//    pcCached_ = true;
//  }
//  return pc_;
//};

template<typename T>
void NormalExtractGpu<T>::compute(const uint16_t* data, uint32_t w, uint32_t h)
{
  w_ = w; h_ = h;
  if (cudaReady_)
  {
    cout<<"cuda already initialized"<<endl;
  }else{
    prepareCUDA(w_,h_);
  }

  checkCudaErrors(cudaMemcpy(d_depth, data, w_ * h_ * sizeof(uint16_t),
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());

#ifdef SMOOTH_DEPTH
  depth2smoothXYZ(invF_,w_,h_); 
#else
  depth2xyzGPU(d_depth,d_x,d_y,d_z,invF_,w_,h_,d_xyz); 
#endif
  // obtain derivatives using sobel 
  computeDerivatives(w_,h_);
#ifndef SMOOTH_DEPTH
  // now smooth the derivatives
  smoothDerivatives(2,w_,h_);
#endif
  // obtain the normals using mainly cross product on the derivatives
  derivatives2normalsGPU(
      d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,w_,h_);
  nCached_ = false;
  pcCached_ = false;

};


template<typename T>
void NormalExtractGpu<T>::computeDerivatives(uint32_t w,uint32_t h)
{
  setConvolutionKernel_small(h_sobel_dif);
  convolutionRowsGPU_small(a,d_x,w,h);
  convolutionRowsGPU_small(b,d_y,w,h);
  convolutionRowsGPU_small(c,d_z,w,h);
  setConvolutionKernel_small(h_sobel_sum);
  convolutionColumnsGPU_small(d_xu,a,w,h);
  convolutionColumnsGPU_small(d_yu,b,w,h);
  convolutionColumnsGPU_small(d_zu,c,w,h);
  convolutionRowsGPU_small(a,d_x,w,h);
  convolutionRowsGPU_small(b,d_y,w,h);
  convolutionRowsGPU_small(c,d_z,w,h);
  setConvolutionKernel_small(h_sobel_dif);
  convolutionColumnsGPU_small(d_xv,a,w,h);
  convolutionColumnsGPU_small(d_yv,b,w,h);
  convolutionColumnsGPU_small(d_zv,c,w,h);
}

template<typename T>
void NormalExtractGpu<T>::smoothDerivatives(uint32_t iterations, uint32_t w,uint32_t h)
{
  setConvolutionKernel(h_kernel_avg);
  for(uint32_t i=0; i<iterations; ++i)
  {
    convolutionRowsGPU(a,d_xu,w,h);
    convolutionRowsGPU(b,d_yu,w,h);
    convolutionRowsGPU(c,d_zu,w,h);
    convolutionColumnsGPU(d_xu,a,w,h);
    convolutionColumnsGPU(d_yu,b,w,h);
    convolutionColumnsGPU(d_zu,c,w,h);
    convolutionRowsGPU(a,d_xv,w,h);
    convolutionRowsGPU(b,d_yv,w,h);
    convolutionRowsGPU(c,d_zv,w,h);
    convolutionColumnsGPU(d_xv,a,w,h);
    convolutionColumnsGPU(d_yv,b,w,h);
    convolutionColumnsGPU(d_zv,c,w,h);
  }
}

template<typename T>
void NormalExtractGpu<T>::depth2smoothXYZ(T invF, uint32_t w,uint32_t h)
{
  depth2floatGPU(d_depth,a,w,h);

//  for(uint32_t i=0; i<3; ++i)
//  {
//    depthFilterGPU(a,w,h);
//  }
  //TODO compare:
  // now smooth the derivatives

#ifdef BILATERAL                                                                
  cout<<"bilateral with "<<w<<" "<<h<<endl;                                     
  bilateralFilterGPU(a,b,w,h,6,20.0,0.05);                                      
  // convert depth into x,y,z coordinates                                       
  depth2xyzFloatGPU(b,d_x,d_y,d_z,invF,w,h,d_xyz);                              
#else 
  setConvolutionKernel(h_kernel_avg);
  for(uint32_t i=0; i<3; ++i)
  {
    convolutionRowsGPU(b,a,w,h);
    convolutionColumnsGPU(a,b,w,h);
  }
  // convert depth into x,y,z coordinates
  depth2xyzFloatGPU(a,d_x,d_y,d_z,invF,w,h,d_xyz); 
#endif                                                                          

}


template<typename T>
void NormalExtractGpu<T>::computeAreaWeights()
{
  if(d_w == NULL) 
    checkCudaErrors(cudaMalloc((void **)&d_w, w_ * h_ * sizeof(T)));
//#ifdef WEIGHTED    
//  weightsFromCovGPU(d_z, d_w, 30.0f, invF_, w_,h_);
  weightsFromAreaGPU(d_z, d_w, w_,h_);
//#endif
}

template<typename T>
void NormalExtractGpu<T>::prepareCUDA(uint32_t w,uint32_t h)
{
  // CUDA preparations
  cout << "Allocating and initializing CUDA arrays..."<<endl;
  checkCudaErrors(cudaMalloc((void **)&d_depth, w * h * sizeof(uint16_t)));
  checkCudaErrors(cudaMalloc((void **)&d_x, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_y, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_z, w * h * sizeof(T)));

  checkCudaErrors(cudaMalloc((void **)&d_n, w * h * X_STEP* sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_xyz, w * h * 4* sizeof(T)));

  checkCudaErrors(cudaMalloc((void **)&d_xu, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_yu, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_zu, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_xv, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_yv, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_zv, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&a, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&b, w * h * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&c, w * h * sizeof(T)));
//#ifdef WEIGHTED
//#else
  d_w = NULL;
//#endif
  cout<<"cuda allocations done "<<d_n<<endl;

  h_sobel_dif[0] = 1;
  h_sobel_dif[1] = 0;
  h_sobel_dif[2] = -1;

  h_sobel_sum[0] = 1;
  h_sobel_sum[1] = 2;
  h_sobel_sum[2] = 1;

  // sig =1.0
  // x=np.arange(7) -3.0
  // 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(x*x/sig**2))
  // 0.00443185,  0.05399097,  0.24197072,  0.39894228,  0.24197072,
  // 0.05399097,  0.00443185
  // sig = 2.0
  // 0.0647588 ,  0.12098536,  0.17603266,  0.19947114,  0.17603266,
  // 0.12098536,  0.0647588 
  /*
     h_kernel_avg[0] = 0.00443185;
     h_kernel_avg[1] = 0.05399097;
     h_kernel_avg[2] = 0.24197072;
     h_kernel_avg[3] = 0.39894228;
     h_kernel_avg[4] = 0.24197072;
     h_kernel_avg[5] = 0.05399097;
     h_kernel_avg[6] = 0.00443185;
     */

  h_kernel_avg[0] = 0.0647588;
  h_kernel_avg[1] = 0.12098536;
  h_kernel_avg[2] = 0.17603266;
  h_kernel_avg[3] = 0.19947114;
  h_kernel_avg[4] = 0.17603266;
  h_kernel_avg[5] = 0.12098536;
  h_kernel_avg[6] = 0.0647588;

  n_ = pcl::PointCloud<pcl::PointXYZRGB>(w,h);
  n_cp_ = pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr(&n_);
  Map<MatrixXf, Aligned, OuterStride<> > nMat = 
    n_.getMatrixXfMap(X_STEP,X_STEP,0);
  h_n = nMat.data();//(T *)malloc(w *h *3* sizeof(T));

  cout<<nMat.rows()<< " "<<nMat.cols()<<" "<<X_STEP<<endl;
  cout<<w<<" "<<h<<endl;

  pc_ = pcl::PointCloud<pcl::PointXYZ>(w,h);
  pc_cp_ = pcl::PointCloud<pcl::PointXYZ>::ConstPtr(&pc_);
  Map<MatrixXf, Aligned, OuterStride<> > pcMat = 
    pc_.getMatrixXfMap(X_STEP,X_STEP,0);
  h_xyz = pcMat.data();//(T *)malloc(w *h *3* sizeof(T));

  h_dbg = (T *)malloc(w *h * sizeof(T));

  cudaReady_ = true;
}




//template class NormalExtractGpu<double>;
template class NormalExtractGpu<float>;
