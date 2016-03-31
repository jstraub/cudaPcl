namespace cudaPcl {

template<typename T>
NormalExtractSimpleGpu<T>::NormalExtractSimpleGpu(T f_depth, uint32_t w, uint32_t h, bool compress)
  : invF_(1./f_depth), w_(w), h_(h), wProc_(w+w%16), hProc_(h+h%64), cudaReady_(false), nCachedPc_(false),
   nCachedImg_(false), pcComputed_(false), nCachedComp_(false),
   compress_(compress), 
   d_z(h,w),
   d_zu(h,w),
   d_nImg_(h*w,3),
   d_haveData_(h,w), d_normalsComp_(h*w,3), d_compInd_(w*h,1)
{
  cout<<"calling prepareCUDA internal processing size " 
    << wProc_ << "x" << hProc_ <<endl;
  cout<<cudaReady_<<endl;
  prepareCUDA();
  cout<<"prepareCUDA done"<<endl;
};

template<typename T>
NormalExtractSimpleGpu<T>::~NormalExtractSimpleGpu()
{
  if(!cudaReady_) return;
  checkCudaErrors(cudaFree(d_depth));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
//  checkCudaErrors(cudaFree(d_z));
  checkCudaErrors(cudaFree(d_nPcl));
  checkCudaErrors(cudaFree(d_xyz));
  checkCudaErrors(cudaFree(d_xu));
  checkCudaErrors(cudaFree(d_yu));
//  checkCudaErrors(cudaFree(d_zu));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
  checkCudaErrors(cudaFree(d_zv));
  checkCudaErrors(cudaFree(a));
  checkCudaErrors(cudaFree(b));
  checkCudaErrors(cudaFree(c));
//#ifdef WEIGHTED
  if(d_w != NULL) checkCudaErrors(cudaFree(d_w));
//#endif
//  free(h_nPcl);
};


template<typename T>
void NormalExtractSimpleGpu<T>::compute(const uint16_t* data, uint32_t w, uint32_t h)
{
  assert(w_ == w);
  assert(h_ == h);

  checkCudaErrors(cudaMemcpy(d_depth, data, w_ * h_ * sizeof(uint16_t),
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());
  computeGpu(d_depth,w,h);

};

template<typename T>
void NormalExtractSimpleGpu<T>::computeGpu(uint16_t* d_depth)
{
  cout<<"NormalExtractSimpleGpu<T>::computeGpu uint16_t "<<w_<<" "<<h_<<endl;
  // convert depth image to xyz point cloud
  depth2xyzGPU(d_depth, d_x, d_y, d_z.data(), invF_, w_, h_, NULL);

  haveDataGpu(d_x,d_haveData_.data(),wProc_*hProc_,1);

  computeDerivatives(wProc_,hProc_);

  // obtain the normals using mainly cross product on the derivatives
  derivatives2normalsGPU(
      d_x,d_y,d_z.data(),
      d_xu,d_yu,d_zu.data(),
      d_xv,d_yv,d_zv,
      d_nImg_.data(),d_haveData_.data(),w_,h_);

  nCachedPc_ = false;
  nCachedImg_ = false;
  pcComputed_ = false;
  nCachedComp_ = false;
  if(compress_)
  {
    compressNormals(w_,h_);
  }
}

template<typename T>
void NormalExtractSimpleGpu<T>::computeGpu(T* depth)
{
//  cv::Mat Id(h,w,CV_32FC1);
//  checkCudaErrors(cudaMemcpy(Id.data, depth, w*h*sizeof(float),
//                cudaMemcpyDeviceToHost));
//  cv::Mat Irgb = OpenniVisualizer::colorizeDepth(Id,0.3,4.0);
//  cv::imshow("Idrgb",Irgb);
//  cv::waitKey(1);

  haveDataGpu(depth,d_haveData_.data(),wProc_*hProc_,1);
//  cv::Mat Z(h,w,CV_8UC1);
//  d_haveData_.get((uint8_t*)Z.data,h,w);
//  cv::imshow("Z",Z*255);
//  cv::waitKey(1);

//  if(true)
//  {
    cout<<"NormalExtractSimpleGpu<T>::computeGpu "<<w_<<" "<<h_<<endl;
    // convert depth image to xyz point cloud
    depth2xyzFloatGPU(depth, d_x, d_y, d_z.data(), invF_, w_, h_, NULL);

    computeDerivatives(wProc_,hProc_);

    // obtain the normals using mainly cross product on the derivatives
    derivatives2normalsGPU(
        d_x,d_y,d_z.data(),
        d_xu,d_yu,d_zu.data(),
        d_xv,d_yv,d_zv,
        d_nImg_.data(),d_haveData_.data(),w_,h_);

//  }else{
//    // potentially faster but sth is wrong with the derivatives2normalsGPU
//    setConvolutionKernel(h_sobel_dif);
//    convolutionRowsGPU(c,depth,w,h);
//    setConvolutionKernel(h_sobel_sum);
//    convolutionColumnsGPU(d_zu.data(),c,w,h);
//    convolutionRowsGPU(b,depth,w,h);
//    setConvolutionKernel(h_sobel_dif);
//    convolutionColumnsGPU(d_zv,b,w,h);
//
//    derivatives2normalsGPU(depth, d_zu.data(), d_zv,
//        d_nImg_.data(),d_haveData_.data(),invF_,w_,h_);
//
//  }

  nCachedPc_ = false;
  nCachedImg_ = false;
  pcComputed_ = false;
  nCachedComp_ = false;
  if(compress_)
  {
    compressNormals(w_,h_);
  }
};

template<typename T>
void NormalExtractSimpleGpu<T>::compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, T radius)
{
  // extract surface normals
  pcl::PointCloud<pcl::Normal> normals;
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(pc);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);
  ne.setRadiusSearch (radius);
  ne.compute (normals);
  setNormalsCpu(normals);
};

template<typename T>
void NormalExtractSimpleGpu<T>::setNormalsCpu(const pcl::PointCloud<pcl::Normal>& normals)
{
  assert(normals.width==w_);
  assert(normals.height==h_);
  // use pitchs to copy correctly: pcl:Normals is float[4] where the
  // first 3 are normals and the 4th is the curvature
  d_nImg_.set(normals.getMatrixXfMap().data(),normals.height,normals.width,3,4);
  haveDataGpu(d_nImg_.data(),d_haveData_.data(),normals.height*normals.width ,3);

  nCachedPc_ = false;
  nCachedImg_ = false;
  pcComputed_ = false;
  nCachedComp_ = false;
  if(compress_)
  {
    compressNormals(normals.width,normals.height);
  }
};

template<typename T>
void NormalExtractSimpleGpu<T>::setNormalsCpu(T* n, uint32_t w, uint32_t h)
{
  assert(w==w_);
  assert(h==h_);
  d_nImg_.set(n,w*h,3);
  haveDataGpu(d_nImg_.data(),d_haveData_.data(),w*h,3);

  nCachedPc_ = false;
  nCachedImg_ = false;
  pcComputed_ = false;
  nCachedComp_ = false;
  if(compress_)
  {
    compressNormals(w,h);
  }
};

template<typename T>
void NormalExtractSimpleGpu<T>::compressNormals(uint32_t w, uint32_t h)
{
    cv::Mat haveDat = haveData();
    indMap_.clear();
    indMap_.reserve(w*h);
    for(uint32_t i=0; i<w*h; ++i) 
      if(haveDat.at<uint8_t>(i) ==1)
        indMap_.push_back(i);
    nComp_ = indMap_.size();
    if (nComp_ > 0)
    {
      d_normalsComp_.resize(nComp_,3);
      // just shuffle the first 640 entries -> to get random init for the algorithms
      std::random_shuffle(indMap_.begin(), indMap_.begin() + std::min(640,nComp_));
      jsc::GpuMatrix<uint32_t> d_indMap_(indMap_); // copy to GPU
      copyShuffleGPU(d_nImg_.data(), d_normalsComp_.data(), d_indMap_.data(), nComp_, 3);
    }
#ifndef NDEBUG
    cout<<"compression of "<<T(w*h-nComp_)/T(w*h)<<"% to "
      <<nComp_<<" datapoints"<<endl;
#endif
};



template<typename T>
void NormalExtractSimpleGpu<T>::uncompressCpu(const uint32_t* in,
    uint32_t Nin, uint32_t* out, uint32_t Nout)
{
  if(indMap_.size() > 0)
  {
    for(uint32_t i=0; i<Nout; ++i) out[i] = INT_MAX;
    for(uint32_t i=0; i<Nin; ++i) out[indMap_[i]] = in[i];
  }
};

template<typename T>
void NormalExtractSimpleGpu<T>::computeDerivatives(uint32_t w,uint32_t h)
{
  setConvolutionKernel(h_sobel_dif);
  convolutionRowsGPU(a,d_x,w,h);
  convolutionRowsGPU(b,d_y,w,h);
  convolutionRowsGPU(c,d_z.data(),w,h);
  setConvolutionKernel(h_sobel_sum);
  convolutionColumnsGPU(d_xu,a,w,h);
  convolutionColumnsGPU(d_yu,b,w,h);
  convolutionColumnsGPU(d_zu.data(),c,w,h);
  convolutionRowsGPU(a,d_x,w,h);
  convolutionRowsGPU(b,d_y,w,h);
  convolutionRowsGPU(c,d_z.data(),w,h);
  setConvolutionKernel(h_sobel_dif);
  convolutionColumnsGPU(d_xv,a,w,h);
  convolutionColumnsGPU(d_yv,b,w,h);
  convolutionColumnsGPU(d_zv,c,w,h);
}


template<typename T>
void NormalExtractSimpleGpu<T>::computeAreaWeights()
{
  if(d_w == NULL)
    checkCudaErrors(cudaMalloc((void **)&d_w, w_ * h_ * sizeof(T)));
//#ifdef WEIGHTED
//  weightsFromCovGPU(d_z, d_w, 30.0f, invF_, w_,h_);
  weightsFromAreaGPU(d_z.data(), d_w, w_,h_);
//#endif
}


template<typename T>
void NormalExtractSimpleGpu<T>::prepareCUDA()
{
  // CUDA preparations
  cout << "Allocating and initializing CUDA arrays... "<< w_<<" "<<h_<<endl;
  cout << "Internalt size ... "<< wProc_<<" "<<hProc_<<endl;
  checkCudaErrors(cudaMalloc((void **)&d_depth, w_ * h_ * sizeof(uint16_t)));
  checkCudaErrors(cudaMalloc((void **)&d_x, wProc_ * hProc_ * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_y, wProc_ * hProc_ * sizeof(T)));
  d_z.resize(hProc_,wProc_);
  d_zu.resize(hProc_,wProc_);

  checkCudaErrors(cudaMalloc((void **)&d_nPcl, w_ * h_ *8* sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_xyz,  w_ * h_ *4* sizeof(T)));

  checkCudaErrors(cudaMalloc((void **)&d_xu, wProc_ * hProc_ * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_yu, wProc_ * hProc_ * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_xv, wProc_ * hProc_ * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_yv, wProc_ * hProc_ * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&d_zv, wProc_ * hProc_ * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&a,    wProc_ * hProc_ * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&b,    wProc_ * hProc_ * sizeof(T)));
  checkCudaErrors(cudaMalloc((void **)&c,    wProc_ * hProc_ * sizeof(T)));

   d_haveData_.resize(hProc_,wProc_);
   d_haveData_.setOnes();
//#ifdef WEIGHTED
//#else
  d_w = NULL;
//#endif

  h_sobel_dif[0] = 1;
  h_sobel_dif[1] = 0;
  h_sobel_dif[2] = -1;

  h_sobel_sum[0] = 1;
  h_sobel_sum[1] = 2;
  h_sobel_sum[2] = 1;

  n_ = pcl::PointCloud<pcl::PointXYZRGB>(w_,h_);
  n_cp_ = pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr(&n_);
  Map<MatrixXf, Aligned, OuterStride<> > nMat =
    n_.getMatrixXfMap(8,8,0);
  h_nPcl = nMat.data();//(T *)malloc(w_ *h_ *3* sizeof(T));

  cout<<nMat.rows()<< " "<<nMat.cols()<<" "<<8<<endl;
  cout<<w_<<" "<<h_<<endl;

  pc_ = pcl::PointCloud<pcl::PointXYZ>(w_,h_);
  pc_cp_ = pcl::PointCloud<pcl::PointXYZ>::ConstPtr(&pc_);
  Map<MatrixXf, Aligned, OuterStride<> > pcMat =
    pc_.getMatrixXfMap(4,4,0);
  h_xyz = pcMat.data();//(T *)malloc(w_ *h_ *3* sizeof(T));

  cudaReady_ = true;
}


template<typename T>
float* NormalExtractSimpleGpu<T>::d_normalsPcl(){
  if(!pcComputed_ && w_ > 0 && h_ > 0)
  {
    xyzImg2PointCloudXYZRGB(d_nImg_.data(), d_nPcl,w_,h_);
    pcComputed_ = true;
  }
  return d_nPcl;
}; 

template<typename T>
pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr NormalExtractSimpleGpu<T>::normalsPc()
{
  if(!nCachedPc_)
  {
    if(!pcComputed_ && w_ > 0 && h_ > 0)
    {
      xyzImg2PointCloudXYZRGB(d_nImg_.data(), d_nPcl,w_,h_);
      pcComputed_ = true;
    }
//    cout<<"w="<<w_<<" h="<<h_<<" "<<X_STEP<<" "<<sizeof(T)<<endl;
    checkCudaErrors(cudaMemcpy(h_nPcl, d_nPcl, w_*h_ *sizeof(float)* 8,
          cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    nCachedPc_ = true;
    std::cout << "cached normals" << std::endl;
  }
  return n_cp_;
};


template<typename T>
cv::Mat NormalExtractSimpleGpu<T>::haveData()
{
//  if(!nCachedImg_)
//  {

//    cv::Mat haveData = cv::Mat::ones(h_,w_,CV_8U)*2; // not necessary
    haveData_ = cv::Mat (h_,w_,CV_8U);
    checkCudaErrors(cudaMemcpy(haveData_.data, d_haveData_.data(),
          w_*h_ *sizeof(uint8_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
//    nCachedImg_ = true;
//  }
  return haveData_;
};

template<typename T>
cv::Mat NormalExtractSimpleGpu<T>::compInd()
{
//  if(!nCachedImg_)
//  {
    compInd_ = cv::Mat (h_,w_,CV_32S); // there is no CV_32U - but an just type cast between those
    d_compInd_.get((uint32_t*)compInd.data,nComp_,1);
//    nCachedImg_ = true;
//  }
  return compInd_;
};

template<>
cv::Mat NormalExtractSimpleGpu<float>::normalsImg()
{
  if(!nCachedImg_)
  {
//    cout<<"NormalExtractSimpleGpu::normalsImg size: "<<w_<<" "<<h_<<endl;
    nImg_ = cv::Mat(h_,w_,CV_32FC3, 0.f);
    checkCudaErrors(cudaMemcpy(nImg_.data, d_nImg_.data(), w_*h_
          *sizeof(float)*3, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    nCachedImg_ = true;
  }
  return nImg_;
};

template<>
cv::Mat NormalExtractSimpleGpu<double>::normalsImg()
{
  if(!nCachedImg_)
  {
    cout<<"NormalExtractSimpleGpu::normalsImg size: "<<w_<<" "<<h_<<endl;
    nImg_ = cv::Mat(h_,w_,CV_64FC3);
    checkCudaErrors(cudaMemcpy(nImg_.data, d_nImg_.data(), w_*h_
          *sizeof(double)*3, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    nCachedImg_ = true;
  }
  return nImg_;
};


template<>
cv::Mat NormalExtractSimpleGpu<float>::normalsComp(int32_t& nComp)
{
  if(!nCachedComp_)
  {
//    cout<<nComp_<<endl;
    normalsComp_ = cv::Mat(nComp_,1,CV_32FC3);
//    cout<<normalsComp_.total()<<" "
//      <<normalsComp_.total()*normalsComp_.elemSize()<<" "
//      <<normalsComp_.elemSize()<<endl;
    d_normalsComp_.get((float*)normalsComp_.data,nComp_,3);
    nCachedComp_ = true;
  }
  nComp = nComp_;
  return normalsComp_;
}

template<>
cv::Mat NormalExtractSimpleGpu<double>::normalsComp(int32_t& nComp)
{
  if(!nCachedComp_)
  {
    normalsComp_ = cv::Mat(nComp_,1,CV_64FC3);
    d_normalsComp_.get((double*)normalsComp_.data,nComp_,3);
    nCachedComp_ = true;
  }
  nComp = nComp_;
  return normalsComp_;
}

//template<typename T>
//pcl::PointCloud<pcl::PointXYZRGB> NormalExtractGpu<T>::normals()
//{
//  if(!nCachedPc_)
//  {
//    checkCudaErrors(cudaMemcpy(h_nPcl, d_nImg_, w_*h_* X_STEP *sizeof(T),
//          cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaDeviceSynchronize());
//    nCachedPc_ = true;
//  }
//  return n_;
//};
//template<typename T>
//pcl::PointCloud<pcl::PointXYZ>::ConstPtr NormalExtractSimpleGpu<T>::pointCloud()
//{
////  if(!pcCached_)
////  {
////    checkCudaErrors(cudaMemcpy(h_xyz, d_xyz, w_*h_*4 *sizeof(T),
////          cudaMemcpyDeviceToHost));
////    checkCudaErrors(cudaDeviceSynchronize());
////    pcCached_ = true;
////  }
//  return pc_cp_;
//};
}
