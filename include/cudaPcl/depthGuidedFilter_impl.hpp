namespace cudaPcl {

template<typename T>
DepthGuidedFilterGpu<T>::DepthGuidedFilterGpu(uint32_t w, uint32_t h,
    double eps, uint32_t B)
      : w_(w),h_(h), eps_(eps), B_(B), d_depthU16(h_,w_),
      d_depth(h_,w_), d_dSum(h_+1,w_+1), d_dSumT(h_+1,w_+1),
      d_dSqSum(h_+1,w_+1), d_haveData_(h_,w_),d_haveData2(h_,w_),
      tLog("t.log",1), aInt(h_+1,w_+1,CV_64F), bInt(h_+1,w_+1,CV_64F),
      Ns(h_+1,w_+1,CV_32S), dSum(h_+1,w_+1,CV_64F),
      dSqSum(h_+1,w_+1,CV_64F), haveData_(h_, w_, CV_8U), haveData2(h_,
          w_, CV_8U), a(h_, w_, CV_64F), b(h_, w_, CV_64F), dSmooth(h_,
            w_, CV_64F), dFlt(h_,w_,CV_64F), d_depthSmooth(h_,w_),
          d_a(h_,w_), d_b(h_,w_), d_Ns(h_+1,w_+1), d_aInt(h_+1,w_+1),
          d_bInt(h_+1,w_+1) 
{
  cout<<"creating depth filter"<<endl;
  stream1 = jsc::GpuMatrix<double>::createStream();
  stream2 = jsc::GpuMatrix<double>::createStream();

  cout<<"init gpu matrices"<<endl;
  d_depth.setZero();
  d_dSum.setZero();
  d_dSumT.setZero();
  d_dSqSum.setZero();
  d_haveData_.setZero();
  d_haveData2.setZero();

  d_a.setZero();
  d_b.setZero();
  d_aInt.setZero();
  d_bInt.setZero();
  d_Ns.setZero();
  d_depthSmooth.setZero();
};

template<typename T>
DepthGuidedFilterGpu<T>::~DepthGuidedFilterGpu()
{
  jsc::GpuMatrix<double>::deleteStream(stream2);
  jsc::GpuMatrix<double>::deleteStream(stream1);
};

template<typename T>
void DepthGuidedFilterGpu<T>::filter(const cv::Mat& depth)
{
  assert( w_ == depth.cols);
  assert( h_ == depth.rows);

  tLog.tic(0);
  // convert uint16_t to float and find locations with data 
  d_depthU16.set((uint16_t*)depth.data,h_,w_);
  depth2floatGPU(d_depthU16.data(),d_depth.data(),
      d_haveData_.data(),w_,h_,-1);
  d_haveData_.get((uint8_t*)haveData_.data,h_,w_);   
  d_depth.getAsync((double*)dFlt.data,h_,w_,stream1); // copy the double image async sinze it takes longer

  tLog.toctic(0,1);
  cv::integral(haveData_,Ns,CV_32S); 
  d_Ns.setAsync((int32_t*)Ns.data,h_+1,w_+1,stream2);

  cudaStreamSynchronize(stream1);  // wait for stream 1 to finish
  cv::integral(dFlt,dSum,dSqSum,CV_64F); // compute this while haveData_ is still copied
  d_dSum.set((double*)dSum.data,h_+1,w_+1); // start copying those already while we compute the other integral
  d_dSqSum.set((double*)dSqSum.data,h_+1,w_+1);

  tLog.toctic(1,2);

  cudaStreamSynchronize(stream2);  // wait for stream 2 to finish
//  guidedFilter_ab_gpu(d_haveData_.data(),d_haveData2.data(),d_a.data(),d_b.data(),d_Ns.data(),
//  guidedFilter_ab_gpu(d_depth.data(),d_haveData_.data(),
  guidedFilter_ab_gpu(d_haveData_.data(),
      d_haveData2.data(),d_a.data(),d_b.data(),d_Ns.data(),
      d_dSum.data(), d_dSqSum.data(),eps_,B_,w_,h_);
  //    d_haveData2.get((uint8_t*)haveData_.data,h_,w_); // get the valid data after
  //    cv::integral(haveData_,Ns,CV_32S); // TODO hide transfer

  d_a.get((double*)a.data,h_,w_); // important to not getAsync since itll wait till the guided filter is done
  d_b.getAsync((double*)b.data,h_,w_,stream1); // get while computing integral image on a

//  cv::imshow("haveData2",haveData_*255); 
//
// update Ns with new counts TODO: update streams
  d_haveData2.get((uint8_t*)haveData2.data,h_,w_);   
  cv::integral(haveData2,Ns,CV_32S); 
  d_Ns.setAsync((int32_t*)Ns.data,h_+1,w_+1,stream2);
//  cv::imshow("haveData3",haveData2*255);
//  cv::imshow("a",a);
//  cv::waitKey(10);
//  cout<<"a"<<endl;

  tLog.toctic(2,3);
  cv::integral(a,aInt,CV_64F);
  d_aInt.setAsync((double*)aInt.data,h_+1,w_+1, stream2); // aready start copying

  cudaStreamSynchronize(stream1);  // wait for stream 1 to finish
  cv::integral(b,bInt,CV_64F);
  tLog.toctic(3,4);

  d_bInt.set((double*)bInt.data,h_+1,w_+1);
  cudaStreamSynchronize(stream2);  // wait for stream 2 to finish
//  guidedFilter_out_gpu(d_haveData2.data(),d_depth.data(),d_aInt.data(),
  guidedFilter_out_gpu(d_haveData_.data(),d_depth.data(),d_aInt.data(),
      d_bInt.data(),d_Ns.data(),d_depthSmooth.data(),B_,w_,h_);
//      d_bInt.data(),d_Ns.data(),d_depthSmooth.data(),B_/2+(B_%2>0?1:0),w_,h_);
  //    d_depthSmooth.get((T*)dSmooth.data,h_,w_);

  tLog.toc(4);
#ifndef NDEBUG
  tLog.logCycle();
  tLog.printStats();
#endif
  
  //    return dSmooth;
}

template<>
cv::Mat DepthGuidedFilterGpu<float>::getOutput()
{
  //TODO: explicit caching - do not reload all the time
  dSmooth = cv::Mat(h_,w_,CV_32F);
  d_depthSmooth.get((float*)dSmooth.data,h_,w_);
  return dSmooth;
};

template<>
cv::Mat DepthGuidedFilterGpu<double>::getOutput()
{
  //TODO: explicit caching - do not reload all the time
  dSmooth = cv::Mat(h_,w_,CV_64F);
  d_depthSmooth.get((double*)dSmooth.data,h_,w_);
  return dSmooth;
};

//template<typename T>
//cv::Mat DepthGuidedFilterGpu<T>::getOutput()
//{
//  dSmooth = cv::Mat(h_,w_,CV_64F);
//  d_depthSmooth.get((T*)dSmooth.data,h_,w_);
//  return dSmooth;
//};
}
