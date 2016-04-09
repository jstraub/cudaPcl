/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <cudaPcl/openniSmoothDepthGpu.hpp>
#include <cudaPcl/normalExtractSimpleGpu.hpp>
#include <cudaPcl/cv_helpers.hpp>

namespace cudaPcl {

/*
 * SmoothNormalsGpu smoothes the depth frame using a guided filter and
 * computes surface normals from it also on the GPU.
 *
 * Needs the focal length of the depth camera f_d and the parameters for the
 * guided filter eps as well as the filter size B.
 */
class SmoothNormalsGpu 
{
  public:
  SmoothNormalsGpu(double f_d, double eps, uint32_t B, bool
      compress=false)
    : eps_(eps), B_(B), f_d_(f_d),
    depthFilter(NULL), normalExtract(NULL), compress_(compress)
  { };

  virtual ~SmoothNormalsGpu() {
    if(normalExtract) delete normalExtract;
  };

  virtual void depth_cb(const uint16_t * depth, uint32_t w, uint32_t h)
  {
    if(w==0 || h==0) return;
    if(!depthFilter)
    {
      depthFilter = new DepthGuidedFilterGpu<float>(w,h,eps_,B_);
      normalExtract = new NormalExtractSimpleGpu<float>(f_d_,w,h,compress_);
    }
    cv::Mat dMap = cv::Mat(h,w,CV_16U,const_cast<uint16_t*>(depth));

//    Timer t;
    depthFilter->filter(dMap);
//    t.toctic("smoothing");
    normalExtract->computeGpu(depthFilter->getDepthDevicePtr());
//    t.toctic("normals");
    normals_cb(normalExtract->d_normalsImg(), normalExtract->d_haveData(),w,h);
//    t.toctic("normals callback");
    if(compress_)
    {
      int32_t nComp =0;
      normalsComp_ = normalExtract->normalsComp(nComp);
      std::cout << "# compressed normals " << nComp << std::endl;
    }
  };

  /* callback with smoothed normals
   *
   * Note that the pointers are to GPU memory as indicated by the "d_" prefix.
   */
  virtual void normals_cb(float* d_normalsImg, uint8_t* d_haveData,
      uint32_t w, uint32_t h)
  {
    if(w==0 || h==0) return;
    normalsImg_ = normalExtract->normalsImg();

    if(false)
    {
      static int frameN = 0;
      if(frameN==0) if(system("mkdir ./normals/") >0){
        cout<<"problem creating subfolder for results"<<endl;
      };

      char path[100];
      // Save the image data in binary format
      sprintf(path,"./normals/%05d.bin",frameN ++);
      if(compress_)
      {
        int nComp;
        normalsComp_ = normalExtract->normalsComp(nComp);
        imwriteBinary(std::string(path), normalsComp_);
      }else
        imwriteBinary(std::string(path), normalsImg_);
    }
  };

  protected:
  double eps_;
  uint32_t B_;
  double f_d_;
  DepthGuidedFilterGpu<float> * depthFilter;
  NormalExtractSimpleGpu<float> * normalExtract;
  bool compress_;
  cv::Mat normalsImg_;
  cv::Mat normalsComp_;
};

}
