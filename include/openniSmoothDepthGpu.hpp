/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <iostream>
#include <stdint.h>
#include <math.h>

#include <openniVisualizer.hpp>
#include <depthGuidedFilter.hpp>

using std::cout;
using std::endl;

/*
 * OpenniSmoothDepthGpu smoothes the depth frame using a guided filter making
 * use of a GPU in less than 10ms.
 */
class OpenniSmoothDepthGpu : public OpenniVisualizer
{
  public:
  OpenniSmoothDepthGpu(double eps, uint32_t B) : OpenniVisualizer(), eps_(eps),
    B_(B), depthFilter(NULL)
  { };

  virtual ~OpenniSmoothDepthGpu() 
  {
    if(depthFilter) delete depthFilter;
  };

  virtual void depth_cb(const uint16_t * depth, uint32_t w, uint32_t h)
  {
    if(!depthFilter)
      depthFilter = new DepthGuidedFilterGpu<float>(w,h,eps_,B_);

    cv::Mat dMap = cv::Mat(h,w,CV_16U,const_cast<uint16_t*>(depth));
    cv::Mat dColor = colorizeDepth(dMap,30,4000);

    Timer t;
    depthFilter->filter(dMap);
    cv::Mat dSmooth = depthFilter->getOutput();
    t.toctic("smoothing");
    cv::Mat dSmoothColor = colorizeDepth(dSmooth,0.03,4.0);

    boost::mutex::scoped_lock updateLock(updateModelMutex);
    this->dColor_ = cv::Mat(h,w*2,CV_8UC3);
    cv::Mat left(this->dColor_, cv::Rect(0,0,w,h)); // ROI constructor
    cv::Mat right(this->dColor_, cv::Rect(w,0,w,h)); // ROI constructor
    dColor.copyTo(left);
    dSmoothColor.copyTo(right);
    this->update_=true;
  };  

  protected:
  double eps_;
  uint32_t B_;
  DepthGuidedFilterGpu<float> * depthFilter;

};

