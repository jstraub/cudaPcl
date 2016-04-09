/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cudaPcl/smoothNormalsGpu.hpp>
#include <rgbdGrabber/realSenseGrabber.hpp>
#include <rgbdGrabber/rgbdGrabberHelpers.hpp>

namespace cudaPcl {

/*
 */
class RealSenseSmoothNormalsGpu : public rgbdGrabber::RealSenseGrabber,
  public SmoothNormalsGpu 
{
  public:
  RealSenseSmoothNormalsGpu(double f_d, double eps, uint32_t B, bool
      compress=false)
    : rgbdGrabber::RealSenseGrabber(640,480,60), SmoothNormalsGpu(f_d,
        eps, B, compress)
  { };

  virtual ~RealSenseSmoothNormalsGpu() { };

  virtual void rgbd_cb(const uint8_t* rgb, const uint16_t * depth)
  {
    depth_cb(depth, w_, h_);

    cv::Mat dMap = cv::Mat(h_,w_,CV_16U,const_cast<uint16_t*>(depth));
    cv::Mat rgbMap = cv::Mat(h_,w_,CV_8UC3,const_cast<uint8_t*>(rgb));
    cv::Mat dRawColor = rgbdGrabber::colorizeDepth(dMap, 30.,4000.);

    cv::Mat dSmooth = depthFilter->getOutput();
    cv::Mat dSmoothColor = rgbdGrabber::colorizeDepth(dSmooth, 0.03,4.);

    cv::imshow("n", normalsImg_);
    cv::imshow("rgb", rgbMap);
    cv::imshow("dRaw", dRawColor);
    cv::imshow("dSmooth", dSmoothColor);
    cv::waitKey(1);
  };

  protected:
};

}

