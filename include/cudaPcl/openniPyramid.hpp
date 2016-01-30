/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <jsCore/timer.hpp>
#include <cudaPcl/openniGrabber.hpp>
#include <cudaPcl/openniVisualizer.hpp>

#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/io/openni_camera/openni_image.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

namespace cudaPcl {

/*
 * Class for obtaining frames via openni via two callbacks: depth_cb as well as
 * rgb_cb
 */
class OpenniPyramid : public OpenniGrabber
{
 public:
  /// Use PyrDown function for downscaling -- convolves with a Gaussian
  /// kernel.
  static const int32_t MODE_GAUSS = 0;
  /// Use resize function for downscaling -- performs linear
  /// interpolation.
  static const int32_t MODE_LINEAR = 1;
  OpenniPyramid(int32_t L, int32_t mode) : OpenniGrabber(), L_(L), mode_(mode) {};
  virtual ~OpenniPyramid() {};

  virtual void depth_cb(const uint16_t * depth, uint32_t w, uint32_t h);

  static cv::Mat createPyramid(cv::Mat& in, int32_t L, int32_t mode,
      std::vector<cv::Mat>& pyr);
 protected:
  int32_t L_;
  int32_t mode_;
  cv::Mat d_;
  cv::Mat d_pyr_;
  std::vector<cv::Mat> d_pyr_lvls_;
  cv::Mat gray_;
  cv::Mat gray_pyr_;
  std::vector<cv::Mat> gray_pyr_lvls_;

  virtual void d_cb_ (const boost::shared_ptr<openni_wrapper::DepthImage>& d);
  virtual void rgb_cb_ (const boost::shared_ptr<openni_wrapper::Image>& rgb);
 private:
};

cv::Mat OpenniPyramid::createPyramid(cv::Mat& in, int32_t L, int32_t mode,
    std::vector<cv::Mat>& pyr)
{
  const uint32_t w = in.cols;
  const uint32_t h = in.rows;
  const uint32_t W = static_cast<uint32_t>(floor(w * 2.*(1.-pow(2.,-L))));
  std::cout << w<< "x"<<h << " L=" << L << " W " << W << std::endl;
  cv::Mat I = cv::Mat::zeros(h, W, in.type());
  uint32_t w0l = 0;
  uint32_t hl = h;
  uint32_t wl = w;
  cv::Mat roi(I,cv::Range(0,hl),cv::Range(w0l,w0l+wl));
  pyr.push_back(roi);
  in.copyTo(roi);
  for (int32_t lvl=1; lvl < L; ++lvl) {
    w0l += wl;
    wl /= 2;
    hl /= 2;
    std::cout << w0l << ", " << wl << "x" << hl << std::endl;
    cv::Mat roiPrev(I,cv::Range(0,hl*2),cv::Range(w0l-wl*2,w0l));
    cv::Mat roiNext(I,cv::Range(0,hl),cv::Range(w0l,w0l+wl));
    if (mode == MODE_GAUSS) {
      cv::pyrDown(roiPrev, roiNext);
    } else if (mode == MODE_LINEAR) {
      cv::resize(roiPrev, roiNext, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    } else {
      std::cout << "INVALID mode for pyramid building: " << mode << std::endl;
    }
    pyr.push_back(roiNext);
  }
  return I;
}

void OpenniPyramid::d_cb_ (const boost::shared_ptr<openni_wrapper::DepthImage>& d)
{
  this->w_=d->getWidth();
  this->h_=d->getHeight();
  const uint16_t* data = d->getDepthMetaData().Data();

  jsc::Timer t0;
  cv::Mat dU16 = cv::Mat(this->h_,this->w_,CV_16U,const_cast<uint16_t*>(data));
  dU16.convertTo(d_, CV_32FC1, 0.001);
  t0.toctic("convert to float");
  d_pyr_ = createPyramid(d_, L_, mode_, d_pyr_lvls_);
  t0.toctic("pyramid creation");

  depth_cb(data,w_,h_);
};

void OpenniPyramid::rgb_cb_ (const boost::shared_ptr<openni_wrapper::Image>& rgb)
{
  int w = rgb->getWidth(); 
  int h = rgb->getHeight(); 
  if(this->rgb_.cols < w)
    this->rgb_ = cv::Mat(h,w,CV_8UC3);
  rgb->fillRGB(w,h,this->rgb_.data);

  jsc::Timer t0;
  cv::cvtColor(this->rgb_, gray_, CV_BGR2GRAY);
  t0.toctic("convert to gray");
  gray_pyr_ = createPyramid(gray_, L_, mode_, gray_pyr_lvls_);
  t0.toctic("pyramid creation");

  rgb_cb(this->rgb_.data,w,h);
}

void OpenniPyramid::depth_cb(const uint16_t * depth, uint32_t w, uint32_t h)
{
  cv::Mat dC_pyr = OpenniVisualizer::colorizeDepth(d_pyr_,0.4,4.); 
  if(dC_pyr.cols > w)
    cv::imshow("pyr", dC_pyr); 
  if(gray_pyr_.cols > w)
    cv::imshow("graypyr", gray_pyr_); 
  cv::waitKey(1);
}

}
