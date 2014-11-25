/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once
#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/io/openni_camera/openni_image.h>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

#include <openniGrabber.hpp>

#pragma GCC system_header
#include <pcl/visualization/cloud_viewer.h>

using std::cout;
using std::endl;

/*
 * OpenniVisualizer visualizes the RGB and depth frame and adds a visualizer
 * for a point-cloud using pcl (but does not display anything). 
 *
 * Importantly all visualization is handled in a separate thread.
 */
class OpenniVisualizer : public OpenniGrabber
{
public:
  OpenniVisualizer() : OpenniGrabber(), update_(false), pc_(new pcl::PointCloud<pcl::PointXYZRGB>(1,1))
  {};

  virtual ~OpenniVisualizer() 
  {};

  virtual void depth_cb(const uint16_t * depth, uint32_t w, uint32_t h)
  {
    cv::Mat dMap = cv::Mat(h,w,CV_16U,const_cast<uint16_t*>(depth));
    boost::mutex::scoped_lock updateLock(updateModelMutex);
    dColor_ = colorizeDepth(dMap,30.,4000.);
    update_=true;
  };  

  static cv::Mat colorizeDepth(const cv::Mat& dMap, float min, float max);
  static cv::Mat colorizeDepth(const cv::Mat& dMap);
  virtual void run();

protected:
  bool update_;
  boost::mutex updateModelMutex;
  cv::Mat dColor_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_;

  virtual void visualize_();
  virtual void visualizeRGB();
  virtual void visualizeD();
  virtual void visualizePc();

  virtual void rgb_cb_ (const boost::shared_ptr<openni_wrapper::Image>& rgb)
  {
    // overwrite to use the lock to update rgb;
    int w = rgb->getWidth(); 
    int h = rgb->getHeight(); 
    // TODO: uggly .. but avoids double copy of the image.
    boost::mutex::scoped_lock updateLock(updateModelMutex);
    if(this->rgb_.cols < w)
    {
      this->rgb_ = cv::Mat(h,w,CV_8UC3);
    }
    rgb->fillRGB(w,h,this->rgb_.data);
    rgb_cb(this->rgb_.data,w,h);
//    update_=true; // only depth updates! otherwise we get weird artifacts.
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
private:
  void visualizerThread();
};

// ------------------------------------ impl ---------------------------------

void OpenniVisualizer::visualize_()
{
  visualizeRGB();
  visualizeD();
  visualizePc();
};

void OpenniVisualizer::visualizeRGB()
{
  if (this->rgb_.rows > 0 && this->rgb_.cols > 0)
    cv::imshow("rgb",this->rgb_);
};

void OpenniVisualizer::visualizeD()
{
  if (dColor_.rows > 0 && dColor_.cols > 0)
    cv::imshow("d",dColor_);
};

void OpenniVisualizer::visualizePc()
{
  if(!viewer_->updatePointCloud(pc_, "pc"))
    viewer_->addPointCloud(pc_, "pc");
}

void OpenniVisualizer::visualizerThread()
{
  // prepare visualizer named "viewer"
  viewer_ = boost::shared_ptr<pcl::visualization::PCLVisualizer>(
      new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer_->initCameraParameters ();
  viewer_->setBackgroundColor (255, 255, 255);
  viewer_->addCoordinateSystem (1.0);
  //  viewer_->setPosition(0,0);
  viewer_->setSize(1000,1000);

  while (!viewer_->wasStopped ())
  {
    viewer_->spinOnce (10);
    cv::waitKey(10);
    // Get lock on the boolean update and check if cloud was updated
    boost::mutex::scoped_lock updateLock(updateModelMutex);
    if (update_)
    {
      visualize_();
      update_ = false;
    }
  }
}

void OpenniVisualizer::run ()
{
  boost::thread visualizationThread(&OpenniVisualizer::visualizerThread,this); 
  this->run_impl();
  while (42) boost::this_thread::sleep (boost::posix_time::seconds (1));
  this->run_cleanup_impl();
  visualizationThread.join();
};

cv::Mat OpenniVisualizer::colorizeDepth(const cv::Mat& dMap, float min,
    float max)
{
//  double Min,Max;
//  cv::minMaxLoc(dMap,&Min,&Max);
//  cout<<"min/max "<<min<<" " <<max<<" actual min/max "<<Min<<" " <<Max<<endl;
  cv::Mat d8Bit = cv::Mat::zeros(dMap.rows,dMap.cols,CV_8UC1);
  cv::Mat dColor;
  dMap.convertTo(d8Bit,CV_8UC1, 255./(max-min));
  cv::applyColorMap(d8Bit,dColor,cv::COLORMAP_JET);
  return dColor;
}

cv::Mat OpenniVisualizer::colorizeDepth(const cv::Mat& dMap)
{
  double min,max;
  cv::minMaxLoc(dMap,&min,&max);
//  cout<<" computed actual min/max "<<min<<" " <<max<<endl;
  cv::Mat d8Bit = cv::Mat::zeros(dMap.rows,dMap.cols,CV_8UC1);
  cv::Mat dColor;
  dMap.convertTo(d8Bit,CV_8UC1, 255./(max-min));
  cv::applyColorMap(d8Bit,dColor,cv::COLORMAP_JET);
  return dColor;
}

