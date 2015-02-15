/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once
#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/io/openni_camera/openni_image.h>

#include <opencv2/core/core.hpp>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

/*
 * Class for obtaining frames via openni via two callbacks: depth_cb as well as
 * rgb_cb
 */
class OpenniGrabber
{
  public:
    OpenniGrabber() 
    {
      // create a new grabber for OpenNI devices
      interface_ = new pcl::OpenNIGrabber();
    };
    virtual ~OpenniGrabber()
    {
      delete interface_;
    };
   
    virtual void depth_cb(const uint16_t * depth, uint32_t w, uint32_t h)
    {

    };  

    virtual void rgb_cb(const uint8_t* rgb, uint32_t w, uint32_t h)
    {

    };  

    virtual void run ()
    {
      this->run_impl();
      while (42) boost::this_thread::sleep (boost::posix_time::seconds (1));
      this->run_cleanup_impl();
    }

protected:
    void d_cb_ (const boost::shared_ptr<openni_wrapper::DepthImage>& d)
    {
//      cout<<"depth "<<d->getFrameID()<< " @"<<d->getTimeStamp()
//        << " size: "<<d->getWidth()<<"x"<<d->getHeight()
//        <<" focal length="<<d->getFocalLength()<<endl;
      w_=d->getWidth();
      h_=d->getHeight();
      const uint16_t* data = d->getDepthMetaData().Data();
      depth_cb(data,w_,h_);
    };

    virtual void rgb_cb_ (const boost::shared_ptr<openni_wrapper::Image>& rgb)
    {
//      cout<<"rgb "<<rgb->getFrameID()<< " @"<<rgb->getTimeStamp()
//        << " size: "<<rgb->getWidth()<<"x"<<rgb->getHeight()<<" px format:"
//        << rgb->getMetaData().PixelFormat()<<endl;
      int w = rgb->getWidth(); 
      int h = rgb->getHeight(); 
      // TODO: uggly .. but avoids double copy of the image.
//      boost::mutex::scoped_lock updateLock(updateModelMutex);
      if(rgb_.cols < w)
      {
        rgb_ = cv::Mat(h,w,CV_8UC3);
      }
      rgb->fillRGB(w,h,rgb_.data);
      rgb_cb(rgb_.data,w,h);
//      updateLock.unlock();
    }

    void run_impl ()
    {
      boost::function<void (const boost::shared_ptr<openni_wrapper::DepthImage>&)>
        f_d = boost::bind (&OpenniGrabber::d_cb_, this, _1);
      boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&)> 
        f_rgb = boost::bind (&OpenniGrabber::rgb_cb_, this, _1);
      // connect callback function for desired signal. 
      boost::signals2::connection c_d = interface_->registerCallback (f_d);
      boost::signals2::connection c_rgb = interface_->registerCallback (f_rgb);
      // start receiving point clouds
      interface_->start ();
    }
    void run_cleanup_impl()
    {
      // stop the grabber
      interface_->stop ();
    }
    cv::Mat rgb_;
    uint32_t w_,h_;
  private:
    pcl::Grabber* interface_;
};
